"activation based embedding "


import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from utils.prompter import Prompter
except ImportError:
    from LoraRetriever.utils.prompter import Prompter


# Prompter is a utility class to create a prompt for a given input
prompter = Prompter("alpaca")


global_layerwise_index = None
global_embed_backbone = None
global_embed_tokenizer = None
global_embed_device = "cuda" if torch.cuda.is_available() else "cpu"


MAX_EMBED_SEQ_LEN = 256
EMBED_BATCH_SIZE = 4


def load_base_model(model_name_or_path="meta-llama/Llama-2-7b-hf"):
    """
    Load the base model and tokenizer from a given model path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16
    )
    base_model.bfloat16()
    return base_model, tokenizer


def _ensure_embedding_backbone(model_size="7b"):
    global global_embed_backbone, global_embed_tokenizer, global_embed_device
    if global_embed_backbone is not None and global_embed_tokenizer is not None:
        return

    model_name = f"meta-llama/Llama-2-{model_size}-hf"
    global_embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if global_embed_tokenizer.pad_token is None:
        global_embed_tokenizer.pad_token = global_embed_tokenizer.eos_token
    global_embed_tokenizer.padding_side = "right"

    dtype = torch.float16 if global_embed_device == "cuda" else torch.float32
    global_embed_backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    global_embed_backbone = global_embed_backbone.to(global_embed_device)
    global_embed_backbone.eval()
    for p in global_embed_backbone.parameters():
        p.requires_grad = False


def _resolve_lora_path(model_name, model_size="7b"):
    local_candidates = [
        os.path.join(".", "lora_modules", f"llama2_{model_size}_lora-{model_name}"),
        os.path.join(
            ".", "LoraRetriever", "lora_modules", f"llama2_{model_size}_lora-{model_name}"
        ),
    ]
    for candidate in local_candidates:
        if os.path.isdir(candidate):
            return candidate

    if model_size == "7b":
        return f"Styxxxx/llama2_7b_lora-{model_name}"
    return f"Styxxxx/llama2_13b_lora-{model_name}"


def _tokenize_texts(texts, max_len=MAX_EMBED_SEQ_LEN):
    return global_embed_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )


class ActivationCapture:
    """
    Context manager that registers forward hooks to capture input activations
    at specific named modules in the base model.
    """

    def __init__(self, model, layer_keys: List[str]):
        self.model = model
        self.layer_keys = set(layer_keys)
        self.activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._hooks = []

    def _make_hook(self, key):
        def hook(module, inputs, output):
            h = inputs[0].detach().cpu().float()
            self.activations[key].append(h)

        return hook

    def __enter__(self):
        self.activations.clear()
        for name, module in self.model.named_modules():
            if name in self.layer_keys:
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _resolve_adapter_file(lora_path, filename):
    """Resolve adapter file from local path or HF repo id."""
    if os.path.isdir(lora_path):
        local_file = os.path.join(lora_path, filename)
        if os.path.exists(local_file):
            return local_file
        return None

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=lora_path, filename=filename)
    except Exception:
        return None


def _load_lora_state(lora_path):
    """Load raw LoRA state dict from .safetensors or .bin (local or HF)."""
    import safetensors.torch as sf

    weights_file = _resolve_adapter_file(lora_path, "adapter_model.safetensors")
    if weights_file is not None:
        state_dict = sf.load_file(weights_file)
    else:
        weights_file = _resolve_adapter_file(lora_path, "adapter_model.bin")
        if weights_file is None:
            raise FileNotFoundError(f"No adapter_model.safetensors/.bin for {lora_path}")
        state_dict = torch.load(weights_file, map_location="cpu")
    return state_dict


def _extract_layer_ab(lora_path):
    """Return ({layer_name: {'A','B'}}, scaling) where scaling=alpha/r."""
    state_dict = _load_lora_state(lora_path)
    cfg = PeftConfig.from_pretrained(lora_path)
    scaling = cfg.lora_alpha / cfg.r

    grouped = {}
    for key, val in state_dict.items():
        if "lora_A.weight" in key:
            layer = key.replace(".lora_A.weight", "").replace("base_model.model.", "")
            grouped.setdefault(layer, {})["A"] = val.float().cpu()
        elif "lora_B.weight" in key:
            layer = key.replace(".lora_B.weight", "").replace("base_model.model.", "")
            grouped.setdefault(layer, {})["B"] = val.float().cpu()

    complete = {layer: parts for layer, parts in grouped.items() if "A" in parts and "B" in parts}
    return complete, scaling


@torch.no_grad()
def compute_activation_delta(
    h: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: float = 1.0,
) -> torch.Tensor:
    """
    Compute Δh = (h @ A^T) @ B^T * scaling and mean-pool across tokens.
    Returns tensor with shape (batch, d_out).
    """
    h = h.to(A.device)
    after_A = h @ A.T
    delta = after_A @ B.T
    delta = delta * scaling

    if attention_mask is not None:
        mask = attention_mask.to(delta.device).unsqueeze(-1).float()
        delta = (delta * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        delta = delta.mean(dim=1)

    return delta


@torch.no_grad()
def _embed_lora_layer(
    texts,
    layer_key,
    A,
    B,
    scaling,
    batch_size=EMBED_BATCH_SIZE,
    max_seq_len=MAX_EMBED_SEQ_LEN,
):
    """
    Activation-delta embedding from the notebook:
    capture layer input h, compute LoRA delta in activation space, then average.
    """
    if not texts:
        return np.zeros(B.shape[0], dtype=np.float32)

    A = A.cpu()
    B = B.cpu()
    all_deltas = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = _tokenize_texts(batch_texts, max_len=max_seq_len)
        inputs_dev = {k: v.to(global_embed_device) for k, v in inputs.items()}

        with ActivationCapture(global_embed_backbone, [layer_key]) as caps:
            global_embed_backbone(**inputs_dev)

        if layer_key not in caps.activations or len(caps.activations[layer_key]) == 0:
            continue

        h = caps.activations[layer_key][0].cpu()
        mask = inputs.get("attention_mask")
        mask = mask.cpu() if mask is not None else None

        delta = compute_activation_delta(
            h=h,
            A=A,
            B=B,
            attention_mask=mask,
            scaling=scaling,
        )
        all_deltas.append(delta)

    if not all_deltas:
        return np.zeros(B.shape[0], dtype=np.float32)

    all_deltas = torch.cat(all_deltas, dim=0)
    mean_delta = all_deltas.mean(dim=0)
    norm = float(mean_delta.norm(p=2))
    if norm <= 1e-12:
        return np.zeros_like(mean_delta.numpy(), dtype=np.float32)

    normed = F.normalize(mean_delta, dim=0)
    return normed.numpy().astype(np.float32)


def build_layerwise_lora_index(lora_index, blend=0.35):
    """
    Build layer-wise index using activation-delta layer embeddings.
    `blend` is kept for API compatibility with mainlayerwise1.py.
    """
    _ = blend
    layerwise_buckets = {}

    for _, info in lora_index.items():
        lora_path = info["lora_path"]
        texts = info["sample_texts"]
        layer_weights, scaling = _extract_layer_ab(lora_path)

        for layer_name, parts in layer_weights.items():
            layer_emb = _embed_lora_layer(
                texts=texts,
                layer_key=layer_name,
                A=parts["A"],
                B=parts["B"],
                scaling=scaling,
            )
            if np.linalg.norm(layer_emb) <= 1e-12:
                continue

            layerwise_buckets.setdefault(layer_name, []).append(
                {
                    "lora_path": lora_path,
                    "embedding": layer_emb,
                }
            )

    layerwise_index = {}
    for layer_name, items in layerwise_buckets.items():
        layerwise_index[layer_name] = {
            "lora_paths": [it["lora_path"] for it in items],
            "matrix": np.stack([it["embedding"] for it in items], axis=0),
        }

    print(f"Built layer-wise index for {len(layerwise_index)} layers")
    return layerwise_index


def initialize_index(models, model_size="7b", blend=0.35):
    """
    Initialize layer-wise retrieval index.
    Same model/config interface as original initialize_index.
    """
    global global_layerwise_index
    _ensure_embedding_backbone(model_size=model_size)

    lora_index = {}

    for model in models:
        lora_path = _resolve_lora_path(model["model_name"], model_size=model_size)
        sample_texts = [sample["inputs"] for sample in model["sample"]]
        if len(sample_texts) == 0:
            continue

        lora_index[model["model_name"]] = {
            "sample_texts": sample_texts,
            "lora_path": lora_path,
        }

    global_layerwise_index = build_layerwise_lora_index(lora_index=lora_index, blend=blend)


@torch.no_grad()
def _compute_query_layer_embeddings(query_text, layer_keys):
    """
    Query embedding from notebook method: capture base hidden state per layer,
    mean-pool over non-padding tokens, then L2-normalize.
    """
    inputs = _tokenize_texts([query_text], max_len=MAX_EMBED_SEQ_LEN)
    inputs_dev = {k: v.to(global_embed_device) for k, v in inputs.items()}
    mask = inputs["attention_mask"].float()

    with ActivationCapture(global_embed_backbone, layer_keys) as caps:
        global_embed_backbone(**inputs_dev)

    query_embs = {}
    for layer_key in layer_keys:
        if layer_key not in caps.activations or len(caps.activations[layer_key]) == 0:
            continue
        h = caps.activations[layer_key][0].cpu()
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        normed = F.normalize(pooled[0], dim=0).numpy().astype(np.float32)
        query_embs[layer_key] = normed

    return query_embs


def _retrieve_topk_per_layer(query_text, layerwise_index, layer_k=1):
    query_embs = _compute_query_layer_embeddings(query_text, list(layerwise_index.keys()))

    hits = {}
    for layer_name, bucket in layerwise_index.items():
        if layer_name not in query_embs:
            continue
        q = query_embs[layer_name]
        sims = np.dot(bucket["matrix"], q)
        top_idx = np.argsort(sims)[::-1][:layer_k]
        hits[layer_name] = [
            {
                "lora_path": bucket["lora_paths"][int(idx)],
                "score": float(sims[int(idx)]),
            }
            for idx in top_idx
        ]
    return hits


def perform_search(query_list, k=20, exclude_list=None, layer_top_k=1):
    """
    Layer-wise retrieval entrypoint compatible with original API.

    Returns:
      - all_results_list: unique selected adapters across query_list
      - mapping_matrix: binary matrix [batch, len(all_results_list)]
    """
    global global_layerwise_index

    all_results_set = set()
    selected_per_query = []

    for j, query in enumerate(query_list):
        exclude_item = exclude_list[j] if exclude_list else None
        layer_hits = _retrieve_topk_per_layer(
            query, global_layerwise_index, layer_k=layer_top_k
        )

        score_by_adapter = defaultdict(float)
        for _, per_layer_hits in layer_hits.items():
            for hit in per_layer_hits:
                adapter = hit["lora_path"]
                if exclude_item is not None and adapter == exclude_item:
                    continue
                score_by_adapter[adapter] += hit["score"]

        ranked = sorted(score_by_adapter.items(), key=lambda x: x[1], reverse=True)
        chosen = [name for name, _ in ranked[:k]]

        all_results_set.update(chosen)
        selected_per_query.append(chosen)

    all_results_list = list(all_results_set)
    mapping_matrix = []
    for chosen in selected_per_query:
        chosen_set = set(chosen)
        mapping_vector = [1 if result in chosen_set else 0 for result in all_results_list]
        mapping_matrix.append(mapping_vector)

    return all_results_list, mapping_matrix


def init_vector_db(config_path="./config/config2.json"):
    """
    Initialize the vector database with configurations from the specified JSON file.
    """
    resolved_config = config_path
    if not os.path.exists(resolved_config):
        fallback = os.path.join(".", "LoraRetriever", "config", "config2.json")
        if os.path.exists(fallback):
            resolved_config = fallback

    with open(resolved_config, "r") as file:
        lora_configs = json.load(file)

    initialize_index(lora_configs)


def load_peft_model(lora_module_list, base_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_lists = []

    for i, lora_model in enumerate(lora_module_list):
        print(f"\nLoading adapter {i}: {lora_model}")
        if i == 0:
            peft_model = PeftModel.from_pretrained(base_model, lora_model, f"adapter{i}")
        else:
            peft_model.load_adapter(lora_model, f"adapter{i}")
        lora_lists.append(f"adapter{i}")

        print(f"Adapter config: {peft_model.peft_config[f'adapter{i}']}")

    print(f"\nSetting adapters: {lora_lists}")
    peft_model.set_adapter(lora_lists)

    print(f"Active adapters after set_adapter: {peft_model.active_adapters}")

    for name, module in peft_model.named_modules():
        if "q_proj" in name and hasattr(module, "lora_A"):
            print(f"\n{name}:")
            print(f"  lora_A type: {type(module.lora_A)}")
            if isinstance(module.lora_A, dict):
                print(f"  Keys in lora_A: {module.lora_A.keys()}")
                for key in module.lora_A.keys():
                    if module.lora_A[key] is not None:
                        weight_shape = (
                            module.lora_A[key].weight.shape
                            if hasattr(module.lora_A[key], "weight")
                            else "No weight attr"
                        )
                        print(f"    {key}: {weight_shape}")
            break

    peft_model = peft_model.to(device)
    peft_model.eval()
    return peft_model


def check_adapter_compatibility(lora_module_list):
    """Check if all adapters have the same target modules"""
    configs = []
    target_modules = []

    for lora_model in lora_module_list:
        config = PeftConfig.from_pretrained(lora_model)
        configs.append(config)
        print(f"\n{lora_model}:")
        print(f"  Target modules: {config.target_modules}")
        print(f"  LoRA rank (r): {config.r}")
        print(f"  LoRA alpha: {config.lora_alpha}")

        if isinstance(config.target_modules, set):
            target_modules.append(config.target_modules)
        elif isinstance(config.target_modules, list):
            target_modules.append(set(config.target_modules))
        else:
            target_modules.append({config.target_modules})

    unique_targets = set(frozenset(tm) for tm in target_modules)

    if len(unique_targets) > 1:
        print("\n⚠️ WARNING: Adapters have different target modules!")
        for i, tm in enumerate(target_modules):
            print(f"  Adapter {i}: {tm}")
    else:
        print("\n✓ All adapters target the same modules")
        print(f"  Common target modules: {target_modules[0]}")

    ranks = [cfg.r for cfg in configs]
    alphas = [cfg.lora_alpha for cfg in configs]

    if len(set(ranks)) > 1:
        print("\n⚠️ WARNING: Adapters have different LoRA ranks!")
        for i, r in enumerate(ranks):
            print(f"  Adapter {i}: rank={r}")

    if len(set(alphas)) > 1:
        print("\n⚠️ WARNING: Adapters have different LoRA alphas!")
        for i, alpha in enumerate(alphas):
            print(f"  Adapter {i}: alpha={alpha}")

    return configs


def eval_datasets(
    data_path,
    res_path,
    config_path="config/config2.json",
    eval_type="fusion",
    lora_num=3,
    batch_size=1,
    ood=False,
    best_selection=False,
    model_size="7b",
    seed=None,
    layer_top_k=1,
):
    """
    Evaluate the model on given datasets.

    Parameters:
    - data_path: Path to the evaluation dataset.
    - res_path: Path to save the evaluation results.
    - config_path: Path to configuration file for retrieval initialization.
    - eval_type: The merging type for LoRA adapters (e.g., 'fusion').
    - lora_num: Number of LoRA adapters to be retrieved.
    - batch_size: Batch size for evaluation.
    - ood: Flag indicating if out-of-domain exclusion should be applied.
    - best_selection: If True, use the most appropriate LoRA for each input.
    - model_size: Model size of Llama-2.
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_vector_db(config_path)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["inputs"],
            "",
            "",
        )
        return {"full_prompt": full_prompt}

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path)
    else:
        dataset = load_dataset(data_path)

    eval_data = dataset["train"].map(generate_and_tokenize_prompt)

    model_path = f"meta-llama/Llama-2-{model_size}-hf"
    base_model, tokenizer = load_base_model(model_path)
    base_model.eval()

    with torch.no_grad():
        with tqdm(total=len(dataset["train"]), desc="Evaluating", unit="item") as pbar:
            for i in range(0, len(eval_data["full_prompt"]), batch_size):
                input_text = eval_data["inputs"][i : i + batch_size]
                task_names = eval_data["task"][i : i + batch_size]

                exclude_list = None
                if ood:
                    if model_size == "7b":
                        exclude_list = [f"Styxxxx/llama2_7b_lora-{task}" for task in task_names]
                    else:
                        exclude_list = [f"Styxxxx/llama2_13b_lora-{task}" for task in task_names]

                module_list, mapping_matrix = perform_search(
                    input_text,
                    k=lora_num,
                    exclude_list=exclude_list,
                    layer_top_k=layer_top_k,
                )
                input_text = eval_data["full_prompt"][i : i + batch_size]

                if best_selection:
                    if model_size == "7b":
                        exclude_list = [f"Styxxxx/llama2_7b_lora-{task}" for task in task_names]
                    else:
                        exclude_list = [f"Styxxxx/llama2_13b_lora-{task}" for task in task_names]

                    unique_items = list(set(exclude_list))
                    item_to_index = {item: idx for idx, item in enumerate(unique_items)}
                    mapping_matrix = np.zeros((len(exclude_list), len(unique_items)), dtype=int)
                    module_list = unique_items
                    for item_idx, item in enumerate(exclude_list):
                        mapping_matrix[item_idx, item_to_index[item]] = 1

                print("module_list:", module_list)
                if mapping_matrix is None:
                    raise ValueError(
                        "mapping_matrix is None. Retrieval may have failed or returned no adapters."
                    )
                mapping_matrix_tensor = torch.tensor(mapping_matrix).to(device)
                print("mapping_matrix_tensor.shape:", mapping_matrix_tensor.shape)
                print("Number of adapters loaded:", len(module_list))
                if mapping_matrix_tensor.shape[1] != len(module_list):
                    raise ValueError(
                        f"Shape mismatch: mapping_matrix_tensor.shape[1] ({mapping_matrix_tensor.shape[1]}) "
                        f"!= number of adapters ({len(module_list)}). Please check retrieval logic."
                    )
                mapping_matrix_tensor = mapping_matrix_tensor.to(torch.bfloat16)
                mapping_matrix_tensor /= lora_num
                if not torch.isfinite(mapping_matrix_tensor).all():
                    raise ValueError(
                        "lora_mapping contains NaN/inf values; check retrieval/mapping normalization."
                    )
                if (mapping_matrix_tensor < 0).any():
                    raise ValueError(
                        "lora_mapping contains negative values; check retrieval/mapping normalization."
                    )

                _ = check_adapter_compatibility(module_list)
                peft_model = load_peft_model(module_list, base_model)

                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                try:
                    outputs = peft_model.generate(
                        input_ids=inputs["input_ids"],
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=1.0,
                        merging_type=eval_type,
                        lora_mapping=mapping_matrix_tensor,
                    )
                except Exception as e:
                    print("exception", e)
                    continue

                for j, (output, expected_answer) in enumerate(
                    zip(outputs, eval_data["targets"][i : i + batch_size])
                ):
                    generated_answer = tokenizer.decode(output, skip_special_tokens=True)
                    generated_answer = generated_answer.strip().split("### Response:\n")[-1]

                    sample = {
                        "inputs": eval_data["inputs"][i + j],
                        "targets": eval_data["targets"][i + j],
                        "metric": eval_data["metric"][i + j],
                        "domain": eval_data["domain"][i + j],
                        "task": eval_data["task"][i + j],
                        "predicted_answer": generated_answer,
                    }
                    results.append(sample)

                    print(
                        f"generated_answer: {generated_answer}, expected_answer: {expected_answer}"
                    )

                pbar.set_description("Evaluating")
                peft_model.unload()

    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import fire

    fire.Fire(eval_datasets)
