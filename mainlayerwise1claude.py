import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from InstructorEmbedding import INSTRUCTOR

try:
    from utils.prompter import Prompter
except ImportError:
    from LoraRetriever.utils.prompter import Prompter


prompter = Prompter("alpaca")

INSTRUCTION_PREFIX = "Represent the sentence for similar task retrieval: "
global_embed_model = None
global_layerwise_index = None

# "per_layer" is the new true per-layer composition method
SUPPORTED_COMPOSITION_METHODS = ["fusion", "mixture", "per_layer"]


def load_base_model(model_name_or_path="meta-llama/Llama-2-7b-hf"):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16
    )
    base_model.bfloat16()
    return base_model, tokenizer


def _encode_with_instruction(text_list, instruction=INSTRUCTION_PREFIX):
    global global_embed_model
    pairs = [[instruction, text] for text in text_list]
    emb = global_embed_model.encode(pairs)
    return np.asarray(emb, dtype=np.float32)


def _resolve_adapter_file(lora_path, filename):
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


def _extract_layer_deltas(lora_path):
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

    deltas = {}
    for layer, parts in grouped.items():
        if "A" in parts and "B" in parts:
            deltas[layer] = scaling * (parts["B"] @ parts["A"])
    return deltas


def _delta_to_fixed_embedding(delta_w, emb_dim=768):
    flat = delta_w.reshape(-1)
    if flat.numel() == 0:
        return np.zeros(emb_dim, dtype=np.float32)
    v = F.adaptive_avg_pool1d(flat.abs().unsqueeze(0).unsqueeze(0), emb_dim).squeeze()
    v = v / (v.norm(p=2) + 1e-12)
    return v.numpy().astype(np.float32)


def build_layerwise_lora_index(lora_index, emb_dim=768, blend=0.35):
    layerwise_buckets = {}
    for _, info in lora_index.items():
        task_emb = np.asarray(info["embedding"], dtype=np.float32)
        task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-12)
        lora_path = info["lora_path"]

        deltas = _extract_layer_deltas(lora_path)
        for layer_name, delta_w in deltas.items():
            layer_sig = _delta_to_fixed_embedding(delta_w, emb_dim=emb_dim)
            layer_emb = (1.0 - blend) * task_emb + blend * layer_sig
            layer_emb = layer_emb / (np.linalg.norm(layer_emb) + 1e-12)
            layerwise_buckets.setdefault(layer_name, []).append(
                {"lora_path": lora_path, "embedding": layer_emb}
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
    global global_embed_model, global_layerwise_index
    global_embed_model = INSTRUCTOR("Styxxxx/lora_retriever")

    lora_index = {}
    emb_dim = None

    for model in models:
        if model_size == "7b":
            lora_path = f"Styxxxx/llama2_7b_lora-{model['model_name']}"
        else:
            lora_path = f"Styxxxx/llama2_13b_lora-{model['model_name']}"

        sample_texts = [sample["inputs"] for sample in model["sample"]]
        embeddings = _encode_with_instruction(sample_texts)
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-12)

        if emb_dim is None:
            emb_dim = avg_embedding.shape[0]

        lora_index[model["model_name"]] = {
            "embedding": avg_embedding,
            "lora_path": lora_path,
        }

    global_layerwise_index = build_layerwise_lora_index(
        lora_index=lora_index,
        emb_dim=emb_dim if emb_dim is not None else 768,
        blend=blend,
    )


def _retrieve_topk_per_layer(query_text, layerwise_index, layer_k=1):
    q = _encode_with_instruction([query_text])[0].astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)

    hits = {}
    for layer_name, bucket in layerwise_index.items():
        sims = cosine_similarity(q[None, :], bucket["matrix"])[0]
        top_idx = np.argsort(sims)[::-1][:layer_k]
        hits[layer_name] = [
            {
                "lora_path": bucket["lora_paths"][int(idx)],
                "score": float(sims[int(idx)]),
            }
            for idx in top_idx
        ]
    return hits


def _scores_to_normalized_weights(score_by_adapter, chosen_adapters):
    if not chosen_adapters:
        return []
    scores = np.asarray(
        [max(float(score_by_adapter.get(adapter, 0.0)), 0.0) for adapter in chosen_adapters],
        dtype=np.float32,
    )
    total = float(scores.sum())
    if total <= 1e-12:
        return [1.0 / len(chosen_adapters)] * len(chosen_adapters)
    return (scores / total).astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# NEW: Build per-layer weight map
# ---------------------------------------------------------------------------

def build_per_layer_weight_map(layer_hits, lora_path_to_adapter_name):
    """
    Convert raw layer_hits into normalized per-layer weights keyed by PEFT adapter name.

    Unlike global weighting (which collapses all layers into one score per adapter),
    this keeps a separate weight distribution for every transformer layer.

    Args:
        layer_hits: {layer_name: [{"lora_path": ..., "score": ...}]}
            Returned by _retrieve_topk_per_layer.
        lora_path_to_adapter_name: {lora_path: "adapterN"}
            Maps HF repo paths to the PEFT adapter keys (adapter0, adapter1, ...).

    Returns:
        {layer_name: {adapter_name: weight}}

    Example:
        {
          "model.layers.0.self_attn.q_proj": {"adapter0": 0.8, "adapter1": 0.2},
          "model.layers.1.self_attn.v_proj": {"adapter0": 0.1, "adapter1": 0.9},
          ...
        }
    """
    per_layer_weights = {}

    for layer_name, hits in layer_hits.items():
        # Translate lora_path → adapter_name, drop paths not in module_list
        raw = {}
        for hit in hits:
            adapter_name = lora_path_to_adapter_name.get(hit["lora_path"])
            if adapter_name is not None:
                raw[adapter_name] = max(float(hit["score"]), 0.0)

        if not raw:
            continue

        total = sum(raw.values())
        if total > 1e-12:
            per_layer_weights[layer_name] = {k: v / total for k, v in raw.items()}
        else:
            # Uniform fallback
            n = len(raw)
            per_layer_weights[layer_name] = {k: 1.0 / n for k in raw}

    return per_layer_weights


# ---------------------------------------------------------------------------
# NEW: Hook-based per-layer composition
# ---------------------------------------------------------------------------

def apply_per_layer_hooks(peft_model, per_layer_weights, all_adapter_names):
    """
    Register forward hooks that apply a different adapter weight mixture at every
    LoRA layer, enabling true per-layer composition.

    Mechanism
    ---------
    Standard PEFT with multiple active adapters sums each adapter's LoRA delta
    with weight 1.0 (i.e. full contribution from every adapter):

        out = base(x) + sum_i [ lora_i(x) * scaling_i ]   # default, weight=1

    The hook intercepts the output and applies a correction so that each adapter
    uses the per-layer weight instead:

        corrected = out + sum_i [ (w_i - 1.0) * lora_i(x) * scaling_i ]
                  = base(x) + sum_i [ w_i * lora_i(x) * scaling_i ]   # desired

    This avoids any modification to PEFT internals or the generate() call.

    Args:
        peft_model: loaded PeftModel with all adapters active.
        per_layer_weights: {layer_name: {adapter_name: weight}}
            Output of build_per_layer_weight_map.
        all_adapter_names: list of all PEFT adapter keys (e.g. ["adapter0", "adapter1"]).

    Returns:
        List of hook handles — call handle.remove() on each to clean up.
    """
    hooks = []
    stored_inputs = {}  # layer full_name → last input tensor

    def make_pre_hook(full_name):
        """Store the input tensor so the post-hook can recompute LoRA deltas."""
        def pre_hook(module, args):
            stored_inputs[full_name] = args[0]
        return pre_hook

    def make_post_hook(full_name, layer_weights):
        """
        Correct the output so each adapter uses its per-layer weight instead of
        the default weight of 1.0.
        """
        def post_hook(module, args, output):
            x = stored_inputs.get(full_name)
            if x is None:
                return output

            correction = torch.zeros_like(output)

            for adapter_name in all_adapter_names:
                # Skip adapters not present in this layer (some layers may not
                # have been targeted by all adapters).
                if adapter_name not in module.lora_A:
                    continue
                if adapter_name not in module.lora_B:
                    continue

                # Per-layer weight for this adapter; fall back to uniform if
                # this layer was not in the retrieved index.
                default_w = 1.0
                per_layer_w = layer_weights.get(adapter_name, default_w)

                delta = per_layer_w - default_w
                if abs(delta) < 1e-6:
                    continue  # No correction needed — saves compute

                # Re-compute LoRA delta: B( A( dropout(x) ) ) * scaling
                # In eval mode dropout is identity, so this is deterministic.
                scaling = module.scaling.get(adapter_name, 1.0)
                lora_out = (
                    module.lora_B[adapter_name](
                        module.lora_A[adapter_name](
                            module.lora_dropout[adapter_name](x)
                        )
                    )
                    * scaling
                )
                correction = correction + delta * lora_out

            return output + correction

        return post_hook

    # Walk all modules and attach hooks to LoRA Linear layers
    for full_name, module in peft_model.named_modules():
        if not (hasattr(module, "lora_A") and isinstance(module.lora_A, dict)):
            continue

        # Match the full PEFT module name against the index layer names.
        # Index keys look like "model.layers.0.self_attn.q_proj" while PEFT
        # names look like "base_model.model.model.layers.0.self_attn.q_proj".
        matched_layer = None
        for layer_name in per_layer_weights:
            if full_name.endswith(layer_name):
                matched_layer = layer_name
                break

        if matched_layer is None:
            # Layer not in index — use uniform weights (no correction applied)
            layer_weights = {a: 1.0 for a in all_adapter_names}
        else:
            layer_weights = per_layer_weights[matched_layer]

        h_pre = module.register_forward_pre_hook(make_pre_hook(full_name))
        h_post = module.register_forward_hook(make_post_hook(full_name, layer_weights))
        hooks.extend([h_pre, h_post])

    print(f"Registered {len(hooks) // 2} per-layer hook pairs across LoRA modules")
    return hooks


def remove_hooks(hooks):
    """Remove all registered forward hooks."""
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Existing search / retrieval (unchanged)
# ---------------------------------------------------------------------------

def perform_search(
    query_list,
    k=20,
    exclude_list=None,
    layer_top_k=1,
    return_details=False,
):
    global global_layerwise_index

    all_results_set = set()
    selected_per_query = []
    weights_per_query = []
    retrieval_details = []

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
        chosen_weights = _scores_to_normalized_weights(score_by_adapter, chosen)

        all_results_set.update(chosen)
        selected_per_query.append(chosen)
        weights_per_query.append(chosen_weights)

        if return_details:
            retrieval_details.append(
                {
                    "layerwise_hits": layer_hits,
                    "ranked_adapters": [
                        {"lora_path": adapter, "score": float(score)}
                        for adapter, score in ranked
                    ],
                    "selected_adapters": chosen,
                    "selected_weights": chosen_weights,
                }
            )

    all_results_list = sorted(list(all_results_set))
    mapping_matrix = []
    for chosen, chosen_weights in zip(selected_per_query, weights_per_query):
        weight_map = {adapter: weight for adapter, weight in zip(chosen, chosen_weights)}
        mapping_vector = [float(weight_map.get(result, 0.0)) for result in all_results_list]
        mapping_matrix.append(mapping_vector)

    if return_details:
        return all_results_list, mapping_matrix, retrieval_details

    return all_results_list, mapping_matrix


def _resolve_eval_types(eval_type="fusion", eval_types=None):
    if eval_types is None:
        return [eval_type]
    if isinstance(eval_types, str):
        token = eval_types.strip().lower()
        if token in {"all", "auto", "both"}:
            return SUPPORTED_COMPOSITION_METHODS.copy()
        methods = [m.strip() for m in eval_types.split(",") if m.strip()]
    else:
        methods = [str(m).strip() for m in eval_types if str(m).strip()]
    valid = [m for m in methods if m in SUPPORTED_COMPOSITION_METHODS]
    if not valid:
        return [eval_type]
    seen = set()
    ordered = []
    for m in valid:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def init_vector_db(config_path="./config/config2.json"):
    with open(config_path, "r") as file:
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

    peft_model = peft_model.to(device)
    peft_model.eval()
    return peft_model


def check_adapter_compatibility(lora_module_list):
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
    else:
        print("\n✓ All adapters target the same modules")
        print(f"  Common target modules: {target_modules[0]}")

    ranks = [cfg.r for cfg in configs]
    alphas = [cfg.lora_alpha for cfg in configs]
    if len(set(ranks)) > 1:
        print("\n⚠️ WARNING: Adapters have different LoRA ranks!")
    if len(set(alphas)) > 1:
        print("\n⚠️ WARNING: Adapters have different LoRA alphas!")
    return configs


# ---------------------------------------------------------------------------
# Evaluation loop — updated to support "per_layer" composition
# ---------------------------------------------------------------------------

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
    eval_types=None,
):
    """
    Evaluate the model on given datasets.

    Parameters:
    - data_path: Path to the evaluation dataset.
    - res_path: Path to save the evaluation results.
    - config_path: Path to configuration file for retrieval initialization.
    - eval_type: The merging type for LoRA adapters.
                 Now also accepts "per_layer" for true per-layer composition.
    - eval_types: Optional comma-separated/list composition methods.
                  Use 'all' for all supported (fusion, mixture, per_layer).
    - lora_num: Number of LoRA adapters to be retrieved.
    - batch_size: Batch size for evaluation.
                  NOTE: per_layer composition currently supports batch_size=1 only.
    - ood: Flag indicating if out-of-domain exclusion should be applied.
    - best_selection: If True, use the known correct LoRA (upper-bound baseline).
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
    methods_to_run = _resolve_eval_types(eval_type=eval_type, eval_types=eval_types)

    # Warn if per_layer is requested with batch_size > 1
    if "per_layer" in methods_to_run and batch_size > 1:
        print(
            "WARNING: per_layer composition currently supports batch_size=1 only. "
            "Per-layer weights will be taken from the first query in each batch."
        )

    init_vector_db(config_path)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(data_point["inputs"], "", "")
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

                module_list, mapping_matrix, retrieval_details = perform_search(
                    input_text,
                    k=lora_num,
                    exclude_list=exclude_list,
                    layer_top_k=layer_top_k,
                    return_details=True,
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
                    raise ValueError("mapping_matrix is None. Retrieval may have failed.")

                mapping_matrix_tensor = torch.tensor(mapping_matrix).to(device)
                row_sums = mapping_matrix_tensor.sum(dim=1, keepdim=True)
                if (row_sums <= 0).any():
                    raise ValueError("lora_mapping has zero-sum rows; check retrieval.")
                mapping_matrix_tensor = mapping_matrix_tensor / row_sums
                mapping_matrix_tensor = mapping_matrix_tensor.to(torch.bfloat16)

                if not torch.isfinite(mapping_matrix_tensor).all():
                    raise ValueError("lora_mapping contains NaN/inf values.")
                if (mapping_matrix_tensor < 0).any():
                    raise ValueError("lora_mapping contains negative values.")

                # Build per-layer weight map once per batch for use in "per_layer" mode.
                # Maps PEFT adapter keys (adapter0, adapter1, ...) to per-layer weights.
                # We use the first query's layer_hits; see batch_size warning above.
                lora_path_to_adapter_name = {
                    path: f"adapter{idx}" for idx, path in enumerate(module_list)
                }
                all_adapter_names = [f"adapter{idx}" for idx in range(len(module_list))]

                # per_layer_weight_map shape:
                #   {layer_name: {adapter_name: weight}}
                # Different from global mapping_matrix which is [batch, num_adapters].
                first_query_layer_hits = retrieval_details[0]["layerwise_hits"]
                per_layer_weight_map = build_per_layer_weight_map(
                    layer_hits=first_query_layer_hits,
                    lora_path_to_adapter_name=lora_path_to_adapter_name,
                )

                _ = check_adapter_compatibility(module_list)
                peft_model = load_peft_model(module_list, base_model)

                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                for composition_method in methods_to_run:
                    try:
                        if composition_method == "per_layer":
                            # ---------------------------------------------------
                            # TRUE PER-LAYER COMPOSITION
                            # Each transformer layer uses its own adapter weights
                            # determined by retrieval similarity at that layer.
                            #
                            # Example weight distribution across two layers:
                            #   layer 0 q_proj: adapter0=0.9, adapter1=0.1
                            #   layer 1 v_proj: adapter0=0.2, adapter1=0.8
                            #
                            # This is in contrast to fusion/mixture which apply
                            # the same global weights to every layer.
                            # ---------------------------------------------------
                            hooks = apply_per_layer_hooks(
                                peft_model=peft_model,
                                per_layer_weights=per_layer_weight_map,
                                all_adapter_names=all_adapter_names,
                            )
                            try:
                                # No lora_mapping passed — hooks handle weighting
                                outputs = peft_model.generate(
                                    input_ids=inputs["input_ids"],
                                    max_new_tokens=50,
                                    do_sample=False,
                                    temperature=1.0,
                                )
                            finally:
                                # Always clean up hooks, even if generation fails
                                remove_hooks(hooks)

                        else:
                            # ---------------------------------------------------
                            # ORIGINAL GLOBAL COMPOSITION (fusion / mixture)
                            # Same weights applied uniformly across all layers.
                            # ---------------------------------------------------
                            outputs = peft_model.generate(
                                input_ids=inputs["input_ids"],
                                max_new_tokens=50,
                                do_sample=False,
                                temperature=1.0,
                                merging_type=composition_method,
                                lora_mapping=mapping_matrix_tensor,
                            )

                    except Exception as e:
                        print(f"exception ({composition_method})", e)
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
                            "composition_method": composition_method,
                            "retrieval": retrieval_details[j],
                            # NEW: also log per-layer weights for analysis
                            "per_layer_weight_map": (
                                per_layer_weight_map if composition_method == "per_layer" else None
                            ),
                            "predicted_answer": generated_answer,
                        }
                        results.append(sample)
                        print(
                            f"[{composition_method}] generated: {generated_answer}, "
                            f"expected: {expected_answer}"
                        )

                pbar.set_description("Evaluating")
                peft_model.unload()

    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import fire
    fire.Fire(eval_datasets)