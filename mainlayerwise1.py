## Why You Need Both
"""
They answer two different questions:

| Parameter | Question it answers |
|---|---|
| `layer_top_k` | How **wide** is the search per layer? (exploration) |
| `lora_num` | How many adapters do we actually **load**? (budget) |

Consider this scenario with `layer_top_k=3` and `lora_num=2`:
```
Layer 0  → [AdapterA, AdapterB, AdapterC]
Layer 1  → [AdapterA, AdapterD, AdapterB]
Layer 2  → [AdapterE, AdapterA, AdapterB]

After score aggregation:
  AdapterA = 8.7  (won in all 3 layers)
  AdapterB = 5.2  (appeared in all 3 layers)
  AdapterC = 1.1  (only layer 0)
  AdapterD = 0.9  (only layer 1)
  AdapterE = 0.8  (only layer 2)

lora_num=2 → final selection: [AdapterA, AdapterB]

"""


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


# Prompter is a utility class to create a prompt for a given input
prompter = Prompter("alpaca")


INSTRUCTION_PREFIX = "Represent the sentence for similar task retrieval: "
global_embed_model = None
global_layerwise_index = None


SUPPORTED_COMPOSITION_METHODS = ["fusion", "mixture"]


def load_base_model(model_name_or_path="meta-llama/Llama-2-7b-hf"):
    """ 
    Load the base model and tokenizer from a given model path.
    """
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


def _extract_layer_deltas(lora_path):
    """Return {layer_name: deltaW} where deltaW = (alpha/r) * (B @ A)."""
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
    """Convert variable-size delta matrix into fixed-size vector via adaptive pooling."""
    flat = delta_w.reshape(-1)
    if flat.numel() == 0:
        return np.zeros(emb_dim, dtype=np.float32)

    v = F.adaptive_avg_pool1d(flat.abs().unsqueeze(0).unsqueeze(0), emb_dim).squeeze()
    v = v / (v.norm(p=2) + 1e-12)
    return v.numpy().astype(np.float32)


def build_layerwise_lora_index(lora_index, emb_dim=768, blend=0.35):
    """
    Build layer-wise index from per-task embeddings + per-layer LoRA signatures.
    """
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


def _retrieve_topk_per_layer(
    query_text,
    layerwise_index,
    layer_k=1,
    exclude_item=None,
    query_idx=None,
    debug=False,
):
    q = _encode_with_instruction([query_text])[0].astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)

    hits = {}
    for layer_name, bucket in layerwise_index.items():
        total_candidates = len(bucket["lora_paths"])
        valid_idx = list(range(total_candidates))
        if exclude_item is not None:
            valid_idx = [
                idx for idx, path in enumerate(bucket["lora_paths"]) if path != exclude_item
            ]

        if len(valid_idx) == 0:
            hits[layer_name] = []
            if debug:
                print(
                    "[retrieve-debug-layer] "
                    f"query_idx={query_idx}, layer={layer_name}, "
                    f"total_candidates={total_candidates}, post_exclude_candidates=0, "
                    f"exclude_item={exclude_item}"
                )
            continue

        sims = cosine_similarity(q[None, :], bucket["matrix"][valid_idx])[0]
        top_local_idx = np.argsort(sims)[::-1][:layer_k]
        top_idx = [valid_idx[int(idx)] for idx in top_local_idx]
        hits[layer_name] = [
            {
                "lora_path": bucket["lora_paths"][int(idx)],
                "score": float(sims[int(local_idx)]),
            }
            for local_idx, idx in zip(top_local_idx, top_idx)
        ]
        if debug:
            print(
                "[retrieve-debug-layer] "
                f"query_idx={query_idx}, layer={layer_name}, "
                f"total_candidates={total_candidates}, post_exclude_candidates={len(valid_idx)}, "
                f"requested_top_k={layer_k}, returned_hits={len(hits[layer_name])}, "
                f"exclude_item={exclude_item}"
            )
    return hits


def _scores_to_normalized_weights(score_by_adapter, chosen_adapters, fallback_weights=None):
    """Convert adapter scores to non-negative normalized weights aligned with chosen_adapters."""
    if not chosen_adapters:
        return []

    scores = np.asarray(
        [max(float(score_by_adapter.get(adapter, 0.0)), 0.0) for adapter in chosen_adapters],
        dtype=np.float32,
    )
    total = float(scores.sum())

    if total <= 1e-12:
        if fallback_weights is not None and len(fallback_weights) == len(chosen_adapters):
            fallback = np.asarray(fallback_weights, dtype=np.float32)
            fallback = np.clip(fallback, 0.0, None)
            fb_total = float(fallback.sum())
            if fb_total > 1e-12:
                return (fallback / fb_total).astype(np.float32).tolist()
        # Fallback to uniform to keep mapping well-defined.
        return [1.0 / len(chosen_adapters)] * len(chosen_adapters)

    return (scores / total).astype(np.float32).tolist()


def perform_search(
    query_list,
    k=20,
    exclude_list=None,
    layer_top_k=3,
    return_details=False,
    return_layerwise_mapping=False,
    debug=False,
):
    """
    Layer-wise retrieval entrypoint compatible with original API.

    Returns:
      - all_results_list: unique selected adapters across query_list
      - mapping_matrix: global soft weights [batch, len(all_results_list)]
      - layerwise_mapping_matrix (optional): per-layer soft weights with same row/col layout
    """
    global global_layerwise_index

    all_results_set = set()
    selected_per_query = []
    weights_per_query = []
    layerwise_weights_per_query = []
    retrieval_details = []
    total_layers = len(global_layerwise_index) if global_layerwise_index else 0
    if global_layerwise_index:
        total_entries = sum(len(v["lora_paths"]) for v in global_layerwise_index.values())
        unique_index_adapters = len(
            {path for v in global_layerwise_index.values() for path in v["lora_paths"]}
        )
    else:
        total_entries = 0
        unique_index_adapters = 0

    for j, query in enumerate(query_list):
        exclude_item = exclude_list[j] if exclude_list else None
        if debug:
            print(
                "[retrieve-debug-query-start] "
                f"query_idx={j}, exclude_item={exclude_item}, layer_top_k={layer_top_k}, k={k}, "
                f"index_layers={total_layers}, index_entries={total_entries}, "
                f"index_unique_adapters={unique_index_adapters}"
            )
        layer_hits = _retrieve_topk_per_layer(
            query,
            global_layerwise_index,
            layer_k=layer_top_k,
            exclude_item=exclude_item,
            query_idx=j,
            debug=debug,
        )

        score_by_adapter = defaultdict(float)
        score_by_adapter_per_layer = defaultdict(lambda: defaultdict(float))
        for layer_name, per_layer_hits in layer_hits.items():
            for hit in per_layer_hits:
                adapter = hit["lora_path"]
                score_by_adapter[adapter] += hit["score"]
                score_by_adapter_per_layer[layer_name][adapter] += hit["score"]

        ranked = sorted(score_by_adapter.items(), key=lambda x: x[1], reverse=True)
        if debug:
            total_hits = sum(len(v) for v in layer_hits.values())
            unique_hit_adapters = len(
                {hit["lora_path"] for per_layer_hits in layer_hits.values() for hit in per_layer_hits}
            )
            print(
                "[retrieve-debug-query-after-layer-topk] "
                f"query_idx={j}, total_hits={total_hits}, unique_hit_adapters={unique_hit_adapters}, "
                f"ranked_adapter_count={len(ranked)}"
            )
            print(
                "[retrieve-debug-query-top-ranked] "
                f"query_idx={j}, top10={[name for name, _ in ranked[:10]]}"
            )
        ranked_per_layer = {
            layer_name: sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
            for layer_name, layer_scores in score_by_adapter_per_layer.items()
        }
        layerwise_selected_adapters = {
            layer_name: [name for name, _ in layer_rank[:k]]
            for layer_name, layer_rank in ranked_per_layer.items()
        }
        chosen_set = set()
        for layer_selected in layerwise_selected_adapters.values():
            chosen_set.update(layer_selected)
        if not chosen_set and ranked:
            chosen_set.update([name for name, _ in ranked[:k]])
        chosen = sorted(chosen_set, key=lambda name: float(score_by_adapter.get(name, 0.0)), reverse=True)
        chosen_weights = _scores_to_normalized_weights(score_by_adapter, chosen)
        if debug:
            print(
                "[retrieve-debug-query-selected] "
                f"query_idx={j}, selected_count={len(chosen)}, selected_top10={chosen[:10]}"
            )
            if not chosen:
                print(
                    "[retrieve-debug-query-empty] "
                    f"query_idx={j}, exclude_item={exclude_item}, "
                    "No adapters remained after pre-filter + per-layer top-k."
                )

        layerwise_weight_map = {}
        for layer_name in layer_hits.keys():
            layer_weights = _scores_to_normalized_weights(
                score_by_adapter_per_layer[layer_name],
                chosen,
                fallback_weights=chosen_weights,
            )
            layerwise_weight_map[layer_name] = {
                adapter: weight for adapter, weight in zip(chosen, layer_weights)
            }

        all_results_set.update(chosen)
        selected_per_query.append(chosen)
        weights_per_query.append(chosen_weights)
        layerwise_weights_per_query.append(layerwise_weight_map)

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
                    "layerwise_selected_adapters": layerwise_selected_adapters,
                    "layerwise_selected_weights": {
                        layer_name: [
                            {"lora_path": adapter, "weight": float(weight_map[adapter])}
                            for adapter in chosen
                            if weight_map[adapter] > 0
                        ]
                        for layer_name, weight_map in layerwise_weight_map.items()
                    },
                }
            )

    all_results_list = sorted(list(all_results_set))
    mapping_matrix = []
    for chosen, chosen_weights in zip(selected_per_query, weights_per_query):
        weight_map = {adapter: weight for adapter, weight in zip(chosen, chosen_weights)}
        mapping_vector = [float(weight_map.get(result, 0.0)) for result in all_results_list]
        mapping_matrix.append(mapping_vector)

    layerwise_mapping_matrix = {}
    if return_layerwise_mapping:
        layer_names = sorted(global_layerwise_index.keys()) if global_layerwise_index else []
        for layer_name in layer_names:
            layer_rows = []
            for chosen, chosen_weights, layer_weight_map in zip(
                selected_per_query, weights_per_query, layerwise_weights_per_query
            ):
                fallback_map = {adapter: weight for adapter, weight in zip(chosen, chosen_weights)}
                weight_map = layer_weight_map.get(layer_name, fallback_map)
                layer_rows.append(
                    [float(weight_map.get(result, 0.0)) for result in all_results_list]
                )
            layerwise_mapping_matrix[layer_name] = layer_rows

    if return_details and return_layerwise_mapping:
        return all_results_list, mapping_matrix, retrieval_details, layerwise_mapping_matrix

    if return_details:
        return all_results_list, mapping_matrix, retrieval_details

    if return_layerwise_mapping:
        return all_results_list, mapping_matrix, layerwise_mapping_matrix

    return all_results_list, mapping_matrix


def _resolve_eval_types(eval_type="fusion", eval_types=None):
    """Resolve composition methods to evaluate."""
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

    # Keep stable order and remove duplicates.
    seen = set()
    ordered = []
    for m in valid:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _normalize_rows_with_fallback(matrix, fallback_matrix=None, eps=1e-12):
    """Row-normalize non-negative matrix; rows with zero sum fall back to provided rows."""
    mat = np.asarray(matrix, dtype=np.float32).copy()
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {mat.shape}")
    mat = np.clip(mat, 0.0, None)

    row_sums = mat.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= eps
    if np.any(zero_rows):
        if fallback_matrix is not None:
            fb = np.asarray(fallback_matrix, dtype=np.float32)
            if fb.shape != mat.shape:
                raise ValueError(
                    f"Fallback matrix shape {fb.shape} must match matrix shape {mat.shape}."
                )
            mat[zero_rows] = np.clip(fb[zero_rows], 0.0, None)
        else:
            if mat.shape[1] <= 0:
                raise ValueError("Cannot normalize matrix with zero columns.")
            mat[zero_rows] = 1.0 / mat.shape[1]
        row_sums = mat.sum(axis=1, keepdims=True)

    return mat / np.clip(row_sums, eps, None)


def _prune_adapter_mappings(
    module_list,
    mapping_matrix,
    layerwise_mapping_matrix,
    weight_prune_eps=0.0,
    mass_prune_eps=1e-8,
    max_adapters_to_load=None,
):
    """
    Prune low-impact adapters before loading to reduce memory/latency.

    - weight_prune_eps: zero-out per-entry weights below this threshold before renormalization.
    - mass_prune_eps: drop adapters whose total mass across (global + all layers + all rows) is below this.
    - max_adapters_to_load: optional hard cap on loaded adapters based on aggregate mass.
    """
    if not module_list:
        raise ValueError("module_list is empty; retrieval returned no adapters.")

    global_map = np.asarray(mapping_matrix, dtype=np.float32)
    if global_map.ndim != 2 or global_map.shape[1] != len(module_list):
        raise ValueError(
            f"Invalid mapping_matrix shape {global_map.shape} for {len(module_list)} adapters."
        )
    if weight_prune_eps > 0:
        global_map[np.abs(global_map) < weight_prune_eps] = 0.0
    global_map = _normalize_rows_with_fallback(global_map)

    normalized_layers = {}
    for layer_name, layer_matrix in layerwise_mapping_matrix.items():
        arr = np.asarray(layer_matrix, dtype=np.float32)
        if arr.shape != global_map.shape:
            raise ValueError(
                f"Layer {layer_name} shape {arr.shape} does not match global mapping shape {global_map.shape}."
            )
        if weight_prune_eps > 0:
            arr[np.abs(arr) < weight_prune_eps] = 0.0
        normalized_layers[layer_name] = _normalize_rows_with_fallback(arr, fallback_matrix=global_map)

    adapter_mass = global_map.sum(axis=0)
    for arr in normalized_layers.values():
        adapter_mass += arr.sum(axis=0)

    keep_idx = np.where(adapter_mass > float(mass_prune_eps))[0]
    if keep_idx.size == 0:
        keep_idx = np.asarray([int(np.argmax(adapter_mass))], dtype=np.int64)

    if (
        max_adapters_to_load is not None
        and int(max_adapters_to_load) > 0
        and keep_idx.size > int(max_adapters_to_load)
    ):
        keep_count = int(max_adapters_to_load)
        keep_idx = keep_idx[np.argsort(adapter_mass[keep_idx])[::-1][:keep_count]]

    keep_idx = keep_idx.astype(np.int64)
    pruned_module_list = [module_list[int(i)] for i in keep_idx]

    pruned_global_map = _normalize_rows_with_fallback(global_map[:, keep_idx])
    pruned_layer_maps = {
        layer_name: _normalize_rows_with_fallback(arr[:, keep_idx], fallback_matrix=pruned_global_map)
        for layer_name, arr in normalized_layers.items()
    }

    kept = len(pruned_module_list)
    removed = len(module_list) - kept
    prune_stats = {"original": len(module_list), "kept": kept, "removed": removed}
    return (
        pruned_module_list,
        pruned_global_map.astype(np.float32),
        {k: v.astype(np.float32) for k, v in pruned_layer_maps.items()},
        prune_stats,
    )


def _has_local_adapter_files(lora_path):
    """Return True if adapter config + weights are available locally (no network)."""
    if os.path.isdir(lora_path):
        has_config = os.path.exists(os.path.join(lora_path, "adapter_config.json"))
        has_weights = os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")) or os.path.exists(
            os.path.join(lora_path, "adapter_model.bin")
        )
        return has_config and has_weights

    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(repo_id=lora_path, filename="adapter_config.json", local_files_only=True)
    except Exception:
        return False

    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        try:
            hf_hub_download(repo_id=lora_path, filename=filename, local_files_only=True)
            return True
        except Exception:
            continue
    return False


def _filter_unavailable_adapters(module_list, mapping_matrix, layerwise_mapping_matrix):
    """
    Filter adapters that are not locally available and shrink mapping matrices accordingly.
    """
    keep_idx = []
    dropped = []
    for idx, adapter in enumerate(module_list):
        if _has_local_adapter_files(adapter):
            keep_idx.append(idx)
        else:
            dropped.append(adapter)

    if len(keep_idx) == len(module_list):
        return module_list, mapping_matrix, layerwise_mapping_matrix, dropped

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    filtered_modules = [module_list[int(i)] for i in keep_idx]

    global_map = np.asarray(mapping_matrix, dtype=np.float32)
    filtered_global_map = global_map[:, keep_idx] if keep_idx.size > 0 else np.zeros((global_map.shape[0], 0), dtype=np.float32)

    filtered_layer_maps = {}
    for layer_name, layer_matrix in layerwise_mapping_matrix.items():
        layer_arr = np.asarray(layer_matrix, dtype=np.float32)
        filtered_layer_maps[layer_name] = (
            layer_arr[:, keep_idx] if keep_idx.size > 0 else np.zeros((layer_arr.shape[0], 0), dtype=np.float32)
        )

    return filtered_modules, filtered_global_map, filtered_layer_maps, dropped


def init_vector_db(config_path="./config/config2.json"):
    """
    Initialize the vector database with configurations from the specified JSON file.
    """
    with open(config_path, "r") as file:
        lora_configs = json.load(file)

    initialize_index(lora_configs)


def load_peft_model(lora_module_list, base_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_lists = []
    if isinstance(base_model, PeftModel):
        # Avoid adapter namespace leakage if caller accidentally passes a wrapped model.
        base_model = base_model.unload()

    for i, lora_model in enumerate(lora_module_list):
        print(f"\nLoading adapter {i}: {lora_model}")
        if i == 0:
            peft_model = PeftModel.from_pretrained(
                base_model,
                lora_model,
                f"adapter{i}",
                local_files_only=True,
            )
        else:
            peft_model.load_adapter(
                lora_model,
                f"adapter{i}",
                local_files_only=True,
            )
        lora_lists.append(f"adapter{i}")

        print(f"Adapter config: {peft_model.peft_config[f'adapter{i}']}")

    print(f"\nSetting adapters: {lora_lists}")
    # Important: multi-adapter activation must go through base_model on PeftModel.
    # Using peft_model.set_adapter(list) can leave only one adapter active in layers.
    peft_model.base_model.set_adapter(lora_lists)

    print(f"Active adapters after set_adapter: {peft_model.active_adapters}")
    if len(peft_model.active_adapters) != len(lora_lists):
        raise RuntimeError(
            f"Adapter activation mismatch: requested {len(lora_lists)} adapters "
            f"but model reports {len(peft_model.active_adapters)} active. "
            "This will break lora_mapping width checks."
        )

    for name, module in peft_model.named_modules():
        if "q_proj" in name and hasattr(module, "lora_A"):
            print(f"\n{name}:")
            print(f"  lora_A type: {type(module.lora_A)}")
            if hasattr(module.lora_A, "keys"):
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


def _normalize_mapping_columns(mapping_tensor):
    row_sums = mapping_tensor.sum(dim=1, keepdim=True)
    zero_mask = row_sums.squeeze(-1) <= 0
    if zero_mask.any():
        # If alignment removed all mass for a row, fall back to a deterministic one-hot.
        # Prefer the strongest absolute column in that row when possible, otherwise col 0.
        repaired = mapping_tensor[zero_mask]
        if repaired.shape[1] == 0:
            raise ValueError("lora_mapping has zero adapter width after alignment.")
        fallback_idx = repaired.abs().argmax(dim=1)
        repaired = torch.zeros_like(repaired)
        repaired.scatter_(1, fallback_idx.unsqueeze(1), 1.0)
        mapping_tensor = mapping_tensor.clone()
        mapping_tensor[zero_mask] = repaired
        row_sums = mapping_tensor.sum(dim=1, keepdim=True)
    mapping_tensor = mapping_tensor / row_sums
    if not torch.isfinite(mapping_tensor).all():
        raise ValueError("lora_mapping contains NaN/inf after adapter alignment.")
    if (mapping_tensor < 0).any():
        raise ValueError("lora_mapping contains negatives after adapter alignment.")
    return mapping_tensor


def _align_mapping_to_consistent_adapters(peft_model, lora_mapping_tensor, layerwise_mapping_tensor):
    """
    Some adapters can be missing on a subset of layers. Keep only adapters present on all LoRA layers
    so every layer sees a consistent mapping width.
    """
    expected = [f"adapter{i}" for i in range(lora_mapping_tensor.shape[1])]
    expected_set = set(expected)

    common_keys = None
    for _, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module.lora_A, "keys"):
            continue
        keys = set(module.lora_A.keys()) & expected_set
        if common_keys is None:
            common_keys = set(keys)
        else:
            common_keys &= keys

    if common_keys is None:
        return peft_model, lora_mapping_tensor, layerwise_mapping_tensor

    aligned_adapters = [name for name in expected if name in common_keys]
    if len(aligned_adapters) == 0:
        raise RuntimeError("No common adapters found across LoRA layers after loading.")

    if len(aligned_adapters) == len(expected):
        return peft_model, lora_mapping_tensor, layerwise_mapping_tensor

    keep_idx = torch.tensor([expected.index(name) for name in aligned_adapters], device=lora_mapping_tensor.device)
    print(
        "[adapter-alignment] "
        f"requested={len(expected)}, consistent_across_layers={len(aligned_adapters)}, "
        f"kept={aligned_adapters}"
    )

    lora_mapping_tensor = lora_mapping_tensor.index_select(dim=1, index=keep_idx)
    lora_mapping_tensor = _normalize_mapping_columns(lora_mapping_tensor)
    lora_mapping_tensor = lora_mapping_tensor.to(torch.bfloat16)

    aligned_layerwise = {}
    for layer_name, layer_tensor in layerwise_mapping_tensor.items():
        aligned_tensor = layer_tensor.index_select(dim=1, index=keep_idx)
        aligned_tensor = _normalize_mapping_columns(aligned_tensor)
        aligned_layerwise[layer_name] = aligned_tensor.to(torch.bfloat16)

    peft_model.base_model.set_adapter(aligned_adapters)
    return peft_model, lora_mapping_tensor, aligned_layerwise


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
    layer_top_k=3,
    eval_types=None,
    weight_prune_eps=1e-6,
    mass_prune_eps=1e-6,
    max_adapters_to_load=None,
    retrieval_debug=True,
):
    """
    Evaluate the model on given datasets.

    Parameters:
    - data_path: Path to the evaluation dataset.
    - res_path: Path to save the evaluation results.
    - config_path: Path to configuration file for retrieval initialization.
    - eval_type: The merging type for LoRA adapters (e.g., 'fusion').
    - eval_types: Optional comma-separated/list composition methods. Use 'all' for all supported.
    - lora_num: Per-layer top-k adapters before forming the global union to load.
    - batch_size: Batch size for evaluation.
    - ood: Flag indicating if out-of-domain exclusion should be applied.
    - best_selection: If True, use the most appropriate LoRA for each input.
    - model_size: Model size of Llama-2.
    - weight_prune_eps: Per-entry threshold; smaller weights are zeroed before renormalization.
    - mass_prune_eps: Drop adapters with very low total mass across global + layerwise mappings.
    - max_adapters_to_load: Optional cap on number of adapters loaded per batch.
    - retrieval_debug: Print detailed retrieval diagnostics per query/layer.
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

    # dataset["train"] = dataset["train"].filter(
    #     lambda data_point: data_point.get("metric") != "rouge"
    # )
    print(dataset["train"][:10])
    eval_data = dataset["train"].map(generate_and_tokenize_prompt)

    model_path = f"meta-llama/Llama-2-{model_size}-hf"
    base_model, tokenizer = load_base_model(model_path)
    base_model.eval()

    skipped_empty_retrieval = 0
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

                module_list, mapping_matrix, retrieval_details, layerwise_mapping_matrix = perform_search(
                    input_text,
                    k=lora_num,
                    exclude_list=exclude_list,
                    layer_top_k=layer_top_k,
                    return_details=True,
                    return_layerwise_mapping=True,
                    debug=retrieval_debug,
                )
                if retrieval_debug:
                    print(
                        "[retrieve-debug-batch-summary] "
                        f"batch_start={i}, batch_size={len(input_text)}, "
                        f"exclude_list={exclude_list}, module_count={len(module_list)}"
                    )
                if len(module_list) == 0:
                    skipped_empty_retrieval += len(input_text)
                    print(
                        f"[skip-empty-retrieval] sample_idx={i}, "
                        f"domain={eval_data['domain'][i]}, task={eval_data['task'][i]}, "
                        f"ood={ood}, exclude={exclude_list}"
                    )
                    pbar.update(len(input_text))
                    continue
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
                    layer_names = sorted(global_layerwise_index.keys()) if global_layerwise_index else []
                    layerwise_mapping_matrix = {
                        layer_name: mapping_matrix.astype(np.float32).tolist()
                        for layer_name in layer_names
                    }

                (
                    module_list,
                    mapping_matrix,
                    layerwise_mapping_matrix,
                    prune_stats,
                ) = _prune_adapter_mappings(
                    module_list=module_list,
                    mapping_matrix=mapping_matrix,
                    layerwise_mapping_matrix=layerwise_mapping_matrix,
                    weight_prune_eps=weight_prune_eps,
                    mass_prune_eps=mass_prune_eps,
                    max_adapters_to_load=max_adapters_to_load,
                )
                print(
                    "Adapter prune stats:",
                    prune_stats,
                    f"(weight_prune_eps={weight_prune_eps}, mass_prune_eps={mass_prune_eps}, "
                    f"max_adapters_to_load={max_adapters_to_load})",
                )
                (
                    module_list,
                    mapping_matrix,
                    layerwise_mapping_matrix,
                    unavailable_adapters,
                ) = _filter_unavailable_adapters(
                    module_list=module_list,
                    mapping_matrix=mapping_matrix,
                    layerwise_mapping_matrix=layerwise_mapping_matrix,
                )
                if unavailable_adapters:
                    print(
                        "[adapter-local-cache-filter] "
                        f"removed={len(unavailable_adapters)} adapters not found locally: "
                        f"{unavailable_adapters}"
                    )
                if len(module_list) == 0:
                    skipped_empty_retrieval += len(input_text)
                    print(
                        f"[skip-no-local-adapters] sample_idx={i}, "
                        f"domain={eval_data['domain'][i]}, task={eval_data['task'][i]}"
                    )
                    pbar.update(len(input_text))
                    continue

                print("module_list:", module_list)
                if mapping_matrix is None:
                    raise ValueError(
                        "mapping_matrix is None. Retrieval may have failed or returned no adapters."
                    )
                mapping_matrix_tensor = torch.tensor(mapping_matrix, dtype=torch.float32).to(device)
                print("mapping_matrix_tensor.shape:", mapping_matrix_tensor.shape)
                print("Number of adapters loaded:", len(module_list))
                if mapping_matrix_tensor.shape[1] != len(module_list):
                    raise ValueError(
                        f"Shape mismatch: mapping_matrix_tensor.shape[1] ({mapping_matrix_tensor.shape[1]}) "
                        f"!= number of adapters ({len(module_list)}). Please check retrieval logic."
                    )
                row_sums = mapping_matrix_tensor.sum(dim=1, keepdim=True)
                if (row_sums <= 0).any():
                    raise ValueError(
                        "lora_mapping has zero-sum rows; check layerwise retrieval and exclusions."
                    )
                mapping_matrix_tensor = mapping_matrix_tensor / row_sums
                mapping_matrix_tensor = mapping_matrix_tensor.to(torch.bfloat16)
                if not torch.isfinite(mapping_matrix_tensor).all():
                    raise ValueError(
                        "lora_mapping contains NaN/inf values; check retrieval/mapping normalization."
                    )
                if (mapping_matrix_tensor < 0).any():
                    raise ValueError(
                        "lora_mapping contains negative values; check retrieval/mapping normalization."
                    )
                layerwise_mapping_tensor = {"__default__": mapping_matrix_tensor}
                for layer_name, layer_matrix in layerwise_mapping_matrix.items():
                    layer_tensor = torch.tensor(layer_matrix, dtype=torch.float32).to(device)
                    if layer_tensor.shape[1] != len(module_list):
                        raise ValueError(
                            f"Layer {layer_name} shape mismatch: mapping width ({layer_tensor.shape[1]}) "
                            f"!= number of adapters ({len(module_list)})."
                        )
                    layer_row_sums = layer_tensor.sum(dim=1, keepdim=True)
                    if (layer_row_sums <= 0).any():
                        raise ValueError(
                            f"Layer {layer_name} has zero-sum rows in lora_mapping."
                        )
                    layer_tensor = layer_tensor / layer_row_sums
                    layer_tensor = layer_tensor.to(torch.bfloat16)
                    if not torch.isfinite(layer_tensor).all():
                        raise ValueError(f"Layer {layer_name} lora_mapping contains NaN/inf.")
                    if (layer_tensor < 0).any():
                        raise ValueError(f"Layer {layer_name} lora_mapping contains negatives.")
                    layerwise_mapping_tensor[layer_name] = layer_tensor
                print("Per-layer lora mapping entries:", len(layerwise_mapping_tensor) - 1)

                _ = check_adapter_compatibility(module_list)
                peft_model = load_peft_model(module_list, base_model)
                peft_model, mapping_matrix_tensor, layerwise_mapping_tensor = _align_mapping_to_consistent_adapters(
                    peft_model=peft_model,
                    lora_mapping_tensor=mapping_matrix_tensor,
                    layerwise_mapping_tensor=layerwise_mapping_tensor,
                )

                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                for composition_method in methods_to_run:
                    try:
                        outputs = peft_model.generate(
                            input_ids=inputs["input_ids"],
                            max_new_tokens=50,
                            do_sample=False,
                            temperature=1.0,
                            merging_type=composition_method,
                            lora_mapping=layerwise_mapping_tensor,
                        )
                    except Exception as e:
                        print(
                            f"exception ({composition_method}) at sample_idx={i}, "
                            f"domain={eval_data['domain'][i]}, task={eval_data['task'][i]}, "
                            f"adapters={len(module_list)}, global_map_shape={tuple(mapping_matrix_tensor.shape)}: {e}"
                        )
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
                            "predicted_answer": generated_answer,
                        }
                        results.append(sample)

                        print(
                            f"[{composition_method}] generated_answer: {generated_answer}, expected_answer: {expected_answer}"
                        )

                pbar.update(len(input_text))
                pbar.set_description("Evaluating")
                # Keep base_model clean between batches; unload() returns the reset base model.
                base_model = peft_model.unload()
                base_model.eval()

    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Skipped samples due to empty retrieval: {skipped_empty_retrieval}")


if __name__ == "__main__":
    import fire

    fire.Fire(eval_datasets)
