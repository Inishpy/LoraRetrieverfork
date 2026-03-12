
"""Approach 2: Direct Weight Matrix Surgery (Your True Vision)
Bypass PEFT entirely and directly stitch together the best LoRA matrices at each layer:
pythondef build_franken_lora(base_model, layerwise_best_adapters):
    "
    For each transformer layer, load the ΔW from whichever
    adapter scored highest for that specific layer and inject
    it directly into the base model weights.
    "
    for layer_idx, layer_name in enumerate(base_model.layers):
        # Each layer independently picks its best adapter
        best_adapter = layerwise_best_adapters[layer_name]  # e.g. AdapterC for layer 5
        delta_w = _extract_layer_deltas(best_adapter)[layer_name]
        
        # Inject directly: W_new = W_base + ΔW_best_for_this_layer
        layer.q_proj.weight.data += delta_w["q_proj"]
        layer.v_proj.weight.data += delta_w["v_proj"]
Now layer 0 literally has AdapterA's matrices baked in, layer 1 has AdapterC's, and so on — no blending overhead at inference time."""



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
global_lora_factor_cache = {}


SUPPORTED_COMPOSITION_METHODS = ["fusion", "mixture", "surgery"]


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


def _candidate_layer_mapping_keys(module_key):
    if not module_key:
        return []

    keys = []

    def add_key(key):
        if key and key not in keys:
            keys.append(key)

    add_key(module_key)

    stripped = module_key
    while stripped.startswith("base_model.model."):
        stripped = stripped[len("base_model.model.") :]
        add_key(stripped)
    while stripped.startswith("model."):
        stripped = stripped[len("model.") :]
        add_key(stripped)

    current = list(keys)
    for key in current:
        if not key.startswith("model."):
            add_key(f"model.{key}")
        if not key.startswith("base_model.model."):
            add_key(f"base_model.model.{key}")

    return keys


def _resolve_layer_pattern_value(layer_name, pattern_dict, default_value):
    if not pattern_dict:
        return default_value
    for key in _candidate_layer_mapping_keys(layer_name):
        if key in pattern_dict:
            return pattern_dict[key]
    return default_value


def _extract_layer_lora_factors(lora_path):
    """
    Return low-rank factors per layer:
      {layer_name: {"A": A_cpu_fp32, "B": B_cpu_fp32, "scaling": alpha/r}}
    """
    state_dict = _load_lora_state(lora_path)
    cfg = PeftConfig.from_pretrained(lora_path)

    rank_pattern = getattr(cfg, "rank_pattern", None) or {}
    alpha_pattern = getattr(cfg, "alpha_pattern", None) or {}
    default_r = float(cfg.r)
    default_alpha = float(cfg.lora_alpha)

    grouped = {}
    for key, val in state_dict.items():
        if "lora_A.weight" in key:
            layer = key.replace(".lora_A.weight", "").replace("base_model.model.", "")
            grouped.setdefault(layer, {})["A"] = val.float().cpu().contiguous()
        elif "lora_B.weight" in key:
            layer = key.replace(".lora_B.weight", "").replace("base_model.model.", "")
            grouped.setdefault(layer, {})["B"] = val.float().cpu().contiguous()

    factors = {}
    for layer, parts in grouped.items():
        if "A" not in parts or "B" not in parts:
            continue

        layer_r = float(_resolve_layer_pattern_value(layer, rank_pattern, default_r))
        layer_alpha = float(_resolve_layer_pattern_value(layer, alpha_pattern, default_alpha))
        if layer_r <= 0:
            continue

        factors[layer] = {
            "A": parts["A"],
            "B": parts["B"],
            "scaling": layer_alpha / layer_r,
        }

    return factors


def _get_layer_lora_factors_cached(lora_path):
    cached = global_lora_factor_cache.get(lora_path)
    if cached is None:
        cached = _extract_layer_lora_factors(lora_path)
        global_lora_factor_cache[lora_path] = cached
    return cached


def _resolve_param_weight_name(layer_name, param_lookup):
    if layer_name in param_lookup:
        return param_lookup[layer_name]
    for key in _candidate_layer_mapping_keys(layer_name):
        if key in param_lookup:
            return param_lookup[key]
    return None


def _resolve_adapter_layer_factors(adapter_factors, layer_name):
    if layer_name in adapter_factors:
        return adapter_factors[layer_name]
    for key in _candidate_layer_mapping_keys(layer_name):
        if key in adapter_factors:
            return adapter_factors[key]
    return None


def _apply_layerwise_weight_surgery(
    base_model,
    module_list,
    layerwise_mapping_matrix,
    sample_idx,
    sign=1.0,
    weight_eps=1e-8,
    layer_active_k=None,
):
    """
    In-place update of base model weights with per-layer weighted LoRA deltas.
    Use sign=+1 to apply and sign=-1 to revert.
    """
    if not module_list:
        return {"layers_touched": 0, "updates_applied": 0, "missing_weight_layers": 0}

    if sample_idx < 0:
        raise ValueError(f"sample_idx must be non-negative, got {sample_idx}")

    named_params = dict(base_model.named_parameters())
    param_lookup = {
        name[: -len(".weight")]: name
        for name in named_params.keys()
        if name.endswith(".weight")
    }

    layers_touched = 0
    updates_applied = 0
    missing_weight_layers = set()
    skipped_shape_mismatches = 0

    k = None
    if layer_active_k is not None and int(layer_active_k) > 0:
        k = int(layer_active_k)

    with torch.no_grad():
        for layer_name, layer_rows in layerwise_mapping_matrix.items():
            arr = np.asarray(layer_rows, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != len(module_list):
                raise ValueError(
                    f"Layer {layer_name} expected shape [batch, {len(module_list)}], got {arr.shape}"
                )
            if sample_idx >= arr.shape[0]:
                continue

            weights = np.clip(arr[sample_idx], 0.0, None)
            if k is not None and weights.size > k:
                top_idx = np.argsort(weights)[::-1][:k]
                sparse = np.zeros_like(weights)
                sparse[top_idx] = weights[top_idx]
                weights = sparse

            total = float(weights.sum())
            if total <= weight_eps:
                continue
            weights = weights / total

            param_name = _resolve_param_weight_name(layer_name, param_lookup)
            if param_name is None:
                missing_weight_layers.add(layer_name)
                continue

            param = named_params[param_name]
            if param.ndim != 2:
                missing_weight_layers.add(layer_name)
                continue

            layer_applied = False
            for adapter_idx, adapter_weight in enumerate(weights):
                adapter_weight = float(adapter_weight)
                if abs(adapter_weight) <= weight_eps:
                    continue

                adapter_path = module_list[adapter_idx]
                adapter_factors = _get_layer_lora_factors_cached(adapter_path)
                factors = _resolve_adapter_layer_factors(adapter_factors, layer_name)
                if factors is None:
                    continue

                A = factors["A"].to(device=param.device, dtype=param.dtype)
                B = factors["B"].to(device=param.device, dtype=param.dtype)
                if (
                    A.ndim != 2
                    or B.ndim != 2
                    or B.shape[1] != A.shape[0]
                    or B.shape[0] != param.shape[0]
                    or A.shape[1] != param.shape[1]
                ):
                    skipped_shape_mismatches += 1
                    continue
                alpha = float(sign * adapter_weight * factors["scaling"])

                # Efficient in-place low-rank update: W <- W + alpha * (B @ A)
                if param.device.type == "cpu" and param.dtype in (torch.float16, torch.bfloat16):
                    delta = torch.matmul(B.float(), A.float())
                    param.data.add_(delta.to(param.dtype), alpha=alpha)
                else:
                    param.data.addmm_(B, A, beta=1.0, alpha=alpha)
                layer_applied = True
                updates_applied += 1

            if layer_applied:
                layers_touched += 1

    return {
        "layers_touched": layers_touched,
        "updates_applied": updates_applied,
        "missing_weight_layers": len(missing_weight_layers),
        "skipped_shape_mismatches": skipped_shape_mismatches,
    }


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

    for i, lora_model in enumerate(lora_module_list):
        print(f"\nLoading adapter {i}: {lora_model}")
        if i == 0:
            peft_model = PeftModel.from_pretrained(base_model, lora_model, f"adapter{i}")
        else:
            peft_model.load_adapter(lora_model, f"adapter{i}")
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
    surgery_layer_top_k=3,
):
    """
    Evaluate the model on given datasets.

    Parameters:
    - data_path: Path to the evaluation dataset.
    - res_path: Path to save the evaluation results.
    - config_path: Path to configuration file for retrieval initialization.
    - eval_type: Composition method to run. One of: 'fusion', 'mixture', 'surgery'.
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
    - surgery_layer_top_k: Optional top-k active adapters per layer for surgery mode.
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

    eval_data = dataset["train"].map(generate_and_tokenize_prompt)

    model_path = f"meta-llama/Llama-2-{model_size}-hf"
    base_model, tokenizer = load_base_model(model_path)
    base_model = base_model.to(device)
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

                print("module_list:", module_list)
                if mapping_matrix is None:
                    raise ValueError(
                        "mapping_matrix is None. Retrieval may have failed or returned no adapters."
                    )
                print("Number of adapters loaded:", len(module_list))
                peft_methods = [m for m in methods_to_run if m in ("fusion", "mixture")]
                run_surgery = "surgery" in methods_to_run

                layerwise_mapping_tensor = None
                mapping_matrix_tensor = None
                peft_model = None

                if peft_methods:
                    mapping_matrix_tensor = torch.tensor(mapping_matrix, dtype=torch.float32).to(device)
                    print("mapping_matrix_tensor.shape:", mapping_matrix_tensor.shape)
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

                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                for composition_method in peft_methods:
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
                        global_map_shape = (
                            tuple(mapping_matrix_tensor.shape)
                            if mapping_matrix_tensor is not None
                            else None
                        )
                        print(
                            f"exception ({composition_method}) at sample_idx={i}, "
                            f"domain={eval_data['domain'][i]}, task={eval_data['task'][i]}, "
                            f"adapters={len(module_list)}, global_map_shape={global_map_shape}: {e}"
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

                if run_surgery:
                    for j, expected_answer in enumerate(eval_data["targets"][i : i + batch_size]):
                        applied_stats = None
                        surgery_applied = False
                        try:
                            applied_stats = _apply_layerwise_weight_surgery(
                                base_model=base_model,
                                module_list=module_list,
                                layerwise_mapping_matrix=layerwise_mapping_matrix,
                                sample_idx=j,
                                sign=1.0,
                                layer_active_k=surgery_layer_top_k,
                            )
                            surgery_applied = True
                            if retrieval_debug:
                                print(f"[surgery-apply] sample_idx={i + j}, stats={applied_stats}")

                            single_inputs = {k: v[j : j + 1] for k, v in inputs.items()}
                            surgery_outputs = base_model.generate(
                                input_ids=single_inputs["input_ids"],
                                attention_mask=single_inputs.get("attention_mask", None),
                                max_new_tokens=50,
                                do_sample=False,
                                temperature=1.0,
                            )
                        except Exception as e:
                            print(
                                f"exception (surgery) at sample_idx={i + j}, "
                                f"domain={eval_data['domain'][i + j]}, task={eval_data['task'][i + j]}, "
                                f"adapters={len(module_list)}: {e}"
                            )
                            continue
                        finally:
                            if surgery_applied:
                                try:
                                    revert_stats = _apply_layerwise_weight_surgery(
                                        base_model=base_model,
                                        module_list=module_list,
                                        layerwise_mapping_matrix=layerwise_mapping_matrix,
                                        sample_idx=j,
                                        sign=-1.0,
                                        layer_active_k=surgery_layer_top_k,
                                    )
                                    if retrieval_debug:
                                        print(f"[surgery-revert] sample_idx={i + j}, stats={revert_stats}")
                                except Exception as revert_error:
                                    raise RuntimeError(
                                        f"Failed to revert surgery weights for sample {i + j}: {revert_error}"
                                    ) from revert_error

                        generated_answer = tokenizer.decode(surgery_outputs[0], skip_special_tokens=True)
                        generated_answer = generated_answer.strip().split("### Response:\n")[-1]

                        sample = {
                            "inputs": eval_data["inputs"][i + j],
                            "targets": eval_data["targets"][i + j],
                            "metric": eval_data["metric"][i + j],
                            "domain": eval_data["domain"][i + j],
                            "task": eval_data["task"][i + j],
                            "composition_method": "surgery",
                            "retrieval": retrieval_details[j],
                            "predicted_answer": generated_answer,
                        }
                        results.append(sample)

                        print(
                            f"[surgery] generated_answer: {generated_answer}, expected_answer: {expected_answer}"
                        )

                pbar.update(len(input_text))
                pbar.set_description("Evaluating")
                if peft_model is not None:
                    peft_model.unload()

    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Skipped samples due to empty retrieval: {skipped_empty_retrieval}")


if __name__ == "__main__":
    import fire

    fire.Fire(eval_datasets)
