import os
import json
import argparse
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
import pandas as pd


def calculate_bleu(references, candidates):
    scores = [sentence_bleu([ref.split()], cand.split()) for ref, cand in zip(references, candidates)]
    return np.round(np.mean(scores) * 100, 1) if scores else 0


def calculate_rouge(references, candidates):
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references, avg=True)
    rouge_1 = np.round(scores["rouge-1"]["f"] * 100, 1)
    rouge_2 = np.round(scores["rouge-2"]["f"] * 100, 1)
    rouge_l = np.round(scores["rouge-l"]["f"] * 100, 1)
    return rouge_1, rouge_2, rouge_l


def cal_correct(generated_answer, expected_answer):
    return generated_answer.strip().lower().replace(".", "") == expected_answer.strip().lower().replace(".", "")


def calculate_em(references, candidates):
    references = [ref.split("\n\n")[0] for ref in references]
    em_scores = [1 if cal_correct(ref, cand) else 0 for ref, cand in zip(references, candidates)]
    return np.round(np.mean(em_scores) * 100, 1) if em_scores else 0


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def make_entry_key(entry):
    # Include domain/task to avoid collisions when the same input text appears in multiple datasets.
    return (
        entry.get("domain", ""),
        entry.get("task", ""),
        entry.get("inputs", ""),
    )


def organize_by_domain_task(entries):
    organized_data = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        domain = entry["domain"]
        task = entry["task"]
        organized_data[domain][task].append(entry)
    return organized_data


def process_seeds_folder_with_input_filter(original_folder, modified_folder):
    seed_folders = sorted(
        d for d in os.listdir(original_folder) if os.path.isdir(os.path.join(original_folder, d)) and d.startswith("seed")
    )
    if not seed_folders:
        raise ValueError("No seed subfolders found in {}".format(original_folder))

    domain_specific_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_result_files = set()
    matched_samples = 0
    total_original_samples = 0
    filter_stats = []

    for seed_name in seed_folders:
        original_seed_dir = os.path.join(original_folder, seed_name)
        modified_seed_dir = os.path.join(modified_folder, seed_name)
        if not os.path.isdir(modified_seed_dir):
            continue

        result_files = [f for f in os.listdir(original_seed_dir) if f.endswith(".json")]
        for file_name in result_files:
            original_file_path = os.path.join(original_seed_dir, file_name)
            modified_file_path = os.path.join(modified_seed_dir, file_name)
            if not os.path.exists(modified_file_path):
                continue

            all_result_files.add(file_name)
            original_data = load_json(original_file_path)
            modified_data = load_json(modified_file_path)

            modified_keys = {make_entry_key(entry) for entry in modified_data}
            filtered_original = [entry for entry in original_data if make_entry_key(entry) in modified_keys]

            total_original_samples += len(original_data)
            matched_samples += len(filtered_original)
            filter_stats.append(
                {
                    "seed": seed_name,
                    "file_name": file_name,
                    "original": len(original_data),
                    "matched": len(filtered_original),
                    "filtered_out": len(original_data) - len(filtered_original),
                }
            )

            organized_data = organize_by_domain_task(filtered_original)
            for domain, tasks_data in organized_data.items():
                for task, entries in tasks_data.items():
                    if not entries:
                        continue
                    metric = entries[0]["metric"]
                    references = [entry["targets"] for entry in entries]
                    candidates = [entry["predicted_answer"] for entry in entries]

                    if metric == "bleu":
                        score = calculate_bleu(references, candidates)
                        domain_specific_metrics[domain][metric][file_name].append(score)
                    elif metric == "rouge":
                        rouge_1, rouge_2, rouge_l = calculate_rouge(references, candidates)
                        domain_specific_metrics[domain]["rouge-1"][file_name].append(rouge_1)
                        domain_specific_metrics[domain]["rouge-2"][file_name].append(rouge_2)
                        domain_specific_metrics[domain]["rouge-l"][file_name].append(rouge_l)
                    elif metric == "em":
                        score = calculate_em(references, candidates)
                        domain_specific_metrics[domain][metric][file_name].append(score)

    if not all_result_files:
        raise ValueError(
            "No matching seed/file pairs found between original and modified folders.\n"
            "Checked original: {}\n"
            "Checked modified: {}".format(original_folder, modified_folder)
        )

    return domain_specific_metrics, sorted(all_result_files), matched_samples, total_original_samples, filter_stats


def convert_to_latex_mean_std(data, result_files):
    data_list = []
    for domain, metrics in data.items():
        for metric, files in metrics.items():
            row = {"Domain-Metric": f"{domain}-{metric}"}
            for file_name in result_files:
                numeric_scores = [score for score in files[file_name] if isinstance(score, (int, float))]
                if numeric_scores:
                    mean = np.mean(numeric_scores)
                    std = np.std(numeric_scores)
                    row[file_name] = "{:.1f} ± {:.1f}".format(mean, std)
                else:
                    row[file_name] = "-"
            data_list.append(row)

    df = pd.DataFrame(data_list)
    columns_ordered = ["Domain-Metric"] + result_files
    df = df[columns_ordered]
    return df.to_latex(index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize ORIGINAL seed results after filtering to only the inputs present in "
            "matching seed/file JSONs from MODIFIED results."
        )
    )
    parser.add_argument(
        "--original-seeds-folder",
        type=str,
        required=True,
        help="Path to original experiment folder containing seed* subfolders.",
    )
    parser.add_argument(
        "--modified-seeds-folder",
        type=str,
        required=True,
        help="Path to modified experiment folder containing seed* subfolders.",
    )
    parser.add_argument(
        "--only-file",
        type=str,
        default=None,
        help="Optional JSON filename to process (e.g., selection_ood.json). If set, other files are ignored.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    (
        processed_data,
        result_files,
        matched_samples,
        total_original_samples,
        filter_stats,
    ) = process_seeds_folder_with_input_filter(args.original_seeds_folder, args.modified_seeds_folder)

    if args.only_file:
        keep = args.only_file
        filter_stats = [row for row in filter_stats if row["file_name"] == keep]
        result_files = [f for f in result_files if f == keep]

        for domain in list(processed_data.keys()):
            for metric in list(processed_data[domain].keys()):
                files = processed_data[domain][metric]
                for f in list(files.keys()):
                    if f != keep:
                        del files[f]
                if not files:
                    del processed_data[domain][metric]
            if not processed_data[domain]:
                del processed_data[domain]

        matched_samples = sum(row["matched"] for row in filter_stats)
        total_original_samples = sum(row["original"] for row in filter_stats)

        if not filter_stats:
            raise ValueError("No matching data found for --only-file '{}'".format(keep))

    print("# Per-seed/file filtering")
    for row in sorted(filter_stats, key=lambda x: (x["seed"], x["file_name"])):
        keep_pct = (100.0 * row["matched"] / row["original"]) if row["original"] else 0.0
        print(
            "{} {}: original={} matched={} filtered_out={} keep={:.2f}%".format(
                row["seed"],
                row["file_name"],
                row["original"],
                row["matched"],
                row["filtered_out"],
                keep_pct,
            )
        )

    print(
        "# Filtered original samples: {}/{} ({:.2f}%)".format(
            matched_samples,
            total_original_samples,
            (100.0 * matched_samples / total_original_samples) if total_original_samples else 0.0,
        )
    )
    print(convert_to_latex_mean_std(processed_data, result_files))


if __name__ == "__main__":
    main()
