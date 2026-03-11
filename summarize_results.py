import os
import json
import argparse
from collections import defaultdict
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
import pandas as pd

# Function to calculate BLEU score
def calculate_bleu(references, candidates):
    scores = [sentence_bleu([ref.split()], cand.split()) for ref, cand in zip(references, candidates)]
    return np.round(np.mean(scores) * 100, 1) if scores else 0

# Function to calculate ROUGE score
def calculate_rouge(references, candidates):
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references, avg=True)
    rouge_1 = np.round(scores['rouge-1']['f'] * 100, 1)
    rouge_2 = np.round(scores['rouge-2']['f'] * 100, 1)
    rouge_l = np.round(scores['rouge-l']['f'] * 100, 1)
    return rouge_1, rouge_2, rouge_l

# Function to calculate Exact Match score
def calculate_em(references, candidates):
    references = [ref.split("\n\n")[0] for ref in references]
    em_scores = [1 if cal_correct(ref, cand) else 0 for ref, cand in zip(references, candidates)]
    return np.round(np.mean(em_scores) * 100, 1) if em_scores else 0

def cal_correct(generated_answer, expected_answer):
    is_correct = generated_answer.strip().lower().replace(".", "") == expected_answer.strip().lower().replace(".", "")
    return is_correct

# Function to process a file
def process_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    organized_data = defaultdict(lambda: defaultdict(list))
    for entry in data:
        domain = entry['domain']
        task = entry['task']
        organized_data[domain][task].append(entry)
    
    return organized_data

# Function to process all files in a folder and aggregate scores by domain and metric
def process_seeds_folder(parent_folder):
    # Find all seed subfolders
    seed_folders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder)
                    if os.path.isdir(os.path.join(parent_folder, d)) and d.startswith("seed")]
    if not seed_folders:
        raise ValueError("No seed subfolders found in {}".format(parent_folder))

    # Get the set of result files from the first seed folder
    result_files = [f for f in os.listdir(seed_folders[0]) if f.endswith('.json')]

    # Structure: domain -> metric -> file_name -> list of scores (one per seed)
    domain_specific_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file_name in result_files:
        for seed_folder in seed_folders:
            file_path = os.path.join(seed_folder, file_name)
            if not os.path.exists(file_path):
                continue
            domains_data = process_file(file_path)

            for domain, tasks_data in domains_data.items():
                for task, entries in tasks_data.items():
                    metric = entries[0]['metric']
                    references = [entry['targets'] for entry in entries]
                    candidates = [entry['predicted_answer'] for entry in entries]

                    if metric == 'bleu':
                        score = calculate_bleu(references, candidates)
                        domain_specific_metrics[domain][metric][file_name].append(score)
                    elif metric == 'rouge':
                        rouge_1, rouge_2, rouge_l = calculate_rouge(references, candidates)
                        domain_specific_metrics[domain]['rouge-1'][file_name].append(rouge_1)
                        domain_specific_metrics[domain]['rouge-2'][file_name].append(rouge_2)
                        domain_specific_metrics[domain]['rouge-l'][file_name].append(rouge_l)
                    elif metric == 'em':
                        score = calculate_em(references, candidates)
                        domain_specific_metrics[domain][metric][file_name].append(score)
    return domain_specific_metrics, result_files


def get_latest_timestamp_results_dir(base_results_folder):
    """Return the latest timestamped results directory under base_results_folder.

    Expected timestamp format: YYYYMMDD_HHMMSS
    """
    timestamp_dirs = []
    for d in os.listdir(base_results_folder):
        full_path = os.path.join(base_results_folder, d)
        if not os.path.isdir(full_path):
            continue
        try:
            datetime.strptime(d, "%Y%m%d_%H%M%S")
            timestamp_dirs.append(d)
        except ValueError:
            continue

    if not timestamp_dirs:
        raise ValueError(
            "No timestamped result directories found in {}. "
            "Expected format: YYYYMMDD_HHMMSS".format(base_results_folder)
        )

    latest_dir = sorted(timestamp_dirs)[-1]
    return os.path.join(base_results_folder, latest_dir)

# Function to convert data to LaTeX format with domain and metric averages
def convert_to_latex_mean_std(data, result_files):
    data_list = []
    for domain, metrics in data.items():
        for metric, files in metrics.items():
            row = {'Domain-Metric': f"{domain}-{metric}"}
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
    columns_ordered = ['Domain-Metric'] + result_files
    df = df[columns_ordered]

    return df.to_latex(index=False)

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize seed results. If --seeds-folder is provided, it is used directly. "
            "Otherwise, the most recent timestamped folder under --base-results-folder is used."
        )
    )
    parser.add_argument(
        "--seeds-folder",
        type=str,
        default=None,
        help="Path to folder containing seed* subfolders. Overrides timestamp lookup.",
    )
    parser.add_argument(
        "--base-results-folder",
        type=str,
        default="results",
        help="Base folder containing timestamped result directories (default: results).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seeds_folder:
        folder_path = args.seeds_folder
    else:
        folder_path = get_latest_timestamp_results_dir(args.base_results_folder)

    processed_data, result_files = process_seeds_folder(folder_path)
    latex_table = convert_to_latex_mean_std(processed_data, result_files)
    print(latex_table)


if __name__ == "__main__":
    main()
