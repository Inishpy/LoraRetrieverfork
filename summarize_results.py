import os
import json
from collections import defaultdict
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
def process_folder(folder_path):
    domain_specific_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
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
    
    return domain_specific_metrics

# Function to convert data to LaTeX format with domain and metric averages
def convert_to_latex_modified(data, folder_path):
    data_list = []
    for domain, metrics in data.items():
        for metric, files in metrics.items():
            row = {'Domain-Metric': f"{domain}-{metric}"}
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    numeric_scores = [score for score in files[file_name] if isinstance(score, (int, float))]
                    average_score = np.mean(numeric_scores) if numeric_scores else 0
                    row[file_name] = "{:.1f}".format(average_score)  # Format to one decimal place
            data_list.append(row)

    df = pd.DataFrame(data_list)
    columns_ordered = ['Domain-Metric'] + [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]
    df = df[columns_ordered]

    return df.to_latex(index=False)

# Example usage
folder_path = 'results'  # Replace with your actual folder path
processed_data = process_folder(folder_path)
latex_table = convert_to_latex_modified(processed_data, folder_path)
print(latex_table)
