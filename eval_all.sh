#!/bin/bash

# Create timestamped run directories so all seed folders and logs are grouped per run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_RESULTS_DIR="results/${TIMESTAMP}"
RUN_LOGS_DIR="logs/${TIMESTAMP}"
mkdir -p "${RUN_RESULTS_DIR}"
mkdir -p "${RUN_LOGS_DIR}"

# List of seeds
for SEED in 0 1 2 3
do
    # Create results subfolder for this seed
    mkdir -p "${RUN_RESULTS_DIR}/seed${SEED}"

    # Run all experiments for this seed
    CUDA_VISIBLE_DEVICES=1  python mainlayerwise4.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/mixture_ood.json" --eval_type surgery  --lora_num 3 --batch_size 8 --ood=True --seed ${SEED} > "${RUN_LOGS_DIR}/mixture_ood_seed${SEED}.log" 2>&1 
    CUDA_VISIBLE_DEVICES=1  python mainlayerwise4.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/mixture.json" --eval_type surgery  --lora_num 3 --batch_size 8 --seed ${SEED} > "${RUN_LOGS_DIR}/mixture_seed${SEED}.log" 2>&1 
    
    CUDA_VISIBLE_DEVICES=1 python mainlayerwise4.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/fusion.json" --eval_type surgery --lora_num 3 --batch_size 8 --seed ${SEED} > "${RUN_LOGS_DIR}/fusion_seed${SEED}.log" 2>&1 
    CUDA_VISIBLE_DEVICES=1 python mainlayerwise4.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/fusion_ood.json" --eval_type surgery --lora_num 3 --batch_size 8 --ood=True --seed ${SEED} > "${RUN_LOGS_DIR}/fusion_ood_seed${SEED}.log" 2>&1 
    CUDA_VISIBLE_DEVICES=1 python mainlayerwise4.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/selection.json" --eval_type surgery --lora_num 1 --batch_size 8 --seed ${SEED} > "${RUN_LOGS_DIR}/selection_seed${SEED}.log" 2>&1 
    CUDA_VISIBLE_DEVICES=1 python mainlayerwise4.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/selection_ood.json" --eval_type surgery --lora_num 1 --batch_size 8 --ood=True --seed ${SEED} > "${RUN_LOGS_DIR}/selection_ood_seed${SEED}.log" 2>&1 
    CUDA_VISIBLE_DEVICES=1 python mainlayerwise4.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/best_selection.json" --eval_type surgery  --lora_num 1 --batch_size 8 --best_selection=True --seed ${SEED} > "${RUN_LOGS_DIR}/best_selection_seed${SEED}.log" 2>&1 
    
done
