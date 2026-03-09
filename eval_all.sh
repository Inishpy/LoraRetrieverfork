#!/bin/bash

mkdir -p logs

# Create timestamped run directory so all seed folders are grouped per run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "${RUN_RESULTS_DIR}"

# List of seeds
for SEED in 0 
do
    # Create results subfolder for this seed
    mkdir -p "${RUN_RESULTS_DIR}/seed${SEED}"

    # Run all experiments for this seed
    python main.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/mixture.json" --eval_type mixture --lora_num 3 --batch_size 8 --seed ${SEED} > logs/mixture_seed${SEED}.log 2>&1 &
    python main.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/mixture_ood.json" --eval_type mixture --lora_num 3 --batch_size 8 --ood=True --seed ${SEED} > logs/mixture_ood_seed${SEED}.log 2>&1 
    python main.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/fusion.json" --eval_type fusion --lora_num 3 --batch_size 8 --seed ${SEED} > logs/fusion_seed${SEED}.log 2>&1 &
    python main.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/fusion_ood.json" --eval_type fusion --lora_num 3 --batch_size 8 --ood=True --seed ${SEED} > logs/fusion_ood_seed${SEED}.log 2>&1 
    python main.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/selection.json" --eval_type fusion --lora_num 1 --batch_size 8 --seed ${SEED} > logs/selection_seed${SEED}.log 2>&1 &
    python main.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/selection_ood.json" --eval_type fusion --lora_num 1 --batch_size 8 --ood=True --seed ${SEED} > logs/selection_ood_seed${SEED}.log 2>&1 
    python main.py --data_path dataset/combined_test.json --res_path "${RUN_RESULTS_DIR}/seed${SEED}/best_selection.json" --eval_type mixture --lora_num 1 --batch_size 8 --best_selection=True --seed ${SEED} > logs/best_selection_seed${SEED}.log 2>&1 
done
