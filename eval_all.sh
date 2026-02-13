#!/bin/bash

mkdir -p logs

# List of seeds
for SEED in 0 1 2 3 4
do
    # Create results subfolder for this seed
    mkdir -p results/seed${SEED}

    # Run all experiments for this seed
    python main.py --data_path dataset/combined_test.json --res_path results/seed${SEED}/mixture.json --eval_type mixture --lora_num 3 --batch_size 8 --seed ${SEED} > logs/mixture_seed${SEED}.log 2>&1 &
    python main.py --data_path dataset/combined_test.json --res_path results/seed${SEED}/mixture_ood.json --eval_type mixture --lora_num 3 --batch_size 8 --ood=True --seed ${SEED} > logs/mixture_ood_seed${SEED}.log 2>&1 
    python main.py --data_path dataset/combined_test.json --res_path results/seed${SEED}/fusion.json --eval_type fusion --lora_num 3 --batch_size 8 --seed ${SEED} > logs/fusion_seed${SEED}.log 2>&1 &
    python main.py --data_path dataset/combined_test.json --res_path results/seed${SEED}/fusion_ood.json --eval_type fusion --lora_num 3 --batch_size 8 --ood=True --seed ${SEED} > logs/fusion_ood_seed${SEED}.log 2>&1 
    python main.py --data_path dataset/combined_test.json --res_path results/seed${SEED}/selection.json --eval_type fusion --lora_num 1 --batch_size 8 --seed ${SEED} > logs/selection_seed${SEED}.log 2>&1 &
    python main.py --data_path dataset/combined_test.json --res_path results/seed${SEED}/selection_ood.json --eval_type fusion --lora_num 1 --batch_size 8 --ood=True --seed ${SEED} > logs/selection_ood_seed${SEED}.log 2>&1 
    python main.py --data_path dataset/combined_test.json --res_path results/seed${SEED}/best_selection.json --eval_type mixture --lora_num 1 --batch_size 8 --best_selection=True --seed ${SEED} > logs/best_selection_seed${SEED}.log 2>&1 
done