import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel
import json
import numpy as np
from utils.instructor_retrieval import perform_search, initialize_index
from datasets import load_dataset
from utils.prompter import Prompter
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Prompter is a utility class to create a prompt for a given input
prompter = Prompter("alpaca")

def load_base_model(model_name_or_path='meta-llama/Llama-2-7b-hf'):
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

def init_vector_db(config_path='./config/config2.json'):
    """
    Initialize the vector database with configurations from the specified JSON file.
    """
    with open(config_path, 'r') as file:
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
        
        # Check what was loaded
        print(f"Adapter config: {peft_model.peft_config[f'adapter{i}']}")

    print(f"\nSetting adapters: {lora_lists}")
    peft_model.set_adapter(lora_lists)
    
    # Debug: Check active adapters and their structure
    print(f"Active adapters after set_adapter: {peft_model.active_adapters}")
    
    # Check a specific layer to see what's there
    for name, module in peft_model.named_modules():
        if 'q_proj' in name and hasattr(module, 'lora_A'):
            print(f"\n{name}:")
            print(f"  lora_A type: {type(module.lora_A)}")
            if isinstance(module.lora_A, dict):
                print(f"  Keys in lora_A: {module.lora_A.keys()}")
                for key in module.lora_A.keys():
                    if module.lora_A[key] is not None:
                        weight_shape = module.lora_A[key].weight.shape if hasattr(module.lora_A[key], 'weight') else 'No weight attr'
                        print(f"    {key}: {weight_shape}")
            break  # Just check first q_proj layer
    
    peft_model = peft_model.to(device)
    peft_model.eval()
    return peft_model

def check_adapter_compatibility(lora_module_list):
    """Check if all adapters have the same target modules"""
    from peft import PeftConfig
    
    configs = []
    target_modules = []
    
    for lora_model in lora_module_list:
        config = PeftConfig.from_pretrained(lora_model)
        configs.append(config)
        print(f"\n{lora_model}:")
        print(f"  Target modules: {config.target_modules}")
        print(f"  LoRA rank (r): {config.r}")
        print(f"  LoRA alpha: {config.lora_alpha}")
        
        # Handle different types of target_modules
        if isinstance(config.target_modules, set):
            target_modules.append(config.target_modules)
        elif isinstance(config.target_modules, list):
            target_modules.append(set(config.target_modules))
        else:
            target_modules.append({config.target_modules})
    
    # Check if all have same target modules
    unique_targets = set(frozenset(tm) for tm in target_modules)
    
    if len(unique_targets) > 1:
        print("\n⚠️ WARNING: Adapters have different target modules!")
        for i, tm in enumerate(target_modules):
            print(f"  Adapter {i}: {tm}")
    else:
        print("\n✓ All adapters target the same modules")
        print(f"  Common target modules: {target_modules[0]}")
    
    # Check for different ranks or alphas
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

# Call before loading

def eval_datasets(
    data_path, 
    res_path, 
    config_path="config/config2.json", 
    eval_type="fusion", 
    lora_num=3, 
    batch_size=1, 
    ood=False, 
    best_selection=False, 
    model_size='7b',
    seed=None
):
    """
    Evaluate the model on given datasets.

    Parameters:
    - data_path: Path to the evaluation dataset.
    - res_path: Path to save the evaluation results.
    - config_path: Path to configuration file for vector DB initialization.
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
            
    correct_count = 0
    results = []  # Initialize a list to store question and response data
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize vector database for retrieval
    init_vector_db(config_path)

    def generate_and_tokenize_prompt(data_point):
        """
        Generate the full prompt for a given data point and return it.
        """
        full_prompt = prompter.generate_prompt(
            data_point["inputs"],
            "",
            "",
        )
        return {"full_prompt": full_prompt}

    # Load the dataset
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path)
    else:
        dataset = load_dataset(data_path)

    # Prepare the dataset with full prompts
    eval_data = dataset["train"].map(generate_and_tokenize_prompt)

    model_path = f"meta-llama/Llama-2-{model_size}-hf"
    base_model, tokenizer = load_base_model(model_path)
    base_model.eval()

    with torch.no_grad():
        with tqdm(total=len(dataset["train"]), desc="Evaluating", unit="item") as pbar:
            for i in range(0, len(eval_data["full_prompt"]), batch_size):
                input_text = eval_data["inputs"][i : i + batch_size]
                task_names = eval_data["task"][i : i + batch_size]

                # If out-of-domain filtering is required, specify exclusion list
                exclude_list = None
                if ood:
                    if model_size == '7b':
                        exclude_list = [f"Styxxxx/llama2_7b_lora-{task}" for task in task_names]
                    else:
                        exclude_list = [f"Styxxxx/llama2_13b_lora-{task}" for task in task_names]

                # Perform retrieval to get top-k LoRA modules
                module_list, mapping_matrix = perform_search(input_text, k=lora_num, exclude_list=exclude_list)
                input_text = eval_data["full_prompt"][i : i + batch_size]

                # If best_selection is True, re-map module_list and mapping_matrix for a more constrained set
                if best_selection:
                    if model_size == '7b':
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
                    raise ValueError("mapping_matrix is None. Retrieval may have failed or returned no adapters.")
                mapping_matrix_tensor = torch.tensor(mapping_matrix).to(device)
                print("mapping_matrix_tensor.shape:", mapping_matrix_tensor.shape)
                print("Number of adapters loaded:", len(module_list))
                if mapping_matrix_tensor.shape[1] != len(module_list):
                    raise ValueError(f"Shape mismatch: mapping_matrix_tensor.shape[1] ({mapping_matrix_tensor.shape[1]}) != number of adapters ({len(module_list)}). Please check retrieval logic.")
                mapping_matrix_tensor = mapping_matrix_tensor.to(torch.bfloat16)
                mapping_matrix_tensor /= lora_num
                
                configs = check_adapter_compatibility(module_list)
                # Load the PEFT model with selected adapters
                peft_model = load_peft_model(module_list, base_model)

                # Tokenize the input text
                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                
                print("pin1")
                # Generate model outputs with given parameters
                try:
                    outputs = peft_model.generate(
                        input_ids=inputs["input_ids"],
                        max_new_tokens=50,
                        temperature=0.001,
                        merging_type=eval_type,
                        lora_mapping=mapping_matrix_tensor
                        # module_list=module_list  # Ensure adapter stacking order matches mapping matrix
                    )
                except Exception as e:
                    print("exception", e)
                    continue
                
                # Process and store results
                for j, (output, expected_answer) in enumerate(zip(outputs, eval_data["targets"][i : i + batch_size])):
                    generated_answer = tokenizer.decode(output, skip_special_tokens=True)
                    generated_answer = generated_answer.strip().split('### Response:\n')[-1]

                    sample = {
                        'inputs': eval_data["inputs"][i+j],
                        'targets': eval_data["targets"][i+j],
                        'metric': eval_data["metric"][i+j],
                        'domain': eval_data["domain"][i+j],
                        'task': eval_data["task"][i+j],
                        'predicted_answer': generated_answer
                    }
                    results.append(sample)

                    print(f"generated_answer: {generated_answer}, expected_answer: {expected_answer}")

                pbar.set_description("Evaluating")
                peft_model.unload()

    # Save the results to a JSON file
    import os
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import fire
    fire.Fire(eval_datasets)
