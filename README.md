# LoraRetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild



The official repository containing the code, data, model, for our paper accpected by ACL 2024 findings. [LoraRetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild](https://aclanthology.org/2024.findings-acl.263/).

<img src="imgs/lora_retriever_framework.jpg" alt="lora_retriever_framework" />

## Prerequisites

### Enviorment preparation

```bash
conda create --name lora_retriever python=3.10

conda activate lora_retriever

pip install -e peft/

pip install -r requirements.txt

```

## Usage

```
# evaluation script
bash eval_all.sh

# results summarization script
python summarize_results.py
```

##  Resources

### Data

The evaluation data counld be found in `./data` directory. 

### Models

We have released all of our LoRA modules [Llama-2-7b](https://huggingface.co/collections/Styxxxx/loraretriever-llama2-7b-loras-67247d00659f5ac3117f108c), [Llama-2-13b](https://huggingface.co/collections/Styxxxx/loraretriever-llama2-13b-loras-67248e72554de9192a6e1e89) on huggingface.

## Citation 

```bibtex
@inproceedings{zhao-etal-2024-loraretriever,
    title = "{L}ora{R}etriever: Input-Aware {L}o{RA} Retrieval and Composition for Mixed Tasks in the Wild",
    author = "Zhao, Ziyu  and
      Gan, Leilei  and
      Wang, Guoyin  and
      Zhou, Wangchunshu  and
      Yang, Hongxia  and
      Kuang, Kun  and
      Wu, Fei",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.263",
    doi = "10.18653/v1/2024.findings-acl.263",
    pages = "4447--4462",
}
```

## References

The code refers to the repo [LoraHub](https://github.com/sail-sg/lorahub.git), [Alapaca-LoRA](https://github.com/tloen/alpaca-lora.git), [Instructor](https://github.com/xlang-ai/instructor-embedding.git).

