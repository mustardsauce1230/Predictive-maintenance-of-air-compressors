---
base_model: tiiuae/falcon-rw-1b
library_name: peft
model_name: falcon-finetuned
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for falcon-finetuned

This model is a fine-tuned version of [tiiuae/falcon-rw-1b](https://huggingface.co/tiiuae/falcon-rw-1b).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/taanmallik3000-jadavpur-university/huggingface/runs/g2y04n2w) 


This model was trained with SFT.

### Framework versions

- PEFT 0.15.2
- TRL: 0.19.0
- Transformers: 4.53.0
- Pytorch: 2.6.0+cu124
- Datasets: 3.6.0
- Tokenizers: 0.21.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```