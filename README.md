<h1 align="center">
   <img src="./fig/title.jpg" alt="" width="4%"> Look As You Think: Unifying Reasoning and Visual Evidence Attribution for Verifiable Document RAG via Reinforcement Learning
</h1>

This is the official code of the paper **Look As You Think: Unifying Reasoning and Visual Evidence Attribution for Verifiable Document RAG via Reinforcement Learning** by Shuochen Liu, Pengfei Luo, Chao Zhang, Yuhao Chen, Haotian Zhang, Qi Liu, Xin Kou, Tong Xu, Enhong Chen (**Accepted as Poster of AAAI'2026**)

## Overview

TL;DR: In this paper, we introduce the **Chain of Evidence (CoE)** paradigm, which models stepwise inference by grounding each chain-of-thought (CoT) reasoning step. To realize CoE, we propose **Look As You Think (LAT)**, a two-stage reinforcement learning (RL) framework that trains VLMs to unify CoT reasoning and visual grounding by generating progressive reasoning process paired with an aligned visual attribution for each reference element.

<p align="center"><img src="./fig/framework.jpg" alt="" width="80%"></p>


If you find this repository or paper useful, you can cite:
```
Coming soon
```


## Dependencies

The required dependencies and their versions can be found in the [`requirements.txt`](requirements.txt). 
To install all the required packages along with their dependencies, run
```sh
# python >= 3.10

pip install -r requirements.txt
```

## Run
**1. Download Data**

Prepare [VISA](https://huggingface.co/collections/MrLight/visa-rag-with-visual-source-attribution) datasets. Place the downloaded datasets under the `/data/visa/`(paper/wiki/fine-web) directories. Modify the paths as necessary to match your local environment.

To obtain images for the multi-candidate setup, please run [/src/image_address.py](/src/image_address.py).

**2. Cold start**


```sh
bash scripts/sft_inference.sh
```
> [!Important]
> 1. After each training session, merge the LoRA parameters by executing the following code.
> 2. For multi-image training scenarios, initialize the multi-image model using the single-image trained version. Subsequently, perform supervised fine-tuning (SFT) on the multi-image CoE data in $\mathcal{D}_{\text{final}}$, fine-tuning only the LoRA adapter of the language model while keeping the vision transformer (ViT) frozen to minimize GPU memory consumption.
```
from peft import PeftModel

model = PeftModel.from_pretrained(model, lora_name_or_path)
model = model.merge_and_unload()
model.save_pretrained("merged_model")
```


**3. Reinforcement Learning**
```sh
bash scripts/grpo.sh
```
After SFT training, the LoRA parameters need to be merged into the base model, and the ·model_name_or_path· should be updated accordingly.


**4. Evaluate Model**
```sh
python src/evaluation.py
```
During evaluation, it is necessary to manually specify the model and the corresponding LoRA parameters to reproduce the results, while selecting the appropriate evaluation settings and dataset names.


## Case

<p align="center"><img src="./fig/case1.jpg" width="80%"></p>


More details and analyses about experimental results can be found in our paper.






## Acknowledgement
Our code have been developed based on [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [VISA](https://arxiv.org/abs/2412.14457). We thank their valuable works.


