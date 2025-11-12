<h1 align="center">
   Look As You Think: Unifying Reasoning and Visual Evidence Attribution for Verifiable Document RAG via Reinforcement Learning
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
pip install -r requirements.txt
```

## Run
**1. Download Data**

Prepare [VISA](https://huggingface.co/collections/MrLight/visa-rag-with-visual-source-attribution) datasets.

To obtain images for the multi-candidate setup, please run `/src/image_address.py`.

**2. Cold start**
```sh
bash scripts/sft_inference.sh
```

**3. Reinforcement Learning**
```sh
bash scripts/grpo.sh
```

**4. Evaluate Model**
```sh
python test.py
```

## Acknowledgement
Our code have been developed based on [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [VISA](https://arxiv.org/abs/2412.14457). We thank their valuable works.


