# ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment

![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)

## Introduction

We introduce ETS (Energy-Guided Test-Time Scaling), a training-free method for sampling directly from the optimal RL policy in masked language models. 

![](main_fig.png)

## Setup
Run the following script to setup environment.

```bash
git clone https://github.com/sheriyuo/ETS.git
cd ETS
pip install -e .
```

## Evaluation

### Autoregressive model

```bash
cd qwen
bash eval.sh
```

## Citation
