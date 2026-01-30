# ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2601.21484)

## Introduction

We introduce ETS (Energy-Guided Test-Time Scaling), a training-free method for sampling directly from the optimal RL policy in masked language models (both autoregressive model and diffusion language model). 

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

```bibtex
@article{li2026ets,
  title={ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment},
  author={Xiuyu, Li and Jinkai, Zhang and Mingyang, Yi and Yu, Li and Longqiang, Wang and Yue, Wang and Ju, Fan},
  journal={arXiv preprint arXiv:2601.21484},
  year={2026}
}
```
