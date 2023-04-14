
<p align="center">
    <a href=""> <img src="pics/logo-ind.png" width="400"/></a>
<p>
<p align="center">  
    <a href="https://pypi.org/project/neuralkg-ind/">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/neuralkg-ind">
    </a>
    <a href="https://github.com/zjukg/NeuralKG-ind/blob/main/LICENSE">
        <img alt="Pypi" src="https://img.shields.io/badge/license-Apache--2.0-yellowgreen">
    </a>
    <!-- <a href="">
        <img alt="LICENSE" src="https://img.shields.io/badge/license-MIT-brightgreen">
    </a> -->
    <a href="https://zjukg.github.io/NeuralKG-ind/neuralkg_ind.model.html">
        <img alt="Documentation" src="https://img.shields.io/badge/Doc-online-blue">
    </a>
</p>

<h1 align="center">
    <p>A Python Library for Inductive Knowledge Graph Representation Learning</p>
</h1>
<p align="center">
    <b> English | <a href="https://github.com/zjukg/NeuralKG-ind/blob/main/README_CN.md">中文</a> </b>
</p>

NeuralKG-ind is a python-based library for inductive knowledge graph representation learning, which includes **standardized processes**, **rich existing methods**, **decoupled modules**, and **comprehensive evaluation metrics**. We provide [comprehensive documents](https://zjukg.github.io/NeuralKG-ind/neuralkg_ind.model.html) for beginners.

<br>

# Table of Contents

- [Table of Contents](#table-of-contents)
- [😃What's New](#whats-new)
  - [Feb, 2023](#feb-2023)
- [Overview](#overview)
- [Demo](#demo)
- [Implemented Methods](#implemented-methods)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Reproduced Results](#reproduced-results)
- [Notebook Guide](#notebook-guide)
- [Detailed Documentation](#detailed-documentation)
- [NeuralKG-ind Core Team](#neuralkg-ind-core-team)
<!-- * [To do](#to-do) -->


<br>

# 😃What's New

## Feb, 2023
* We have released a paper **NeuralKG-ind: A Python Library for Inductive Knowledge Graph Representation Learning**

<br>

# Overview

<h3 align="center">
    <img src="pics/overview.png", width="600">
</h3>
<!-- <p align="center">
    <a href=""> <img src="pics/overview.png" width="400"/></a>
<p> -->


NeuralKG-ind is built on [PyTorch Lightning](https://www.pytorchlightning.ai/) and based on [NeuralKG](https://github.com/zjukg/NeuralKG). It provides  a general workflow for developing models handling inductive tasks on KGs. It has the following features:

+  **Standardized process.** According to existing methods, we standardized the overall process of constructing an inductive knowledge graph representation learning model, including data processing, sampler and trainer construction, and evaluation of link prediction and triple classification tasks. We also provide auxiliary functions, including log management and hyper-parameter tuning, for model training and analysis.


+  **Rich existing methods.** We re-implemented 5 inductive knowledge graph representation learning methods proposed in recent 3 years, including [GraIL](https://arxiv.org/abs/1911.06962), [CoMPILE](https://arxiv.org/pdf/2012.08911), [SNRI](https://arxiv.org/abs/2208.00850), [RMPI](https://arxiv.org/abs/2210.03994) and [MorsE](https://arxiv.org/abs/2110.14170), enabling users to apply these models off the shelf.

+  **Decoupled modules.** We provide a lot of decoupled modules, such as the subgraph extraction function, the node labeling function, neighbor aggregation functions, compound graph neural network layers, and KGE score functions, enabling users to construct a new inductive knowledge graph representation learning model faster.

+  **Long-term supports.** We provide long-term support on NeuralKG-ind, including maintaining detailed documentation, creating straightforward quick-start, adding new models, solving issues, and dealing with pull requests.

<br>

# Demo

There is a demonstration of NeuralKG-ind.
<!-- ![框架](./pics/demo.gif) -->
<img src="pics/demo_l.gif">
<!-- <img src="pics/demo.gif" width="900" height="476" align=center> -->

<br>

# Implemented Methods

<table>
    <tr>  
        <th rowspan="1">Pattern</th><th colspan="1">Components</th><th colspan="1">Models</th>
    </tr>
    <tr>
        <td rowspan="3">Traditional KGRL method</td><td>Conventional KGEs</td><td><a href="https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html">TransE</a>, <a href="https://ojs.aaai.org/index.php/AAAI/article/view/8870">TransH</a>, <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523/">TransR</a>, <a href="http://proceedings.mlr.press/v48/trouillon16.pdf">ComplEx</a>, <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf">DistMult</a>, <a href="https://arxiv.org/abs/1902.10197">RotatE</a>, <a href="https://arxiv.org/abs/1707.01476">ConvE</a>, <a href="https://arxiv.org/pdf/2007.06267.pdf">BoxE</a>, <a href="https://arxiv.org/abs/1903.04750">CrossE</a>, <a href="https://arxiv.org/abs/1802.04868">SimplE</a>, <a href="https://arxiv.org/abs/1911.09419">HAKE</a>, <a href="https://arxiv.org/pdf/2011.03798.pdf">PairRE</a>, <a href="https://ojs.aaai.org/index.php/AAAI/article/view/16850">DualE</a></td>
        <tr>
            <td>GNN-based KGEs</td><td><a href="https://arxiv.org/abs/1703.06103">RGCN</a>, <a href="https://arxiv.org/abs/1906.01195">KBAT</a>, <a href="https://arxiv.org/abs/1906.01195">CompGCN</a>, <a href="https://link.springer.com/chapter/10.1007/978-981-15-3412-6_8">XTransE</a></td>
        </tr>
        <tr>
            <td>Rule-based KGEs</td><td><a href="https://aclanthology.org/P18-1011/">ComplEx-NNE+AER</a>, <a href="https://arxiv.org/abs/1711.11231">RUGE</a>, <a href="https://arxiv.org/abs/1903.08948">IterE</a></td>
        </tr>
    </tr>
    <tr>
        <td><strong>Inductive KGRL method</strong></td><td><strong>GNN-based inductive models</strong></td><td><a href="https://arxiv.org/abs/1911.06962">GraIL</a>, <a href="https://arxiv.org/pdf/2012.08911">CoMPILE</a>, <a href="https://arxiv.org/abs/2208.00850">SNRI</a>, <a href="https://arxiv.org/abs/2210.03994">RMPI</a>, <a href="https://arxiv.org/abs/2110.14170">MorsE</a>
    </tr>
</table>

<br>

# Quick Start

## Installation

**Step1** Create a virtual environment using ```Anaconda``` and enter it
```bash
conda create -n neuralkg-ind python=3.8
conda activate neuralkg-ind
```
**Step2** Install the appropriate PyTorch and DGL according to your cuda version

Here we give a sample installation based on cuda == 11.1

+  Install PyTorch
```
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
+ Install DGL
```
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
```
+ Install lmdb
```
pip install lmdb==1.4.0
```
+ Install sklearn
```
pip install scikit-learn==1.2.1
```
**Step3** Install package

+ From Pypi
```bash
pip install neuralkg_ind
```

+ From Source
```bash
git clone https://github.com/zjukg/NeuralKG-ind.git
cd NeuralKG-ind
python setup.py install
```
## Training
```
# Use bash script
sh ./scripts/your-sh

# Use config
python main.py --load_config --config_path <your-config>
```

## Evaluation
```
# Testing AUC and AUC-PR 
python main.py --test_only --checkpoint_dir <your-model-path> --eval_task triple_classification 

# Testing MRR and hit@1,5,10
python main.py --test_only --checkpoint_dir <your-model-path> --eval_task link_prediction --test_db_path <your-db-path> 
```
## Hyperparameter Tuning
NeuralKG-ind utilizes [Weights&Biases](https://wandb.ai/site) supporting various forms of hyperparameter optimization such as grid search, Random search, and Bayesian optimization. The search type and search space are specified in the configuration file in the format "*.yaml" to perform hyperparameter optimization.

The following config file displays hyperparameter optimization of the Grail on the FB15K-237 dataset using bayes search:
```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args}
program: main.py
method: bayes
metric:
  goal: maximize
  name: Eval|auc
parameters:
  dataset_name:
    value: FB15K237
  model_name:
    value: Grail
  loss_name:
    values: [Margin_Loss]
  train_sampler_class:
    values: [SubSampler]
  emb_dim:
    values: [32, 64]
  lr:
    values: [1e-2, 5e-3, 1e-3]
  train_bs:
    values: [64, 128]
  num_neg:
    values: [16, 32]
```
<br>

# Reproduced Results
There are some reproduced model results on FB15K-237 dataset and partial results on NELL-995 using NeuralKG as below. See more results in [here](https://zjukg.github.io/NeuralKG-ind/result.html)


<table>
    <tr>  
        <th rowspan="2">Method</th><th colspan="6">FB15K-237_v1</th><th colspan="6">FB15K-237_v2</th>
    </tr>
    <tr>  
       <th>AUC</th><th>AUC-PR</th><th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th><th>AUC</th><th>AUC-PR</th><th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th>
    </tr>
    <tr>
        <td>GraIL</td><td>0.802</td><td>0.821</td><td>0.452</td><td>0.359</td><td>0.561</td><td>0.624</td><td>0.873</td><td>0.900</td><td>0.642</td><td>0.539</td><td>0.767</td><td>0.831</td>
    </tr>
    <tr>
        <td>CoMPILE</td><td>0.800</td><td>0.835</td><td>0.516</td><td>0.437</td><td>0.600</td><td>0.668</td><td>0.876</td><td>0.905</td><td>0.617</td><td>0.509</td><td>0.741</td><td>0.813</td>
    </tr>
    <tr>
        <td>SNRI</td><td>0.792</td><td>0.883</td><td>0.495</td><td>0.390</td><td>0.600</td><td>0.720</td><td>0.884</td><td>0.906</td><td>0.646</td><td>0.536</td><td>0.781</td><td>0.857</td>
    </tr>
    <tr>
        <td>RMPI</td><td>0.803</td><td>0.823</td><td>0.532</td><td>0.451</td><td>0.620</td><td>0.689</td><td>0.851</td><td>0.882</td><td>0.632</td><td>0.523</td><td>0.763</td><td>0.830</td>
    </tr>
    <tr>
        <td>MorsE</td><td>0.844</td><td>0.847</td><td>0.591</td><td>0.470</td><td>0.723</td><td>0.833</td><td>0.963</td><td>0.960</td><td>0.754</td><td>0.643</td><td>0.897</td><td>0.950</td>
    </tr>
</table>

<br>

<table>
    <tr>  
        <th rowspan="2">Method</th><th colspan="6">FB15K-237_v3</th><th colspan="6">FB15K-237_v4</th>
    </tr>
    <tr>  
       <th>AUC</th><th>AUC-PR</th><th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th><th>AUC</th><th>AUC-PR</th><th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th>
    </tr>
    <tr>
        <td>GraIL</td><td>0.871</td><td>0.899</td><td>0.637</td><td>0.530</td><td>0.765</td><td>0.828</td><td>0.911</td><td>0.921</td><td>0.639</td><td>0.521</td><td>0.797</td><td>0.880</td>
    </tr>
    <tr>
        <td>CoMPILE</td><td>0.906</td><td>0.925</td><td>0.670</td><td>0.568</td><td>0.796</td><td>0.859</td><td>0.927</td><td>0.932</td><td>0.704</td><td>0.604</td><td>0.831</td><td>0.894</td>
    </tr>
    <tr>
        <td>SNRI</td><td>0.870</td><td>0.884</td><td>0.642</td><td>0.525</td><td>0.775</td><td>0.871</td><td>0.899</td><td>0.916</td><td>0.681</td><td>0.573</td><td>0.821</td><td>0.894</td>
    </tr>
    <tr>
        <td>RMPI</td><td>0.876</td><td>0.866</td><td>0.662</td><td>0.569</td><td>0.767</td><td>0.827</td><td>0.905</td><td>0.916</td><td>0.647</td><td>0.535</td><td>0.787</td><td>0.866</td>
    </tr>
    <tr>
        <td>MorsE</td><td>0.959</td><td>0.952</td><td>0.745</td><td>0.637</td><td>0.878</td><td>0.954</td><td>0.963</td><td>0.952</td><td>0.742</td><td>0.662</td><td>0.888</td><td>0.958</td>
    </tr>
</table>

<br>

<table>
    <tr>  
        <th rowspan="2">Method</th><th colspan="6">NELL-995_v1</th><th colspan="6">NELL-995_v2</th>
    </tr>
    <tr>  
       <th>AUC</th><th>AUC-PR</th><th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th><th>AUC</th><th>AUC-PR</th><th>MRR</th><th>Hits@1</th><th>Hit@5</th><th>Hit@10</th>
    </tr>
    <tr>
        <td>GraIL</td><td>0.814</td><td>0.750</td><td>0.467</td><td>0.395</td><td>0.515</td><td>0.575</td><td>0.929</td><td>0.947</td><td>0.735</td><td>0.624</td><td>0.884</td><td>0.933</td>
    </tr>
    <tr>
        <td>SNRI</td><td>0.737</td><td>0.720</td><td>0.523</td><td>0.475</td><td>0.545</td><td>0.595</td><td>0.864</td><td>0.884</td><td>0.630</td><td>0.507</td><td>0.774</td><td>0.863</td>
    </tr>
</table>



<br>

# Notebook Guide


😃We use colab to provide some notebooks to help users use our library.

[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r1SpUI60CukwUq8KabEQ2Gq0UyWyI0QM?usp=sharing)


<br>

# Detailed Documentation
https://zjukg.github.io/NeuralKG-ind/neuralkg_ind.model.html


<!-- <br> -->

<!-- # To do -->

<br>


# NeuralKG-ind Core Team 
**Zhejiang University**: Wen Zhang, Zhen Yao, Mingyang Chen, Zhiwei Huang, Huajun Chen


