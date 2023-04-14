
<p align="center">
    <a href=""> <img src="pics/logo.png" width="400"/></a>
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
    <p>用于归纳式知识图谱表示学习的Python工具包</p>
</h1>

NeuralKG-ind是一个基于python的归纳知识图表示学习工具包，它包括**标准化过程**、**丰富的现有模型**、**解耦模块**和**综合评估指标**。我们为初学者提供了详细的[文档](https://zjukg.github.io/NeuralKG-ind/neuralkg_ind.model.html)。
<br>

# 目录

- [目录](#目录)
- [😃最新消息](#最新消息)
  - [2022年2月](#2022年2月)
- [工具概览](#工具概览)
- [运行示例](#运行示例)
- [实现模型](#实现模型)
- [快速上手](#快速上手)
  - [下载](#下载)
  - [模型训练](#模型训练)
  - [模型测试](#模型测试)
  - [超参调节](#超参调节)
- [复现结果](#复现结果)
- [Notebook教程](#notebook教程)
- [详细文档](#详细文档)
- [NeuralKG-ind核心团队](#neuralkg-ind核心团队)
<!-- * [To do](#to-do) -->

<br>

# 😃最新消息

## 2022年2月
* 我们发布了关于该工具包的论文 **NeuralKG-ind: A Python Library for Inductive Knowledge Graph Representation Learning**

<br>

# 工具概览

<h3 align="center">
    <img src="pics/overview.png", width="600">
</h3>
<!-- <p align="center">
    <a href=""> <img src="pics/overview.png" width="400"/></a>
<p> -->


NeuralKG-ind工具包整体基于[PyTorch Lightning](https://www.pytorchlightning.ai/)框架，同时在[NeuralKG](https://github.com/zjukg/NeuralKG)的基础上进行开发，它提供了一个用于处理知识图谱上归纳任务的模型的通用工作流程。Neuralkg-ind具有以下特性：


+  **标准化过程。** 根据现有方法，我们对构建归纳知识图谱表示学习模型的整个过程进行了标准化，包括数据处理、采样和训练以及链接预测和三元组分类任务的评估。我们还提供辅助功能，包括日志管理和超参数调整，用于模型训练和分析。


+ **丰富的现有方法。** 我们重新实现了最近3年提出的5种归纳知识图谱表示学习的方法，包括[GraIL](https://arxiv.org/abs/1911.06962)，[CoMPILE](https://arxiv.org/pdf/2012.08911)，[SNRI](https://arxiv.org/abs/2208.00850)，[RMPI](https://arxiv.org/abs/2210.03994)和[MorsE](https://arxiv.org/abs/2110.14170)，用户可以快速应用这些模型。

+ **解耦模块。** 我们提供了许多解耦模块，如子图提取、节点标记、邻居聚合、图神经网络层和KGE评分方法，使用户能够快速地构建新的归纳知识图谱表示学习模型。

+ **长期支持。** 我们为NeuralKG-ind提供长期的技术支持，包括维护使用文档、创建更好的使用体验、添加新的模型、解决存在的问题以及处理pull request。

<br>

# 运行示例
NeuralKG-ind在自定义知识图谱demo_kg上运行的示例。
<!-- ![框架](./pics/demo.gif) -->
<img src="pics/demo_l.gif">
<!-- <img src="pics/demo.gif" width="900" height="476" align=center> -->

<br>

# 实现模型

<table>
    <tr>  
        <th rowspan="1">方法</th><th colspan="1">类别</th><th colspan="1">模型</th>
    </tr>
    <tr>
        <td rowspan="3">传统知识图谱表示学习方法</td><td>传统知识图谱嵌入</td><td><a href="https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html">TransE</a>, <a href="https://ojs.aaai.org/index.php/AAAI/article/view/8870">TransH</a>, <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523/">TransR</a>, <a href="http://proceedings.mlr.press/v48/trouillon16.pdf">ComplEx</a>, <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf">DistMult</a>, <a href="https://arxiv.org/abs/1902.10197">RotatE</a>, <a href="https://arxiv.org/abs/1707.01476">ConvE</a>, <a href="https://arxiv.org/pdf/2007.06267.pdf">BoxE</a>, <a href="https://arxiv.org/abs/1903.04750">CrossE</a>, <a href="https://arxiv.org/abs/1802.04868">SimplE</a>, <a href="https://arxiv.org/abs/1911.09419">HAKE</a>, <a href="https://arxiv.org/pdf/2011.03798.pdf">PairRE</a>, <a href="https://ojs.aaai.org/index.php/AAAI/article/view/16850">DualE</a></td>
        <tr>
            <td>基于图神经网络的知识图谱嵌入</td><td><a href="https://arxiv.org/abs/1703.06103">RGCN</a>, <a href="https://arxiv.org/abs/1906.01195">KBAT</a>, <a href="https://arxiv.org/abs/1906.01195">CompGCN</a>, <a href="https://link.springer.com/chapter/10.1007/978-981-15-3412-6_8">XTransE</a></td>
        </tr>
        <tr>
            <td>基于规则的知识图谱嵌入</td><td><a href="https://aclanthology.org/P18-1011/">ComplEx-NNE+AER</a>, <a href="https://arxiv.org/abs/1711.11231">RUGE</a>, <a href="https://arxiv.org/abs/1903.08948">IterE</a></td>
        </tr>
    </tr>
    <tr>
        <td><strong>归纳式知识图谱表示学习方法</strong></td><td><strong>基于图神经网络的归纳式知识图谱嵌入</strong></td><td><a href="https://arxiv.org/abs/1911.06962">GraIL</a>, <a href="https://arxiv.org/pdf/2012.08911">CoMPILE</a>, <a href="https://arxiv.org/abs/2208.00850">SNRI</a>, <a href="https://arxiv.org/abs/2210.03994">RMPI</a>, <a href="https://arxiv.org/abs/2110.14170">MorsE</a>
    </tr>
</table>
<br>

# 快速上手

## 下载

**Step1** 使用 ```Anaconda``` 创建虚拟环境，并进入虚拟环境

```bash
conda create -n neuralkg-ind python=3.8
conda activate neuralkg-ind
```
**Step2** 下载适用您CUDA版本的的PyTorch和DGL，下面我们提供一个基于CUDA 11.1的下载样例 

+  下载PyTorch
```
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
+ 下载DGL
```
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
```
+ 下载lmdb
```
pip install lmdb==1.4.0
```
+ 下载sklearn
```
pip install scikit-learn==1.2.1
```
**Step3** 安装NeuralKG-ind

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
## 模型训练
```
# Use bash script
sh ./scripts/your-sh

# Use config
python main.py --load_config --config_path <your-config>
```

## 模型测试
```
# Testing AUC and AUC-PR 
python main.py --test_only --checkpoint_dir <your-model-path> --eval_task triple_classification 

# Testing MRR and hit@1,5,10
python main.py --test_only --checkpoint_dir <your-model-path> --eval_task link_prediction --test_db_path <your-db-path> 
```
## 超参调节
NeuralKG-ind使用[Weights&Biases](https://wandb.ai/site)进行超参数调节，支持多种超参优化例如网格搜索、随机搜索和贝叶斯优化。搜索类型和搜索空间可以通过配置（*.yaml）文件进行设置。

下面展示了在FB15k-237上训练Grail，并使用贝叶斯搜索（bayes search）进行超参数调节的配置文件：

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

# 复现结果
下面展示了使用NeuralKG-ind的不同模型在FB15k-237上的结果和在NELL-995上的部分结果，更多结果请访问[此处](https://zjukg.github.io/NeuralKG-ind/result.html)。


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

# Notebook教程


😃我们使用colab提供部分notebook供用户使用我们的工具包

[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r1SpUI60CukwUq8KabEQ2Gq0UyWyI0QM?usp=sharing)

<br>

# 详细文档
https://zjukg.github.io/NeuralKG-ind/neuralkg_ind.model.html


<!-- <br> -->

<!-- # To do -->

<br>


# NeuralKG-ind核心团队

**浙江大学**: 张文，姚祯，陈名杨，黄志伟，陈华钧

