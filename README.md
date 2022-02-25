
<p align="center">
    <a href=""> <img src="pics/logo.png" width="400"/></a>
<p>
<p align="center">  
    <a href="http://neuralkg.zjukg.cn/">
        <img alt="Website" src="https://img.shields.io/badge/website-online-orange">
    </a>
    <a href="">
        <img alt="Pypi" src="https://img.shields.io/badge/release-v0.1.85-red">
    </a>
    <!-- <a href="">
        <img alt="LICENSE" src="https://img.shields.io/badge/license-MIT-brightgreen">
    </a> -->
    <a href="https://zjukg.github.io/NeuralKG/index.html">
        <img alt="Documentation" src="https://img.shields.io/badge/Doc-online-blue">
    </a>
</p>

<h1 align="center">
    <p>An Open Source Library for Diverse Representation Learning of Knowledge Graphs</p>
</h1>

NeuralKG is a python-based library for diverse representation learning of knowledge graphs implementing **Conventional KGEs**, **GNN-based KGEs**, and **Rule-based
KGEs**. We provide [comprehensive documents](https://zjukg.github.io/NeuralKG/index.html) for beginners and an [online website](http://neuralkg.zjukg.cn/) to organize an open and shared KG representation learning community.
<br>

# Table of Contents

* [What's New](#whats-new)
* [Overview](#Overview)
* [Quick Start](#quick-start)
   * [Requirements](#requirements)
* [To do](#to-do)
* [Citation](#citation)
* [Developers](#developers)

<br>

# 😃What's New

## Feb, 2022
* We have released a paper [NeuralKG: An Open Source Library for Diverse Representation Learning of Knowledge Graphs]()

<br>

# Overview

<h3 align="center">
    <img src="pics/overview.png", width="600">
</h3>
<!-- <p align="center">
    <a href=""> <img src="pics/overview.png" width="400"/></a>
<p> -->


NeuralKG is built on [PyTorch Lightning](https://www.pytorchlightning.ai/). It provides a general workflow of diverse representation learning on KGs and is highly modularized, supporting three series of KGEs. It has the following features:

+  **Support diverse types of methods.** NeuralKG, as a library for diverse representation learning of KGs, provides implementations of three series of KGE methods, including **Conventional KGEs**, **GNN-based KGEs**, and **Rule-based KGEs**.


+ **Support easy customization.** NeuralKG contains fine-grained decoupled modules that are commonly used in different KGEs, including KG Data Preprocessing, Sampler for negative sampling, Monitor for hyperparameter tuning, Trainer covering the training, and model validation.

+ **long-term technical maintenance.** The core team of NeuralKG will offer long-term technical maintenance. Other developers are welcome to pull requests.
<br>

|Components| Models |    
|:---|:--------------|
|KGEModel|[TransE](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html), [TransH](https://ojs.aaai.org/index.php/AAAI/article/view/8870), [TransR](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523/), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf), [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf), [RotatE](https://arxiv.org/abs/1902.10197), [ConvE](https://arxiv.org/abs/1707.01476), [BoxE](https://arxiv.org/pdf/2007.06267.pdf), [CrossE](https://arxiv.org/abs/1903.04750), [SimplE](https://arxiv.org/abs/1802.04868)|
|GNNModel|[RGCN](https://arxiv.org/abs/1703.06103), [KBAT](https://arxiv.org/abs/1906.01195), [CompGCN](https://arxiv.org/abs/1906.01195), [XTransE](https://link.springer.com/chapter/10.1007/978-981-15-3412-6_8)|
|RuleModel|[ComplEx-NNE+AER](https://aclanthology.org/P18-1011/), [RUGE](https://arxiv.org/abs/1711.11231), [IterE](https://arxiv.org/abs/1903.08948)|





