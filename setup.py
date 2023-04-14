#!/usr/bin/env python
# coding: utf-8

import setuptools
import os


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='neuralkg_ind',
    version='1.0.0',
    author='ZJUKG',
    author_email='22151303@zju.edu.cn',
    url='https://github.com/zjukg/NeuralKG-ind',
    description=' A Python Library for Inductive Knowledge Graph Representation Learning',
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'pytorch_lightning==1.5.10',
        'PyYAML>=6.0',
        'wandb>=0.12.7',
        'IPython>=5.0.0'
    ],
    python_requires=">=3.6"
)
