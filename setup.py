#!/usr/bin/env python
# coding: utf-8

import setuptools
import os


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='neuralkg',
    version='1.0.21',
    author='ZJUKG',
    author_email='xnchen2020@zju.edu.cn',
    url='https://github.com/zjukg/NeuralKG',
    description='An Open Source Library for Diverse Representation Learning of Knowledge Graphs',
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
