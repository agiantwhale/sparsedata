---
layout: post
title: Intuition Expalnation of Mixture of Expert Architecture for Neural Network Capacity Improvement
---
Estimating machine learning model capacity is an important process in build a model. Model capacity effectively represents a learner's ability to estimate a function -- in formal literature, this is often explained through VC dimension, whereas in practice number of parameters in a model effectively represents this value. Using a small model can lead to underfitting, whereas using a large model can lead to overfitting and increase inference/serving costs.

This post will explain one possible solution to the prior case, the **Mixture of Expert** architecture in neural networks. This architecture can help in the following scenarios:
* Given a large dataset of distinct subpopulations (with varying underlying distribution), how can I drastically improve my model performance?
* How can I increase the model capacity of my model without incurring extra infrastructure / serving costs?

## 

## References
* [Shazeer, Mirhoseini et al. “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer” *arXiv* (2017).](https://arxiv.org/abs/1701.06538)
