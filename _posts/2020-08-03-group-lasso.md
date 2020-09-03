---
layout: post
title: Intuitive Explanation of Group LASSO Regularization for Neural Network Interpretability
---
Neural networks are often referred to as a black box model because of its lack of interpretability. Most of a network's operations occur in the hidden layers and latent space. As a result, tracing important features in a dataset is not an easy task, especially when the number of features is large. This is often a limiting factor in applying neural networks in fields where explainability is not only favored, but crucial (such as medical diagnostics or finance).

Through this entry, we hope to examine the application of the group LASSO regularization for solving the problems described above.

## What is ridge and LASSO regularization?
The loss function of ridge regression can be defined as

{% latex class=center %}
\begin{align*}
\arg \min_{\beta_0, \beta} \Sigma_{i=1}^n (y_i - (x_i^T\beta + \beta_0))^2 + \lambda \Sigma_{j=1}^p \beta_j^2
\end{align*}
{% endlatex %}

while loss function of LASSO regression can be defined as

{% latex class=center %}
\begin{align*}
\arg \min_{\beta_0, \beta} \Sigma_{i=1}^n (y_i - (x_i^T\beta + \beta_0))^2 + \lambda \Sigma_{j=1}^p \lvert \beta_j \rvert
\end{align*}
{% endlatex %}

The above loss functions can be broken down into
* Predicted output: {% latex %}$x_i^T\beta + \beta_0${% endlatex %}
* Regularization term
    * {% latex %}$\lambda \Sigma_{j=1}^p \lvert \beta_j \rvert${% endlatex %} for LASSO
    * {% latex %}$\lambda \Sigma_{j=1}^p \beta_j^2${% endlatex %} for ridge

Comparison of the two regularization terms shows the intuition behind LASSO regression's better interpretability characteristics. From a Big-O standpoint, {% latex %}$\mathcal{O}(\beta_j^2) > \mathcal{O}(\lvert \beta_j \rvert)${% endlatex %}. The penalty for having one skewed large value is much greater for ridge regression. Ridge regularization aims to reduce variance between the coefficients, therefore driving all features down to zero.

LASSO regularization, on the other hand, will set some feature's coefficients to zero values when deemed necessary, effectively removing them. We then can compare non-zero coefficients to determine the importance of the features.

## What is group LASSO regularization?
From the above example, we observe how LASSO regularization can help with the interpretability of the model. But some problems may benefit from a group of features used together, especially when incorporating domain knowledge into the model.

Group LASSO attempts to solve this problem by separating the entire feature set into separate feature groups. The regularization function can be written as

{% latex class=center %}
\begin{align*}
\arg \min_{\beta_0, \beta} \Sigma_{i=1}^n (y_i - (x_i^T\beta + \beta_0))^2 + \lambda \Sigma_{g=1}^G \sqrt{d_g} \lvert\lvert \beta^g \rvert\rvert_2
\end{align*}
{% endlatex %}

where
* {% latex %}$d_g${% endlatex %} denotes the size of the group.
* {% latex %}$\lvert\lvert \beta^g \rvert\rvert_2${% endlatex %} denotes the L2-norm of the feature group {% latex %}$\beta^g${% endlatex %}.

Let's take a closer look at the regularization term {% latex %}$\lambda \Sigma_{g=1}^G \sqrt{d_g} \lvert\lvert \beta^g \rvert\rvert_2${% endlatex %}.

Note that {% latex %}$\lvert\lvert \beta^g \rvert\rvert_2 \geq 0${% endlatex %}, and we for some {% latex %}$R${% endlatex %} that satisfies {% latex %}$\sqrt{d_g} \lvert\lvert \beta^g \rvert\rvert_2 = \lvert R \rvert${% endlatex %}, we could effectively rewrite the equation as

{% latex class=center %}
\begin{align*}
\lambda \Sigma_{g=1}^G \sqrt{d_g} \lvert\lvert \beta^g \rvert\rvert_2 = \lambda \Sigma_{g=1}^G \lvert R \rvert
\end{align*}
{% endlatex %}

In this case, we have effectively reduced the regularization to LASSO regularization on the inter-group level.

Similarly, let's take a look an subgroup. Expanding the term for some group with cardinality {% latex %}$p${% endlatex %}, the regularization term can be expressed as

{% latex class=center %}
\begin{align*}
\arg \min_{\beta} \lambda \Sigma_{g=1}^G \sqrt{d_g} \lvert\lvert \beta^g \rvert\rvert_2 
&\equiv \arg \min_{\beta^g} \sqrt{d_g} \lvert\lvert \beta^g \rvert\rvert_2  \hspace{10px} (\because \forall \lvert\lvert \beta^g \rvert\rvert_2 >= 0) \\
&= \arg \min_{\beta^g} \sqrt{p} \sqrt{b_1^2 + b_2^2 + b_3^2 + \cdots + b_p^2} \\ 
&= \arg \min_{\beta^g} \sqrt{p} \sqrt{\Sigma_{i=1}^p b_i^2} \\ 
&\equiv \lambda \arg \min_{b} \Sigma_{i=1}^p b_i^2
\end{align*}
{% endlatex %}

Here, we have effectively reduced the regularization to ridge regularization on the intra-group level.

We build on the intuition that while it cannot select certain features within the same group, because of it's LASSO-like nature between feature groups, the model will zero-out entirety of certain coefficient groups.

Additionally, note the two following characteristics:
* When {% latex %}$d_g = 1${% endlatex %}, the regularization term essentially becomes a LASSO (L1) regularization.
* When {% latex %}$G = 1${% endlatex %}, the regularization term essentially becomes a ridge (L2) regularization.

## How can we adapt group LASSO for neural networks?
Up to now, the application of regularization terms have been on linear regression methods where each features are assigned a single coefficient weight. Now, we will take a look at a neural network, specifically on the connections between the first two layer of the network, where each individual features have multiple weights associated to the next layer.

To visualize this, say we have a small neural network with one hidden layer.

```
digraph G {
    graph [bgcolor=black, fontcolor=white];
    node [label=""];
    edge[arrowhead=vee, arrowtail=inv, arrowsize=.7, color=white, fontsize=12,
            fontcolor=white]
    rankdir=LR
    splines=line
    nodesep=.15;
        HelloWorld$
    subgraph cluster_0 {
        node [style=filled,color=white, shape=circle];
		x1[label="x_1"] x2[label="x_2"];
		label="Input features";
	}

	subgraph cluster_1 {
		node [style=filled,color=white, shape=circle];
		a12 a22 a32;
		label="Hidden layer";
	}

	subgraph cluster_2 {
		node [style=filled,color=white, shape=circle];
		O1[label="f(x_1, x_2)"];
		label="Output logit";
	}

    x1 -> a12;
    x1 -> a22;
    x1 -> a32;

    x2 -> a12;
    x2 -> a22;
    x2 -> a32;

    a12 -> O1;
    a22 -> O1;
    a32 -> O1;
}
```

In order for the above feature selection to work, we will need to zero out the weights connected for all of feature {% latex %}$x_2${% endlatex %} (marked in red).

```
digraph G {
    graph [bgcolor=black, fontcolor=white];
    node [label=""];
    edge[arrowhead=vee, arrowtail=inv, arrowsize=.7, color=white, fontsize=12,
            fontcolor=white]
    rankdir=LR
    splines=line
    nodesep=.15;
        HelloWorld$
    subgraph cluster_0 {
        style="";
        node [style=filled,color=white, shape=circle];
		x1[label="x_1"] x2[label="x_2"];
		label="Input features";
	}

	subgraph cluster_1 {
        style="";
		node [style=filled,color=white, shape=circle];
		a12 a22 a32;
		label="Hidden layer";
	}

	subgraph cluster_2 {
        style="";
		node [style=filled,color=white, shape=circle];
		O1[label="f(x_1, x_2)"];
		label="Output logit";
	}

	subgraph edge_1 {
        x1 -> a12;
        x1 -> a22;
        x1 -> a32;
    }

	subgraph edge_3 {
        edge [color=red, penwidth=3];
        x2 -> a12;
        x2 -> a22;
        x2 -> a32;
    }

    a12 -> O1;
    a22 -> O1;
    a32 -> O1;
}
```

In this case, the weights associated with each of the neurons becomes becomes a group of their own. Let {% latex %}$w_1${% endlatex %} and {% latex %}$w_2${% endlatex %} denote the weight vectors for input features {% latex %}$x_1${% endlatex %} and {% latex %}$x_2${% endlatex %} ({% latex %}$w_2${% endlatex %} weights would be marked in red above). We can adapt the group LASSO regularization formulation as

{% latex class=center %}
\begin{align*}
\arg \min_{w} \Sigma_{i=1}^n L(y_i, f(x_1, x_2)) + \lambda \Sigma_{g=1}^{G=2} \sqrt{d_g} \lvert\lvert w_g \rvert\rvert_2
\end{align*}
{% endlatex %}

where {% latex %}$L${% endlatex %} denotes the loss function and {% latex %}$w_i${% endlatex %} denotes the full-connected weights to feature {% latex %}$f_i${% endlatex %}. Since we have two input features, the regularization term would also expand to

{% latex class=center %}
\begin{align*}
\lambda \cdot \sqrt{3} \cdot (\lvert\lvert w_1 \rvert\rvert_2 + \lvert\lvert w_2 \rvert\rvert_2)
\end{align*}
{% endlatex %}

We have essentially derived the Group level lasso regularization on each of the individual features, with the weights corresponding to each feature in a group. We can continue to build on the intuition from the Group LASSO.

While each individual weights inside a weight group will not differ in terms of convergence to zero (all elements of {%latex%}$w_1${%endlatex%}, {%latex%}$w_2${%endlatex%} will either be zero or non-zero), the non-continuous nature of the l2 norm for individual features will introduce sparsity and converge entire feature weights to 0.

From here, it's trivial to apply the same technique to regularizing hidden layers to introduce further sparsity to the model and improve model capacity or prune unneeded connections.

## References
* [Yang, Haiqin et al. “Online Learning for Group Lasso.” *ICML* (2010).](https://icml.cc/Conferences/2010/papers/473.pdf)
* [Scardapane, Simone et al. “Group Sparse Regularization for Deep Neural Networks.” Neurocomputing 241 (2017): 81–89. Crossref. Web.](https://arxiv.org/abs/1607.00485)
