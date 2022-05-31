- **Basic description:**

**Input**: $n$ observations and responses $(x_1, y_1),...,(x_n,y_n) \in \mathcal{X} \times \mathcal{Y}$.
**Goal:** build a predictor $f: \mathcal{X} \rightarrow \mathcal{Y}$ from the observations and responses whose mean *prediction error* on new query observations is minimal.

*Supervised* because we have the correct labels of the training set to use when building predictor.

- **Statistical framework:**

*Hypothesis:*
1. $x_i \sim X$ iid with values in $\mathcal{X} = \mathbb{R}^d$.
2. $y_i \sim Y$ iid with values in $\mathcal{Y} = \{1,...,\kappa\}$ ( #classification ) or $\mathcal{Y} = \mathbb{R}$ ( #regression ).

The **joint distribution** of $X, Y$ encodes the complexity of the problem: $X$ and $Y$ are perfectly dependent if and only if there is a perfect predictor.

**Prediction error** is measured by a *loss function* $L: \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}$.

Goal: minimize *risk* (expected prediction error): $\mathbb{E}_{(X, Y)} L(Y, f(X)).$

In practice: minimize *empirical risk*: $\frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i)).$

