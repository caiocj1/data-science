### Rosenblatt's Perceptron Algorithm
- [[Supervised Learning]] method designed for binary #classification $\newcommand{\resps}{\mathcal{Y}} \resps = \{-1, 1\}$.
- Models directly the separating hyperplane:
$$\left[1 \quad x^T\right]\begin{bmatrix} -\beta_0 \\ \beta \end{bmatrix} = 0$$
The classifier is then $\newcommand{\sign}{\mathrm{sign}} \hat{y}(x) = \sign \left(\left[1 \quad x^T\right]\begin{bmatrix} -\hat{\beta_0} \\ \hat{\beta} \end{bmatrix}\right)$.

- We fit the model by minimizing the distances of misclassified points to decision boundary:
$$ \hat{\beta}, \,\hat{\beta_0} = \mathrm{argmin}_{\beta, \beta_0} \sum_{x_i\text{ misclassified}} -y_i \left[1 \quad x_i^T\right] \begin{bmatrix} -\beta_0 \\ \beta \end{bmatrix}$$
The vector product on the right is  the distance of $x_i$ to the hyperplane multiplied by the norm of $\beta$.

- We then achieve the $\mathrm{argmin}$ by #gradient-descent (piecewise linear function optimization):

1. Init: set $\begin{bmatrix} \hat{\beta_0} \\ \hat{\beta} \end{bmatrix}$ at random;
2. Repeat:
	1. Compute misclassified set $M$;
	2. $\begin{bmatrix} \hat{\beta_0} \\ \hat{\beta} \end{bmatrix} \leftarrow \begin{bmatrix} \hat{\beta_0} \\ \hat{\beta} \end{bmatrix} - \rho \sum_{x_i \in M} \begin{bmatrix} -y_i \\ y_i x_i \end{bmatrix}$;
Until convergence (requires convergence threshold $\epsilon$).

An equivalent algorithm would be:

1. Init: set $\begin{bmatrix} \hat{\beta_0} \\ \hat{\beta} \end{bmatrix}$ at random;
2. Repeat:
	For each $i=1,...,n$ do:
		If $y_i(x_i^T \hat{\beta} - \hat{\beta_0}) < 0$, then $\begin{bmatrix} \hat{\beta_0} \\ \hat{\beta} \end{bmatrix} \leftarrow \begin{bmatrix} \hat{\beta_0} \\ \hat{\beta} \end{bmatrix} - \rho \begin{bmatrix} -y_i \\ y_i x_i \end{bmatrix}$;
Until convergence (requires in principle convergence threshold $\epsilon$).

- **Thm. (Rosenblatt, Novikoff):** If two classes are *linearly separable*, then *stochastic gradient descent* with $\rho = 1$ makes the energy converge to 0 in finitely many steps ($O(R^2/\gamma^2)$ steps).

- Analysis:
	- Advantages:
		1. Linear structure of energy permits the use of **stochastic gradient descent**.
		2. Stochastic gradient descent **scales up well** and **allows for re-training**.
	- Drawbacks:
		1. No unique solution (depends on initialization).
		2. Small margins lead to long convergence times.
		3. Behaves badly on non-separable classes: converges to irrelevant configurations or cyclic behavior with potentially long cycles.

---

### Connectionist viewpoint on perceptron

![[img/perceptron_neuron.png]]

- In practice, **smooth** or **piecewise smooth** activation functions are used:
$$\begin{align} & \text{identity: }t\mapsto t \text{ (neuron implements linear regression)} \\
& \text{ReLU: }t\mapsto \max\{0, t\} \text{ (simpler gradient expressions, faster training)} \\
& \text{Gaussian: }t\mapsto \exp(-t^2) \text{ (neuron implements linear regression with rbf)} \\
& \text{sigmoid: }t\mapsto \frac{1}{1+\exp(-t)} \text{ (neuron implements logistic regression)} \\
& \text{softmax: } (x_1, ..., x_d) \mapsto \frac{\exp(x_i)}{\sum_j \exp(x_j)} \text{ (outputs in } [0,1] \text{ which sum up to 1)}
\end{align}$$
- **Multi-layer perceptron (MLP):**

![[mlp.png]]

Example of "vanilla" MLPs:
1. In k-class #classification, hidden layers use sigmoids while the output layer of k neurons uses softmax.
2. In #regression, hidden layers use sigmoids while output layer has 1 neuron with identity.

- **Approximation power**

![[approx_power.png]]

- **Training:**

*Simple case:*

To better understand the training process, it is useful to visualize what happens in the 1-hidden layer, 1 output neuron case (here, $\newcommand{\reals}{\mathbb{R}} x_i \in \reals^d$):

![[training1.png]]

The input $x$ is passed to the input layer of the network ($d$ neurons for $d$ dimensions) and is one of the $x_i$ in the training phase.

We choose the error as being the residual sum of squares (RSS), which we want to minimize.

Since we have the explicit expression of $y$ as a function of the input $x_i$, we may easily calculate the gradient of $R_i$ with respect to the network parameters, $\beta^i, \; \beta_0^i, \; \gamma$.

We observe that these gradients have a common structure:

![[training2.png]]

*Generalization:*

![[training3.png]]

We then apply "online learning":
1. Perform multiple training epochs;
2. Update (reduce) $\rho$ between epochs until convergence or early stop.

We guarantee a convergence to a local minimum under a proper reduction scheme for $\rho$.

Proper reduction scheme: we cannot update $\rho$ in any way we choose; one often used method is to choose $\rho_i$ such that $\sum_i \rho_i$ diverges but $\sum_i \rho_i^2$ converges (e.g. $\rho_i = 1/i$).

Advantages of online learning:
1. Scales up well
2. Can handle new training data

- **Regularization:** the high number of parameters in neural networks usually leads to overfitting.

Approaches to solving this:
1. Early stop - stop gradient descent after $k$ epochs. Select $k$ by cross-validation.
2. Weight decay - penalize $l^2$-norm of the parameter vector: $\mathrm{RSS} + \lambda \sum_{\text{neuron }j} (\beta_0^j)^2 + \lVert \beta^j \rVert_2^2$. Adds a term $2\lambda \beta^j$ to the gradient $\nabla_{\beta^j} R_i$. $\lambda$ selected by cross-validation.
3. Weight elimination - same, but with $\mathrm{RSS} + \lambda \sum_{\text{neuron }j} \frac{(\beta_0^j)^2 + \lVert \beta^j \rVert_2^2}{1 + (\beta_0^j)^2 + \lVert \beta^j \rVert_2^2}$. Shrinks smaller weights more drastically.