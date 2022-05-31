# Nearest Neighbor Search
### Generalities
- **Importance:** search for the nearest neighbors of a data point in the data set is a fundamental problem across data science, and is useful for:
	1. Clustering ([[k-means]], mean-shift, etc.)
	2. Information retrieval in databases
	3. Information theory (vector quantization)
	4. [[Supervised learning]] ([[kNN-classifiers]])

- **Linear scan:** from an input of a dataset $P \subset \mathbb{R}^d$ and a query point $q \in \mathbb{R}^d$,
1. For each point $p_i$ in the data set,
	1. $d_{min} := \min\{d_{min}, \; d(q, p_i)\}$;
2. Return $d_{min}$ or the index $i_{min}$.

Complexity: $O(dn)$ in both time and space.

- **Strategy and challenges:**

![[knn_strats.png]]

- **Popular approaches:** linear scan; Voronoi diagrams; #tree like data structures (quadtrees or trees that generate *binary space partitions* such as dyadic trees, [[kd-trees]], Random Projection trees, PCA trees); Locality Sensitive Hashing.

### Usage of [[kd-trees]] for NN search

Two strategies are generally used:

- **Defeatist search:** finds an answer extremely quickly, but fails often, specially in high dimensions.

![[knn_defeatist.png]]

- **Backtracking search:** always succeeds, but may take up to linear time.

![[knn_backtrack.png]]

### High dimensions
- **Curse of dimensionality:** every data structure for NN-search has either exponential size or exponential query time in $d$ in the worst case.

This holds both in theory and in practice, and arises from *concentration of measure*.