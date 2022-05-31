# k-means
### Generalities
- One of many #clustering algorithms. From a point cloud with coordinates and a distance or dissimilarity matrix, we wish to partition the set $\newcommand{\reals}{\mathbb{R}} P \subset \mathbb{R}^d$ of data points into homogeneous subsets or **clusters**.

- With k-means, the parameter we wish to minimize when clustering is **total intra-cluster variance**: if $k$ is the number of sought clusters, $\sigma : P \rightarrow \{1, ..., k\}$ is the partition function and $c_1, ..., c_k$ are the centers of each cluster, then we want to minimize
$$\min_{c_1, ..., c_k, \sigma} \frac{1}{n} \sum_{p \ \in \ P} \lVert p - c_{\sigma(p)} \rVert_2^2.$$

