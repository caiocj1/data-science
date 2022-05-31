# kd-trees
#tree

### Description
![[knn_kdtree.png]]

### Recursive Construction
![[kdtrees_build.png]]

We observe that the complexity depends heavily on how the median is computed.

- **Median computation:** either by sorting the points of current cloud ($O(n \log^2 n)$), pre-sorting all points along each coordinate ($O(dn \log n)$) or by the linear median method, randomized or deterministic ($O(n \log n)$).
