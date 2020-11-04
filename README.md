<font size=5>**Fuzzy c-Means Algorithm**</font>

To describe a method to determine the fuzzy $c$-partition matrix $U$ for grouping a collection of $n$ data sets into $c$ classes, we define an objective function $J_{m}$ for a fuzzy $c$-partition:

> $J_{m}(U,v) = \sum_{k=1}^{n}\sum_{i=1}^{c}(\mu_{ik})^{m'}(d_{ik})^2$

where

> $d_{ik} = d(x_{k} - v_{i}) = [\sum_{j=1}^{m}(x_{kj}-v_{ij})^2]^\frac{1}{2}$

and where $\mu_{ik}$ is the membership of the $k$th data point in the $i$th class. $j$ is a variable on the feature space, that is, $j$ = 1, 2, . . .,$m$. The distance measure, $d_{ik}$ is a Euclidean distance between the $i$th cluster center and the $k$th data set. A new parameter is introduced called a **weighting parameter**, $m$ (**Bezdek, 1981**). This value has a range $m = [1,\inf)$ This parameter controls the amount of fuzziness in the classification process. Also, $v_{i}$ is the $i$th cluster center, which is described by $m$ features and can be arranged in vector form as before, $v_{i} = \{v_{i1}, v_{i2}, . . . , v_{im}\}$.

Each of the cluster coordinates for each class can be calculated in a manner similar to the calculation in the crisp case:

> $v_{ij} = \left(\frac{\sum_{k=1}^{n}\mu_{ik}^{m'}\cdot x_{ki}}{\sum_{k=1}^{n}\mu_{ik}^{m'}}\right)$

An effective algorithm for fuzzy classification, called iterative optimization, was proposed by **Bezdek (1981)**. The steps in this algorithm are as follows:

1. Fix $c$ $(2 ≤ c < n)$ and select a value for parameter $m$. Initialize the partition matrix, $U^{(0)}$. Each step in this algorithm will be labeled $r$, where $r$ = 0, 1, 2, . . .
2. Calculate the $c$ centers $\{v^{(r)}_{i}\}$ for each step.
3. Update the partition matrix for the rth step, $U^{(r)}$, as follows:

> $\mu_{ik}^{(r+1)}=\left[\sum_{j=1}^{c}\left(\frac{d_{ik}^{(r)}}{d_{jk}^{(r)}}\right)^{2/(m'-1)}\right]^{-1}$ , for $I_{k} = \emptyset$ , where $\sum_{i\in I_{k}}\mu_{ik}^{(r+1)}=1$

4. If $\Vert U^{(r+1)}-U^{(r)}\Vert \leq \epsilon_{L}$, stop; otherwise set $r = r + 1$ and return to step 2.
