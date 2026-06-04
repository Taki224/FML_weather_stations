# Theoretical Preparation Guide: Spatial Regularization in Federated Learning

This guide outlines the critical theoretical topics and mathematical derivations you must master to confidently present your Federated Learning (FL) solution and satisfy the grading rubric for the **FML2ILV - Federated Machine Learning** course.

---

## 1. Core Framework: Graph Total Variation Minimization (GTVMin)

### What is GTVMin?
**Graph Total Variation Minimization (GTVMin)** is a regularization framework used to train localized models over graph-structured data. Rather than training a single global model for all nodes (which ignores local variance) or completely independent local models (which suffer from data scarcity and noise), GTVMin blends both approaches using a spatial prior: **neighboring nodes in a network should have similar models.**

### The Global Objective Function
The global optimization problem is defined as:
$$ J(\mathbf{w}) = \sum_{i=1}^{n} L(\mathbf{w}^{(i)}) + \alpha \sum_{(i,j) \in \mathcal{E}} A_{i,j} \|\mathbf{w}^{(i)} - \mathbf{w}^{(j)}\|_2^2 $$

Where:
*   $n$: The number of nodes (weather stations).
*   $\mathbf{w}^{(i)} \in \mathbb{R}^d$: The model parameter (weight) vector for node $i$.
*   $L(\mathbf{w}^{(i)})$: The local loss function (e.g., Mean Squared Error) at node $i$.
*   $\mathcal{E}$: The set of edges connecting adjacent stations in the network topology.
*   $A_{i,j}$: The weight of the edge between nodes $i$ and $j$, representing their spatial proximity or similarity.
*   $\alpha \ge 0$: The global regularization parameter that scales the strength of the spatial smoothness constraint.

### Key Justifications to Present:
*   **Why spatial regularization?** Weather patterns (e.g., temperature) are physically continuous in space. Stations close to each other should share similar model parameters because they experience similar meteorological trends.
*   **Why local loss $L(\mathbf{w}^{(i)})$?** It ensures that each station's model remains faithful to its own microclimate and local data distribution.

---

## 2. Graph Topology & Construction Choices

To formulate the network, you map geographic coordinates to a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, A)$.

### Node and Edge Selection:
1.  **Nodes ($\mathcal{V}$):** Weather stations from the Finnish Meteorological Institute (FMI).
2.  **Edges ($\mathcal{E}$):** Constructed using **Delaunay Triangulation** of the stations' coordinates (longitude, latitude).
    *   *Justification:* Delaunay triangulation connects adjacent spatial regions naturally, avoiding the arbitrary thresholding of $k$-nearest neighbors or $\epsilon$-radius graphs. It ensures a planar graph structure where physical neighbors are connected without edges crossing.
3.  **Edge Weights ($A_{i,j}$):**
    $$ A_{i,j} = \max(0, 60 - \text{dist}_{\text{Haversine}}(i, j)) $$
    *   *Justification:* The weight represents meteorological correlation. The **Haversine formula** calculates the great-circle distance between two points on a sphere (the Earth). The weights decrease linearly as distance increases and hit exactly zero at 60 km, reflecting the physical limit of local climatic influence.

---

## 3. Mathematical Derivation of the Fixed-Point Equation

The rubric specifically requires you to derive the local fixed-point update step from the global objective $J(\mathbf{w})$. 

### Step 1: Isolate the perspective of a single node $i$
To minimize the global cost in a decentralized manner, node $i$ optimizes its own model weights $\mathbf{w}^{(i)}$ while keeping its neighbors' models $\mathbf{w}^{(j)}$ fixed. The localized cost function $J_i(\mathbf{w}^{(i)})$ is:
$$ J_i(\mathbf{w}^{(i)}) = L(\mathbf{w}^{(i)}) + \alpha \sum_{j \in \mathcal{N}(i)} A_{i,j} \|\mathbf{w}^{(i)} - \mathbf{w}^{(j)}\|_2^2 $$
where $\mathcal{N}(i)$ is the set of neighbors of node $i$.

### Step 2: Define node degree and neighbor consensus
Define the weighted node degree $d_i$ as the sum of adjacent edge weights:
$$ d_i = \sum_{j \in \mathcal{N}(i)} A_{i,j} $$

Define the aggregate neighbor "message" $\mathbf{\bar{w}}^{(i)}$ as the weighted average of the neighbor models:
$$ \mathbf{\bar{w}}^{(i)} = \frac{1}{d_i} \sum_{j \in \mathcal{N}(i)} A_{i,j} \mathbf{w}^{(j)} $$

### Step 3: Expand and rewrite the regularization term
Expand the quadratic penalty term:
$$ \sum_{j \in \mathcal{N}(i)} A_{i,j} \|\mathbf{w}^{(i)} - \mathbf{w}^{(j)}\|_2^2 = \sum_{j \in \mathcal{N}(i)} A_{i,j} \left( \|\mathbf{w}^{(i)}\|_2^2 - 2\mathbf{w}^{(i)T}\mathbf{w}^{(j)} + \|\mathbf{w}^{(j)}\|_2^2 \right) $$

Distribute the summation:
$$ = \left( \sum_{j \in \mathcal{N}(i)} A_{i,j} \right) \|\mathbf{w}^{(i)}\|_2^2 - 2 \mathbf{w}^{(i)T} \left( \sum_{j \in \mathcal{N}(i)} A_{i,j} \mathbf{w}^{(j)} \right) + \sum_{j \in \mathcal{N}(i)} A_{i,j} \|\mathbf{w}^{(j)}\|_2^2 $$

Substitute $d_i$ and $\mathbf{\bar{w}}^{(i)}$:
$$ = d_i \|\mathbf{w}^{(i)}\|_2^2 - 2 d_i \mathbf{w}^{(i)T} \mathbf{\bar{w}}^{(i)} + \sum_{j \in \mathcal{N}(i)} A_{i,j} \|\mathbf{w}^{(j)}\|_2^2 $$

Since $\mathbf{w}^{(j)}$ is held constant, the third term is a constant with respect to $\mathbf{w}^{(i)}$:
$$ = d_i \left( \|\mathbf{w}^{(i)}\|_2^2 - 2 \mathbf{w}^{(i)T} \mathbf{\bar{w}}^{(i)} + \|\mathbf{\bar{w}}^{(i)}\|_2^2 \right) - d_i \|\mathbf{\bar{w}}^{(i)}\|_2^2 + \text{Constant} $$
$$ = d_i \|\mathbf{w}^{(i)} - \mathbf{\bar{w}}^{(i)}\|_2^2 + \text{Constant} $$

### Step 4: Formulate the local cost and Fixed-Point operator
Substitute this back into $J_i(\mathbf{w}^{(i)})$:
$$ J_i(\mathbf{w}^{(i)}) = L(\mathbf{w}^{(i)}) + \alpha \cdot d_i \|\mathbf{w}^{(i)} - \mathbf{\bar{w}}^{(i)}\|_2^2 + \text{Constant} $$

Thus, the fixed-point operator $\mathcal{F}^{(i)}$ that yields the update for node $i$ at iteration $t+1$ is:
$$ \mathbf{w}^{(i, t+1)} = \mathcal{F}^{(i)}(\mathbf{w}^{(1,t)}, \dots, \mathbf{w}^{(n,t)}) = \arg\min_{\mathbf{w}} \left( L(\mathbf{w}) + \alpha \cdot d_i \|\mathbf{w} - \mathbf{\bar{w}}^{(i, t)}\|_2^2 \right) $$

---

## 4. Implementation Strategy: The Data Augmentation Trick

A major contribution of your project is implementing this custom optimization step via a standard Ridge Regression solver using a **Data Augmentation Trick**.

### The Mathematical Formulation
Standard Ridge Regression solves the optimization problem:
$$ \min_{\mathbf{w}} \|Y - X\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_2^2 $$
Our local objective solves:
$$ \min_{\mathbf{w}} \|Y - X\mathbf{w}\|_2^2 + (\alpha \cdot d_i) \|\mathbf{w} - \mathbf{\bar{w}}^{(i)}\|_2^2 $$

To use `sklearn.linear_model.LinearRegression` (or standard OLS), we construct augmented features $X_{\text{aug}}$ and labels $Y_{\text{aug}}$:
$$ X_{\text{aug}} = \begin{bmatrix} X \\ \sqrt{\alpha \cdot d_i} \cdot I \end{bmatrix}, \quad Y_{\text{aug}} = \begin{bmatrix} Y \\ \sqrt{\alpha \cdot d_i} \cdot \mathbf{\bar{w}}^{(i)} \end{bmatrix} $$

### Proof of Equivalence
Let's compute the loss on the augmented dataset:
$$ \|Y_{\text{aug}} - X_{\text{aug}}\mathbf{w}\|_2^2 = \left\| \begin{bmatrix} Y \\ \sqrt{\alpha \cdot d_i} \cdot \mathbf{\bar{w}}^{(i)} \end{bmatrix} - \begin{bmatrix} X \\ \sqrt{\alpha \cdot d_i} \cdot I \end{bmatrix}\mathbf{w} \right\|_2^2 $$
$$ = \left\| \begin{bmatrix} Y - X\mathbf{w} \\ \sqrt{\alpha \cdot d_i} (\mathbf{\bar{w}}^{(i)} - \mathbf{w}) \end{bmatrix} \right\|_2^2 $$
$$ = \|Y - X\mathbf{w}\|_2^2 + \left\| \sqrt{\alpha \cdot d_i} (\mathbf{w} - \mathbf{\bar{w}}^{(i)}) \right\|_2^2 $$
$$ = \|Y - X\mathbf{w}\|_2^2 + \alpha \cdot d_i \|\mathbf{w} - \mathbf{\bar{w}}^{(i)}\|_2^2 $$
This is mathematically identical to our local objective!

> [!WARNING]
> **Important MSE vs. SSE Correction:**
> If your code uses the Mean Squared Error (MSE) loss $L(\mathbf{w}) = \frac{1}{m_i} \|Y - X\mathbf{w}\|_2^2$ rather than the Sum of Squared Errors (SSE), the penalty term is scaled by the sample size $m_i$. To match this scale, your augmentation must be:
> $$ X_{\text{aug}} = \begin{bmatrix} X \\ \sqrt{m_i \cdot \alpha \cdot d_i} \cdot I \end{bmatrix}, \quad Y_{\text{aug}} = \begin{bmatrix} Y \\ \sqrt{m_i \cdot \alpha \cdot d_i} \cdot \mathbf{\bar{w}}^{(i)} \end{bmatrix} $$
> Keep this distinction in mind if asked about normalization in your presentation.

---

## 5. Experimental Design & Rubric Checklist

To maximize your score on the report and presentation, you must pay attention to specific grading rubric requirements:

1.  **Chronological Split for Time-Series:**
    *   Do not use random train/test splits. Explain that temporal data has high autocorrelation; using future data to predict the past creates data leakage. Split chronologically (e.g., 60% Train, 20% Val, 20% Test).
2.  **Compare *Two* Distinct FL Systems (Rubric Section 1.4, Item 13):**
    *   *Warning:* Simply tuning $\alpha$ is **not** considered comparing two FL systems.
    *   To get full points (2p), compare your Delaunay FL system against another structurally different FL configuration. For example:
        *   **FL on Delaunay Triangulation** vs. **FL on a fully connected graph with distance-based weights** or a **$k$-Nearest Neighbors ($k$-NN) graph**.
        *   **FL using Linear Regression** vs. **FL using Ridge Regression** (regularized locally toward zero) or **FL using a different local loss (MAE)**.
3.  **Baselines:**
    *   Compare against the two structural edge cases:
        1.  **Local-only Training ($\alpha = 0$):** High variance, overfits local sparse data.
        2.  **Global Shared Model ($\alpha \to \infty$):** Underfits, ignores local microclimates.

---

## 6. Key Presentation Q&A (Preparing to be "Grilled")

Here are questions an examiner or reviewer is likely to ask, along with suggested answers:

*   **Q: Why choose Delaunay Triangulation over a simple distance threshold graph?**
    *   *Answer:* A distance threshold graph can lead to disconnected components if stations are far apart, or highly dense/cluttered subgraphs in clusters. Delaunay triangulation guarantees that spatial adjacency is preserved in a planar, connected, and sparse graph structure where adjacent regions naturally share boundaries without crossing.
*   **Q: Is your iterative algorithm guaranteed to converge to the global minimum?**
    *   *Answer:* Yes. The global objective $J(\mathbf{w})$ is a sum of convex local loss functions (MSE) and convex quadratic penalty terms. Since the objective is strictly convex, standard block coordinate descent (alternating localized optimization) is guaranteed to converge to the unique global minimum.
*   **Q: How does local data heterogeneity (non-IID data) affect your method?**
    *   *Answer:* In standard FL, non-IID data causes local updates to drift away from the global model (client drift). In our case, spatial regularization explicitly embraces non-IID data by allowing models to differ across stations, while penalizing sudden spatial variations. This balances individual station characteristics with global geographic trends.
*   **Q: Why use the Haversine distance instead of Euclidean distance?**
    *   *Answer:* Geographic coordinates (latitude/longitude) represent positions on a sphere. Euclidean distance distorts physical distances as you move away from the equator. The Haversine formula correctly calculates the spherical great-circle distance, ensuring physical scaling remains accurate.
