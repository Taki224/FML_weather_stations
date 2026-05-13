gemini --resume 'ef40776b-8807-477e-9626-77a256a0d992' 

# FML Project Proposal & Implementation Guide
## Title: Spatial Regularization: Overcoming Data Scarcity through Network Consensus

---

## 1. Introduction and Motivation

**Core Concept:** 
In real-world Federated Learning (FL) applications, individual clients (in our case, weather stations) often have limited, noisy, or incomplete data. If a single station tries to train a machine learning model using only its own small dataset, the model will likely overfit and perform poorly on unseen data. 

**The FL Solution:**
By linking stations together into a spatial graph, we can use the **Graph Total Variation Minimization (GTVMin)** framework. This mathematically forces neighboring stations (nodes connected by edges) to learn models that are "similar" to each other. We are injecting the physical prior knowledge that "weather is spatially continuous" into the machine learning algorithm.

**Why this project is excellent for learning:**
*   It directly demonstrates the core value of FL: collaboration improves individual performance, especially under data scarcity.
*   You don't need gigabytes of data. A small dataset (e.g., a few weeks) actually highlights the problem of local overfitting, making the FL solution look better by comparison.
*   It allows for a clear mathematical derivation of the fixed-point equation and a straightforward implementation using `scikit-learn`.

---

## 2. Problem Formulation (The FL Design)

This section maps the physical weather problem to the mathematical FL framework.

### 2.1. The Data (Features and Labels)
*   **Task:** Predicting Air Temperature based on other local meteorological factors.
*   **Features ($\mathbf{x}$):** We will use easily available parameters. For example: `[Wind speed, Pressure (msl), Relative humidity]`. (You can adjust this based on what your `download_script.py` pulls easily).
*   **Label ($y$):** Air temperature.
*   **Local Dataset:** At each station $i$, we have a small dataset of $m_i$ readings: $\mathcal{D}^{(i)} = \{(\mathbf{x}^{(i)}_1, y^{(i)}_1), \dots, (\mathbf{x}^{(i)}_{m_i}, y^{(i)}_{m_i})\}$.

### 2.2. The FL Network (Nodes, Edges, Weights)
*   **Nodes ($V$):** Each node $i \in V$ is an FMI weather station.
*   **Edges ($E$):** Edges are defined by the **Delaunay Triangulation** of the stations' geographic coordinates (longitude, latitude). This naturally connects adjacent stations without requiring arbitrary distance thresholds.
*   **Edge Weights ($A_{i,j}$):** (As implemented in your notebook). For an edge between station $i$ and $j$:
    *   $dist = \text{Haversine\_Distance}(i, j)$ in km.
    *   $A_{i,j} = \max(0, 60 - dist)$ 
    *   *Justification:* Stations closer to each other have a stronger meteorological correlation. The weight decreases linearly, hitting zero at 60km, ensuring that only local climatic neighborhoods influence each other.

### 2.3. The Local Models
*   **Hypothesis Class:** Linear Regression. 
*   **Model Parameters:** For each station $i$, the model is defined by a weight vector $\mathbf{w}^{(i)}$. 
*   **Prediction:** $\hat{y} = \mathbf{w}^{(i)T} \mathbf{x}$.
*   *Justification:* Linear models are robust, interpretable, and mathematically tractable for deriving the GTVMin fixed-point equation.

### 2.4. The Local Loss Function
*   **Loss ($L$):** Mean Squared Error (MSE).
*   $L(\mathbf{w}^{(i)}) = \frac{1}{m_i} \sum_{k=1}^{m_i} (\mathbf{w}^{(i)T} \mathbf{x}^{(i)}_k - y^{(i)}_k)^2$
*   *Justification:* Standard loss for regression tasks; penalizes large errors heavily, which is suitable for predicting continuous physical variables like temperature.

---

## 3. Methodology: GTVMin and The Fixed-Point Equation

This is the core mathematical heart of the project required by the rubric.

### 3.1. The Global GTVMin Objective
We want to find the optimal set of models for all stations, $\mathbf{w} = [\mathbf{w}^{(1)}, \mathbf{w}^{(2)}, \dots, \mathbf{w}^{(n)}]$, by minimizing the global objective function $J(\mathbf{w})$:

$$ J(\mathbf{w}) = \sum_{i=1}^{n} L(\mathbf{w}^{(i)}) + \alpha \sum_{(i,j) \in E} A_{i,j} \|\mathbf{w}^{(i)} - \mathbf{w}^{(j)}\|_2^2 $$

**What does this mean?**
*   The first term $\sum L(\mathbf{w}^{(i)})$ tries to make each model perfectly fit its local data.
*   The second term (the GTV penalty) penalizes models if their weights are different from their neighbors' weights.
*   $\alpha$ (alpha) is the **regularization parameter** controlling the trade-off.

### 3.2. Deriving the Fixed-Point Equation
To solve this distributed problem, we look at the optimization from the perspective of a single node $i$. Node $i$ wants to minimize its local cost function $J_i(\mathbf{w}^{(i)})$ while holding its neighbors' models constant.

The local cost for node $i$ is:
$$ J_i(\mathbf{w}^{(i)}) = L(\mathbf{w}^{(i)}) + \alpha \sum_{j \in \mathcal{N}(i)} A_{i,j} \|\mathbf{w}^{(i)} - \mathbf{w}^{(j)}\|_2^2 $$
Where $\mathcal{N}(i)$ is the set of neighbors of node $i$.

We can expand the penalty term:
$$ \|\mathbf{w}^{(i)} - \mathbf{w}^{(j)}\|_2^2 = \|\mathbf{w}^{(i)}\|_2^2 - 2\mathbf{w}^{(i)T}\mathbf{w}^{(j)} + \|\mathbf{w}^{(j)}\|_2^2 $$

Let the sum of weights from node $i$ to all its neighbors be the **degree** $d_i$:
$$ d_i = \sum_{j \in \mathcal{N}(i)} A_{i,j} $$

Substituting this back, the local cost becomes:
$$ J_i(\mathbf{w}^{(i)}) = L(\mathbf{w}^{(i)}) + \alpha \cdot d_i \|\mathbf{w}^{(i)}\|_2^2 - 2\alpha \sum_{j \in \mathcal{N}(i)} A_{i,j} \mathbf{w}^{(i)T}\mathbf{w}^{(j)} + \text{Constant} $$

Let's define a "message" received from neighbors as the weighted average of their models:
$$ \mathbf{\bar{w}}^{(i)} = \frac{1}{d_i} \sum_{j \in \mathcal{N}(i)} A_{i,j} \mathbf{w}^{(j)} $$

Now the local cost simplifies beautifully to:
$$ J_i(\mathbf{w}^{(i)}) = L(\mathbf{w}^{(i)}) + \alpha \cdot d_i \|\mathbf{w}^{(i)} - \mathbf{\bar{w}}^{(i)}\|_2^2 + \text{Constant} $$

**The Resulting Fixed-Point Operator $\mathcal{F}^{(i)}$:**
At iteration $t+1$, node $i$ updates its model by minimizing $J_i$:
$$ \mathbf{w}^{(i, t+1)} = \arg\min_{\mathbf{w}} \left( L(\mathbf{w}) + \alpha \cdot d_i \|\mathbf{w} - \mathbf{\bar{w}}^{(i, t)}\|_2^2 \right) $$

This is the block-wise minimization operator $\mathcal{F}^{(i)}$.

### 3.3. Implementation via `scikit-learn`
Notice the form of the equation above! It is exactly the cost function for **Ridge Regression** (L2 Regularized Linear Regression), but centered around $\mathbf{\bar{w}}^{(i)}$ instead of zero.

Standard Ridge Regression minimizes: $\text{Loss} + \lambda \|\mathbf{w}\|_2^2$
Our equation minimizes: $\text{Loss} + (\alpha \cdot d_i) \|\mathbf{w} - \mathbf{\bar{w}}\|_2^2$

To implement this using `sklearn.linear_model.Ridge`, we can use a mathematical trick: **Data Augmentation**.
Instead of writing a custom optimizer, we add "pseudo-data" to our local dataset to pull the model towards $\mathbf{\bar{w}}^{(i)}$.

**The Trick:**
1.  Original data: $X$ (features), $Y$ (labels).
2.  Pseudo-data features: $\sqrt{\alpha \cdot d_i} \cdot I$ (where $I$ is the identity matrix).
3.  Pseudo-data labels: $\sqrt{\alpha \cdot d_i} \cdot \mathbf{\bar{w}}^{(i)}$.
4.  Stack them together:
    $$ X_{aug} = \begin{bmatrix} X \\ \sqrt{\alpha \cdot d_i} \cdot I \end{bmatrix}, \quad Y_{aug} = \begin{bmatrix} Y \\ \sqrt{\alpha \cdot d_i} \cdot \mathbf{\bar{w}}^{(i)} \end{bmatrix} $$
5.  Run standard `sklearn.linear_model.LinearRegression().fit(X_aug, Y_aug)`!

---

## 4. Practical Implementation Steps (The Algorithm)

Here is the exact algorithmic loop you will implement in code.

**Initialization:**
1.  Download a small timeframe of data (e.g., 2 weeks in winter).
2.  Clean data, align timestamps, handle missing values (impute or drop).
3.  Standardize features (zero mean, unit variance) - *crucial for Ridge/GTVMin*.
4.  Split each station's data chronologically: 60% Train, 20% Val, 20% Test.
5.  Build the Delaunay Graph (you have this) and calculate $A_{i,j}$ and $d_i$.
6.  Initialize models: Train an independent Linear Regression model for each station using ONLY its local training data. Set $\mathbf{w}^{(i, 0)}$ to these weights.

**The FL Loop (Synchronous Message Passing):**
Repeat for $t = 1, 2, \dots, T_{max}$ (or until models stop changing):
1.  **Communicate:** Every node $i$ calculates $\mathbf{\bar{w}}^{(i, t-1)} = \frac{1}{d_i} \sum_{j} A_{i,j} \mathbf{w}^{(j, t-1)}$.
2.  **Update:** Every node $i$ augments its local *training* data $X_{train}^{(i)}, Y_{train}^{(i)}$ using $\alpha$, $d_i$, and $\mathbf{\bar{w}}^{(i, t-1)}$ as shown in section 3.3.
3.  **Fit:** Every node $i$ trains a new `LinearRegression` on the augmented data to get $\mathbf{w}^{(i, t)}$.
4.  **Evaluate:** Calculate the average validation loss across the network.

---

## 5. Experimental Design and Validation

To get maximum points, you must compare structurally different systems.

### 5.1. The Baselines
1.  **Baseline 1: Local-only Training ($\alpha = 0$)**
    *   Stations don't communicate. They train only on their local data.
    *   *Expectation:* High variance, overfits to small local datasets. Poor test performance.
2.  **Baseline 2: Global Shared Model ($\alpha \to \infty$)**
    *   All data from all stations is pooled together to train a single `LinearRegression` model.
    *   *Expectation:* Underfits. It ignores local microclimates (e.g., a coastal station vs. an inland station).
3.  **Your System: FL with Delaunay Graph (Tuned $\alpha$)**
    *   The sweet spot. Models are locally adapted but smoothed by neighbors.

### 5.2. Hyperparameter Tuning (Model Selection)
You must find the best $\alpha$.
1.  Define a grid of $\alpha$ values (e.g., `[0.1, 1, 10, 100, 1000]`).
2.  For each $\alpha$, run the FL Loop until convergence using the *Training* set.
3.  Evaluate the final models on the *Validation* set (calculate mean MSE across all stations).
4.  Select the $\alpha$ that gives the lowest Validation MSE.

### 5.3. Final Evaluation
Take the models trained with the best $\alpha$, and evaluate them *once* on the *Test* set. Report the Mean MSE across all stations. Compare this Test MSE against Baseline 1 and Baseline 2.

---

## 6. How to map this to the Grading Rubric

When writing your final PDF report, structure it exactly matching the rubric headers to make it easy for the instructor to grade:

*   **1.1 Title and Introduction:** "Spatial Regularization in Federated Learning: Predicting Temperature via Network Consensus". Briefly mention the problem of data scarcity and the FL approach.
*   **1.2 Problem Formulation:** Clearly list nodes (stations), edges (Delaunay), weights (60-dist), features, and labels as defined in Section 2.
*   **1.3 Methodology:** Replicate the math in Section 3. Clearly show the fixed-point equation and explicitly mention how `sklearn.linear_model.LinearRegression` is used with the data augmentation trick to implement $\mathcal{F}^{(i)}$.
*   **1.4 Numerical Experiments:** 
    *   Explain your chronological train/val/test split (e.g., first 10 days train, next 2 days val, last 2 days test).
    *   Show a table of Train, Val, and Test losses for the best model.
    *   **Crucial:** Create a bar chart comparing the Test MSE of Local-only, Global Model, and FL Model.
*   **1.5 Reproducibility:** Ensure you list the exact date range downloaded, the specific features chosen, the optimal $\alpha$ found, and random seeds (if any).

---
## Next Steps for You:
1.  Use `download_script.py` to get a small, dense chunk of data (e.g., 2 weeks in winter, timeseries format).
2.  Start a new notebook for the ML part.
3.  Merge the downloaded data with the graph structure you built in `visualise_graph.ipynb`.
4.  Implement the Data Augmentation trick for a single station to test it.
5.  Wrap it in a loop for all stations!
