# Quantum Annealing for Minimum Bisection Problem: A Machine Learning-based Approach for Penalty Parameter Tuning
[[PAPER]]() 

The project QaMBP is the implementation of two experiments: <br>
I. QUBO penalty parameter tuning using GBR, <br>
II. Testing the solution quality of proposed QUBO formulation using D-Wave Systems' quatum annealing solvers.

## Abstract
Minimum Bisection Problem is a fundamental NP-hard problem in combinatorial optimization with applications in parallel computing, network design, and machine learning. This paper explores the feasibility and performance of using D-Wave Systems’ quantum annealing technology to solve the Minimum Bisection Problem, formulated as a Quadratic Unconstrained Binary Optimization model. A central challenge in such formulations is the selection of the penalty parameter, which significantly influences solution quality and constraint satisfaction.
To address this challenge, we propose a novel machine learning-based approach for adaptive penalty parameter tuning using Gradient Boosting Regressor model. The method predicts penalty values based on graph properties such as the number of nodes, and edge density, allowing for dynamic adjustment tailored to each instance of the problem. This enables hybrid quantum-classical solvers to more effectively balance the goal of minimizing cut size with maintaining equal partition sizes.
We evaluate the proposed method on a large dataset of randomly generated Erdős–Rényi graphs with up to 4000 nodes and compare the results with classical partitioning algorithms, Metis and Kernighan–Lin. Experimental findings demonstrate that our adaptive tuning strategy significantly improves the performance of hybrid quantum-classical solvers and consistently outperforms used classical methods, indicating its potential as an alternative for large-scale graph partitioning problem.

## Setup
(recommended: Visual Studio Code)
```
git clone https://github.com/rusnakrenata/qambp.git
cd src
python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
# python run main.py
```


## 1. Penalty Parameter Calibration and QA Testing

<img src="qambp_images/process_diagram.png"/><br>

Fig. 1. Proposed approach for penalty parameter tuning. Given graph *G(n,p)* prepare QUBO matrix <em><b>Q<sub>G,λ</sub></b></em> for MBP and given penalty parameter λ. Test the formulation using D-Wave QA HS, Metis and Kerninghan-lin algorithms, and store data in MariaDB. Analyse the results (solution feasibility and accuracy) and recalculate λ if necessary.

### 1.1. MBP QUBO Formulation and Matrix 

MBP QUBO Formulation: <br>
<br>
![MBP energy equation](https://latex.codecogs.com/png.image?\dpi{110}E_{\text{MBP}}(\mathbf{x})%20=%20\sum_{(i,j)%20\in%20E}%20(x_i%20+%20x_j%20-%202x_ix_j)%20+%20\lambda%20\left(%20\sum_{i%20\in%20V}%20x_i%20-%20\frac{n}{2}%20\right)^2)

Penalty parameter estimation: <br>
<br>
![lambda equation](https://latex.codecogs.com/png.image?\dpi{110}\lambda%20=%20\frac{1%20+%20\min(\max(\deg(G)),%20\frac{n}{2}%20-%201)}{2})

QUBO Matrix:
```
1. Initialize Q_{G,λ} as an empty matrix

2. Update Q_{G,λ} with coefficients from E_cut(x):
   For each edge (i, j) in E:
       Q[i, i] ← Q[i, i] + 1
       Q[j, j] ← Q[j, j] + 1
       Q[i, j] ← Q[i, j] - 2

3. Update Q_{G,λ} with coefficients from E_balance(x):
   For each node i in V:
       Q[i, i] ← Q[i, i] + λ(1 - n)

   For each pair of nodes (i, j), where i ≠ j:
       Q[i, j] ← Q[i, j] + 2λ

```

[QUBO Matrix Code](src/qubo_matrix.py)

### 1.2. QA Testing
[QA Testing Code](src/qa_testing.py)

### 1.3. Benchmark algorithms
[PyMetis Code](src/pymetis_testing.py)
[Kerninghan-lin Code](src/kerninhan_lin_testing.py)


## 2. QUBO penalty parameter tuning using GBR

### 2.1. Training the GBR model
```
1. Extract collected graph properties and partitioning results (λ_min and λ_max) from the database.

2. Compute features: 
   - n (number of nodes)
   - ρ (graph density)
   - λ_est (estimated penalty parameter)

3. Train two Gradient Boosting Regressors (gbr_min, gbr_max) to predict λ_min and λ_max.

4. Evaluate models using:
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² score

5. Save trained models for future predictions.
```

### 2.2. Predicting λ for new graphs
```
1. Compute λ_est = (1 + min(n/2 - 1, max(deg(G)))) / 2 for new graph G(n, p).

2. Construct feature vector:
   f = (n, ρ, λ_est)

3. Predict λ_min and λ_max using trained regressors (gbr_min, gbr_max).

4. Compute final λ:
   λ = λ_est * (λ_min + λ_max) / 2
```

[GBR Training Code](src/gbr_training.py)

## 3. Graph generation
The MBP problem was tested on graphs created using the Erdős–Rényi *G(n, p)* model, where *n* represents the number of nodes, and *p* is the probability of an edge existing between any two nodes, which controls the overall graph density.

Tested values:
- *n* ∈ {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000}
- *p* ∈ {0.1, 0.25, 0.5, 0.75}

[Graph Generation Code](src/graph_generation.py)






