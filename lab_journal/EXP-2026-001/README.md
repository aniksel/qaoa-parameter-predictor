# Predicting QAOA parameters from a QUBO matrix. Max-Cut problem

- **Date:** 2026-06
- **Source:** `baseline_model.ipynb`
- **Pipeline code:** `PL.py` (QAOA circuit in PennyLane), `create_dataset.py` (dataset generation), `energy_count.py` (energy via Qiskit Aer)

## Model and method

<details>
<summary>Expand: model, loss, energy, training</summary>

**Model** (`create_model`) — a fully connected neural network `nn.Sequential`:

- `Flatten` → vector of 25 elements (5×5 matrix)
- `Linear(25 → 128)` → `ReLU`
- `Linear(128 → 64)` → `ReLU`
- `Linear(64 → 2p)`

**Input:** vectorized QUBO matrix. **Output:** `2p` numbers — `p` values of `beta` + `p` values of `gamma`.

**Loss:** normalized energy `(energy − Emin) / (Emax − Emin)`. `Emin` and `Emax` (the minimum and maximum QUBO energy over all `2^n` bitstrings) are **precomputed in advance** for each matrix when the dataset is loaded.

**Energy and gradient** are computed through the QAOA circuit in PennyLane (`PL.py`): the circuit returns the probabilities of all `2^n` bitstrings, and these are exactly what we use — the expected energy is computed from them, and we minimize it (the gradient also goes through PennyLane, backprop).

**Training:** `train_model`, Adam optimizer (`lr=0.001`), **50 epochs**. Train/test split = 80/20.

</details>

---

## Part 1 — Dataset: 100 graphs, n = 5 vertices, m = 6 edges

**Dataset** (`create_dataset.py`, `seed=43`): 100 random graphs, each with `n=5` vertices, `m=6` edges, no isolated vertices.

### Experiments

<details>
<summary>Experiment 1 — Effect of seed</summary>

How stable the result is for different random `seed` values of model initialization. Run over `seed = 1..30` with fixed `p=3`, on 100 matrices; for each seed — the final train and test loss.

![Train loss by seed (p=3)](figures/seed_effect_train_loss_p3.png)
![Train/test loss by seed (p=3)](figures/seed_effect_train_test_loss_p3.png)

Numbers are in `tables/seed_effect_p3.csv` (seed, train_loss, test_loss).

</details>

<details>
<summary>Experiment 2 — Effect of QAOA Depth p</summary>

How the circuit depth `p` affects quality. Final train/test loss for `p = 1..6`.

![Train loss by p](figures/depth_effect_train_loss.png)
![Train/test loss by p](figures/depth_effect_train_test_loss.png)

Numbers are in `tables/depth_effect.csv`.

| p | train loss | test loss |
|--:|-----------:|----------:|
| 1 | 0.3887 | 0.4075 |
| 2 | 0.1865 | 0.2042 |
| 3 | 0.1173 | 0.1233 |
| 4 | 0.0873 | 0.0937 |
| 5 | 0.0591 | 0.0651 |
| 6 | 0.0542 | 0.0681 |

</details>

<details>
<summary>Experiment 3 — Loss vs graph structure</summary>

The relation between loss and structural properties of the graph.

![Loss vs number of triangles](figures/loss_vs_triangles.png)
![Loss vs optimal Max-Cut size](figures/loss_vs_maxcut_size.png)
![Loss vs number of optimal cuts](figures/loss_vs_num_optimal_cuts.png)

**Summary by loss group** (`tables/loss_group_summary.csv`):

| Group | Count | Mean loss | Mean triangles | Mean max degree | Mean Max-Cut size | Mean number of optimal cuts |
|--------|-------:|-------------:|-----------------:|---------------:|-------------------:|------------------------:|
| group 1: low loss | 9 | 0.0329 | 2.00 | 4.00 | 4.0 | 18.0 |
| group 2: middle loss | 26 | 0.0830 | 1.00 | 3.00 | 5.0 | 4.0 |
| group 3: high loss | 45 | 0.1483 | 1.73 | 3.42 | 5.13 | 2.0 |

**Top 10 matrices by lowest loss** (full table — `tables/per_matrix_loss_and_structure.csv`, 80 rows):

| matrix | loss | degree seq | triangles | max-cut | opt. cuts |
|--------|-----:|-----------|----------:|--------:|----------:|
| matrix-33 | 0.0190 | (4,2,2,2,2) | 2 | 4 | 18 |
| matrix-14 | 0.0202 | (4,2,2,2,2) | 2 | 4 | 18 |
| matrix-92 | 0.0236 | (4,2,2,2,2) | 2 | 4 | 18 |
| matrix-28 | 0.0288 | (4,2,2,2,2) | 2 | 4 | 18 |
| matrix-21 | 0.0377 | (4,2,2,2,2) | 2 | 4 | 18 |
| matrix-99 | 0.0392 | (4,2,2,2,2) | 2 | 4 | 18 |
| matrix-67 | 0.0401 | (4,2,2,2,2) | 2 | 4 | 18 |
| matrix-41 | 0.0422 | (4,2,2,2,2) | 2 | 4 | 18 |
| matrix-32 | 0.0453 | (4,2,2,2,2) | 2 | 4 | 18 |
| matrix-91 | 0.0773 | (3,3,2,2,2) | 1 | 5 | 4 |

</details>

---

## Part 2 — Dataset: 205 graphs, n = 5 vertices, m = 6 edges

**Dataset** (`create_dataset.py`, `seed=43`): 205 possible graphs, each with `n=5` vertices, `m=6` edges, no isolated vertices.

**Seed:** the values `seed ∈ {9, 13, 17, 21, 28}` are **not random** — they are the best seeds from the results of previous runs (see Part 1, Experiment 1).

### Experiments

<details>
<summary>Experiment 4 — Energy_top10 per individual matrix </summary>

For each pair `(p, seed)` the model was trained and a curve was built over all matrices in the dataset. The Y axis here is the **mean energy over the 10 most probable bitstrings** (top-10).

![p=1, seed=9](figures/energy_curve_p1_seed9.png)
![p=1, seed=13](figures/energy_curve_p1_seed13.png)
![p=1, seed=17](figures/energy_curve_p1_seed17.png)
![p=1, seed=21](figures/energy_curve_p1_seed21.png)
![p=1, seed=28](figures/energy_curve_p1_seed28.png)
![p=2, seed=9](figures/energy_curve_p2_seed9.png)
![p=2, seed=13](figures/energy_curve_p2_seed13.png)
![p=2, seed=17](figures/energy_curve_p2_seed17.png)
![p=2, seed=21](figures/energy_curve_p2_seed21.png)
![p=2, seed=28](figures/energy_curve_p2_seed28.png)
![p=3, seed=9](figures/energy_curve_p3_seed9.png)
![p=3, seed=13](figures/energy_curve_p3_seed13.png)
![p=3, seed=17](figures/energy_curve_p3_seed17.png)
![p=3, seed=21](figures/energy_curve_p3_seed21.png)
![p=3, seed=28](figures/energy_curve_p3_seed28.png)
![p=4, seed=9](figures/energy_curve_p4_seed9.png)
![p=4, seed=13](figures/energy_curve_p4_seed13.png)
![p=4, seed=17](figures/energy_curve_p4_seed17.png)
![p=4, seed=21](figures/energy_curve_p4_seed21.png)
![p=4, seed=28](figures/energy_curve_p4_seed28.png)
![p=5, seed=9](figures/energy_curve_p5_seed9.png)
![p=5, seed=13](figures/energy_curve_p5_seed13.png)
![p=5, seed=17](figures/energy_curve_p5_seed17.png)
![p=5, seed=21](figures/energy_curve_p5_seed21.png)
![p=5, seed=28](figures/energy_curve_p5_seed28.png)

</details>

<details>
<summary>Experiment 5 — Concentration of beta / gamma values</summary>

Where the predicted `beta`/`gamma` values cluster. Heatmap for each `p`:

![heatmap p=1](figures/param_heatmap_p1.png)
![heatmap p=2](figures/param_heatmap_p2.png)
![heatmap p=3](figures/param_heatmap_p3.png)
![heatmap p=4](figures/param_heatmap_p4.png)
![heatmap p=5](figures/param_heatmap_p5.png)

Parameter distributions (`plot_all`), 25 figures:

![distribution 01](figures/param_distribution_01.png)
![distribution 02](figures/param_distribution_02.png)
![distribution 03](figures/param_distribution_03.png)
![distribution 04](figures/param_distribution_04.png)
![distribution 05](figures/param_distribution_05.png)
![distribution 06](figures/param_distribution_06.png)
![distribution 07](figures/param_distribution_07.png)
![distribution 08](figures/param_distribution_08.png)
![distribution 09](figures/param_distribution_09.png)
![distribution 10](figures/param_distribution_10.png)
![distribution 11](figures/param_distribution_11.png)
![distribution 12](figures/param_distribution_12.png)
![distribution 13](figures/param_distribution_13.png)
![distribution 14](figures/param_distribution_14.png)
![distribution 15](figures/param_distribution_15.png)
![distribution 16](figures/param_distribution_16.png)
![distribution 17](figures/param_distribution_17.png)
![distribution 18](figures/param_distribution_18.png)
![distribution 19](figures/param_distribution_19.png)
![distribution 20](figures/param_distribution_20.png)
![distribution 21](figures/param_distribution_21.png)
![distribution 22](figures/param_distribution_22.png)
![distribution 23](figures/param_distribution_23.png)
![distribution 24](figures/param_distribution_24.png)
![distribution 25](figures/param_distribution_25.png)

</details>
