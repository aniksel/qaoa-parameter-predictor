# Predicting QAOA parameters from QUBO matrices (Max-Cut)

## Results

**See the experiment journal → [lab_journal/EXP-2026-001/README.md](lab_journal/EXP-2026-001/README.md)**

It contains all experiments with figures and tables:
effect of `seed`, effect of QAOA depth `p`, loss vs graph structure,
energy per individual matrix, and concentration of `beta`/`gamma` values.


## Project structure

```
qaoa-parameter-predictor/
├── baseline_model.ipynb     # main pipeline: data → model → training → analysis
├── create_dataset.py        # generate the Max-Cut QUBO dataset (random graphs)
├── PL.py                    # QAOA circuit in PennyLane (energy + gradient)
├── energy_count.py          # energy via Qiskit Aer
├── param_heatmap.py         # beta/gamma concentration heatmap
└── lab_journal/             # research journal
    └── EXP-2026-001/        # README.md + figures/ + tables/
```


## How to run

1. Install the dependencies (PyTorch, PennyLane, Qiskit, pandas, matplotlib).
2. Generate the dataset: `python create_dataset.py`.
3. Open `baseline_model.ipynb` and run the cells.
