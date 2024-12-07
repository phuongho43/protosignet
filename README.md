# protosignet

Designing network motifs via ODE modeling and NSGA-II.

## Getting Started

### 1. Install [UV](https://docs.astral.sh/uv/getting-started/installation/)

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Edit objective function and hyperparameters

protosignet/evolve_motifs.py

### 3. Run NSGA-II to evolve motifs

```bash
$ uv run evolve_motifs
```

### 4. Plot NSGA-II progress and simulate motif

```bash
$ uv run plot_results
```
