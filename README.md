# OpinionDynamicsValidation

# Opinion Dynamics Validation – How to Run

This repository contains the code for running the simulated and ESS experiments from the validation and distortion papers.

---

# 1. Setup

## Install requirements

```bash
pip install -r requirements.txt
```

## Unzip ESS data (required for real experiments)

```bash
unzip datasets/ess/ess_datasets.zip -d datasets/ess
```

If this is not unzipped, `run_real.py` will fail.

The dataset keys used in the experiments are defined in `datasets/ess/header_info.py`.

---

# 2. Main Scripts

You will hopefully only need to use:

```
experiments/run_simulated.py
experiments/run_real.py
experiments/run_plotting.py

Note: experiments/euler_scripts has old scripts I used to use to run on euler but I can't test those now so not sure if they are working.
```

Other folders:

- `models/` → model implementations
- `utils/` → utilities (metrics, optimisation, plotting)
- `exploratory_experiments/` → testing code and old experiments (can likely ignore)
- `SSC2025/` → results from SSC (likely can ignore)
- `results/` → all outputs are saved here. this folder will be accessed when running `experiments/run_plotting.py`

---

# 3. Running Simulated Experiments

```bash
python -m experiments/run_simulated.py --true_model <model> --experiment <experiment>
```

You must specify `--true_model`.

Model list: `deffuant, deffuant_with_repulsion, hk_averaging, ed, duggins, gestefeld_lorenz`

Other useful flags:

```
--prediction_model <model> (will default to same as true model if not added)
--experiment [plot_true | reproducibility | noise | optimized]
--distort_true (generator model is distorted (beta values are randomly generated))
--distort_prediction (prediction model is distorted (beta values are also optimizable))

--seed <int>
--plot_datasets (plots datasets after each trial -- used for debugging)
```

This will:

- Generate synthetic datasets
- Optimise parameters
- Validate on held-out waves
- Save results to `results/`

---

# 4. Running Real ESS Experiments

```bash
python -m experiments/run_real.py -prediction_modeld <model>
```

You must specify `--prediction_model`.

Model list: `deffuant, deffuant_with_repulsion, hk_averaging, ed, duggins, gestefeld_lorenz`

Other useful flags:

```
--distort_prediction (prediction model is distorted (beta values are also optimizable))
```

This will:

- Load ESS data
- Run optimisation
- Evaluate prediction performance
- Save results to `results/`

Make sure `ess_datasets.zip` has been unzipped first.

---

# 5. Plotting Results

```bash
python -m experiments/run_plotting.py
```

The `produce_stripplot()` function should produce a stripplot if you have all model data.
The  `produce_figure(...)` function will produce more specific figures

This reads saved files from `results/` and generates the plots.

You can also directly call plotting functions from `utils`.

---

# Notes

- All results are saved in `results/`.
- Optimisation uses TPE (hyperopt).
- Models and parameter ranges are defined inside `models/`.
- Distortions (if enabled) are applied before model execution and inverted afterwards.
