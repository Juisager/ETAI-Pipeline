# Baseline Predictive Pipeline -- ETAI

```
Name: Joana Geraldes
Student Number: 20231700
```

Link to Professor's Repo: https://github.com/sofiacper/ETAI-Pipeline/tree/main

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Data dictionary

One row = one defendant, with their demographics and criminal history at the time of screening, COMPAS's own risk score for them, and whether they were actually rearrested within two years.

| column | type | description | notable values |
|--------|------|--------------|------------------|
| `id` | identifier | internal record id | not a model feature |
| `sex` | categorical | defendant's sex | `Male`, `Female` |
| `age` | numeric | defendant's age (years) at screening | |
| `age_cat` | categorical | age bucket | `Less than 25`, `25 - 45`, `Greater than 45` |
| `race` | categorical | defendant's race, as recorded | excluded from model features- kept aside only to audit fairness afterward |
| `juv_fel_count` | numeric | number of prior juvenile felony offenses | |
| `juv_misd_count` | numeric | number of prior juvenile misdemeanor offenses | |
| `juv_other_count` | numeric | number of other prior juvenile offenses | |
| `juvenile_total` | numeric | total juvenile offenses | |
| `priors_count` | numeric | number of prior adult offenses | |
| `prior_offenses` | numeric | number of prior offenses | |
| `age_in_months` | numeric | age expressed in months |  |
| `c_charge_degree` | categorical | degree of the current charge | `F` (felony), `M` (misdemeanor) |
| `decile_score` | numeric | COMPAS's own risk score | 1 (lowest) to 10 (highest); excluded from model features, used only for comparison |
| `score_text` | categorical | COMPAS's own risk category | `Low`, `Medium`, `High`; excluded from model features |
| `two_year_recid` | binary | **our target** - was this person rearrested within two years? | `0` = no, `1` = yes |

## Project structure

```
.
├── main.py                  # entry point: run the whole pipeline
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── preprocessing.py      # row-preserving cleaning (incl. domain-rule checks), training-only de-duplication, deployable preprocessing pipeline, and the split that locks the final test set away (week 4)
│   ├── model.py               # model construction (incl. the oblique random forest and AdaBoost builders, week 6; XGBoost, week 8) + VoteSelector, the feature-selection step (week 8) + build_pipeline(): preprocessing + selection + model as one estimator (week 5)
│   ├── evaluate.py           # stratified k-fold cross-validation, out-of-fold report + fairness check (week 4), corrected paired test to compare two models (week 6), several metrics and out-of-fold probabilities (week 7)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   └── results.py            # saves each run's report to disk, and its scores to the runs table (week 7)
├── results/                  # created automatically -- one report per run + runs.csv, the table of all runs (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
└── notebooks
    ├── Notebooks.ipynb      # Notebooks from the classes
    └── Results/             # Created at the end of the notebook to store the week's results
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. Threshold-independent metrics (ROC-AUC/PR-AUC) and a calibration check are deliberately **not** added yet -- not yet |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see "Hyperparameter tuning" below |
| 6 | Ensembles in depth: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`: `oblique_random_forest` (a forest of oblique trees built from scikit-learn parts -- every tree splits on its own random sparse linear combinations of the features, via `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth; stumps by default), both registered in `_MODELS`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size (`max_depth`, `min_samples_leaf`) and decorrelation (`max_features` / number of combinations) but not the number of trees, AdaBoost tunes its base trees' depth, its number of rounds and its learning rate; `corrected_paired_test()` in `src/evaluate.py` compares two models scored on the same folds (Nadeau & Bengio's corrected paired t-test), so a "winner" is only called when the difference is bigger than fold-to-fold noise; `scipy` listed in `requirements.txt` (it already came with scikit-learn, so nothing new to install) -- see "Ensembles" below |
| 7 | Gradient boosting + evaluating with probabilities | Three new model types in `src/model.py`: `gradient_boosting` (scikit-learn's `GradientBoostingClassifier`: each shallow tree fits the gradient of the log-loss left by the trees before it), `hist_gradient_boosting` (the same idea on binned features: much faster on large data, and it handles missing values itself) and `naive_bayes` (`GaussianNB`, a fast probabilistic baseline); gradient boosting is now the model `main.py` runs, each with its search space in `config.yaml`; evaluation now reports **several metrics** (accuracy, balanced accuracy, precision, recall, F1, F-beta, ROC-AUC, PR-AUC) and one **selection metric** (`cv.selection_metric`, ROC-AUC) that tuning maximises and runs are ranked by; out-of-fold **probabilities** (not just labels) feed the reports, with a confusion matrix; `class_weight: balanced` in the commented examples of the models that support it; new `save_scores()` in `src/results.py`: every run adds a row to `results/runs.csv` (the model and its settings, the score on each fold, the no-model floor and the untuned score on the same folds), so any two runs on the same data and folds can be compared with `corrected_paired_test()`; the printed output was reorganised to be easier to read, with the full tables in the results file (optional: keep your own format if you prefer) -- see "Model evaluation" below |
| 8 | XGBoost + feature selection | New model type `xgboost` (`XGBClassifier`, from the `xgboost` package: gradient boosting with extra regularisation, multi-threaded, handles missing values itself), now the model `main.py` runs, with its search space in `config.yaml`; **feature selection** as a learned step of the Pipeline: new `VoteSelector` class in `src/model.py` (a random forest, gradient boosting and an L1 logistic regression vote; features with at least `min_votes` votes are kept), placed between `"prep"` and `"model"` by `build_pipeline()` when the new `feature_selection` section of `config.yaml` switches it on, so it is re-fitted on the training rows of every fold (never on the rows it is scored on); with tuning on, `select__min_votes` is tuned with the model and assessed by nested CV; the pipeline caches fitted preprocessing and selection steps during a run (`Pipeline(memory=...)`, a temporary folder deleted at the end), so tuning doesn't refit them in every trial; the final model's kept features are printed, and the scoreboard records the selection setting -- see "Feature selection" below |
| 9 | Beyond COMPAS | The pipeline is no longer tied to COMPAS: main.py accepts a configuration-file path through --config (defaulting to config.yaml), allowing the original COMPAS setup and the new Adult Income setup to run through the same code; config_adult.yaml defines the Adult dataset, target, sensitive attribute, columns to exclude, diagnostics, preprocessing, model and output settings; COMPAS-specific feature engineering is now conditional on the required columns being present, so it is skipped automatically for Adult; the shared cleaning and feature-splitting functions are reused, keeping preprocessing consistent between training and prediction; the Adult workflow evaluates the selected model once on the previously locked 20% test set, then refits the pipeline on all labelled rows before predicting the 2,000 unlabelled records in adult_new.csv; This prediction stage is activated only when prediction_path is defined in the configuration, so COMPAS does not need a prediction file or any prediction-specific configuration.

## Preprocessing decisions

*(Written straight from the diagnosis in `Practical/W3/notebooks/01_eda_introduction.ipynb`; the preprocessing walkthrough is in `Practical/W4/notebooks/02_preprocessing.ipynb`. This is the summary.)*

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator needed |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator needed |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (excluded from model features anyway) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator needed |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values (out-of-range or negative) | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent category spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact-duplicate rows, all sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant with other columns (correlation r=1.00, or -- for `juvenile_total` -- an exact sum caught only by VIF) | multicollinearity | dropped |

**Encoder/scaler pair:** chosen by hand in `config.yaml` -- **target encoding** (compact, informative and **robust scaling** (median/IQR, so the few extreme counts don't set the scale). The alternatives (`onehot`/`ordinal`/`count`, `none`/`standard`/`minmax`) are one config change away.


## Hyperparameter tuning

*(Walkthrough: `Practical/W5/notebooks/04_hyperparameter_tuning.ipynb`.)* With `tuning.enabled: true` in `config.yaml`, `python main.py` also:

1. **Nested cross-validation** -- for each of the 5 outer CV folds, runs a full Optuna study on the outer-training rows only (each trial = 5-fold CV of the whole pipeline), refits the best candidate on those rows and scores it on the outer-validation fold. This is the honest estimate of the tuned model.
2. **Tuning on the whole development set** -- the same study, once, on every development row: these are the hyperparameters the final model uses.

The best trial's score is printed too, but it is **optimistic** (the maximum of many noisy CV scores) -- report the nested one. Search spaces live in `config.yaml` under `tuning.search_spaces`, one per model type, keyed by Pipeline parameter name (`model__max_depth`); a model without a search space can't be tuned until you add one. Cost: `(cv.n_splits + 1) x n_trials x tuning.n_splits` fits.

## Ensembles

Set `model.type` in `config.yaml` to choose the model; each type has a commented `params` example and its own search space under `tuning.search_spaces`.

| `model.type` | What it is | Search space (what the tuner chooses) |
|---|---|---|
| `random_forest` | many deep-ish trees on bootstrap samples, random features at every split, averaged | `max_depth`, `min_samples_leaf`, `max_features` |
| `oblique_random_forest` | the same, but each tree splits on random linear combinations of the features | the trees' `max_depth`, `min_samples_leaf`; `n_components` (combinations per tree) |
| `adaboost` | shallow trees built in sequence, each focused on the rows the previous ones got wrong; weighted vote | base-tree depth, `n_estimators`, `learning_rate` |

The forests' `n_estimators` is fixed in `params`, not tuned: more trees never overfit, they only cost time. Tuning an ensemble takes minutes per run (900 fits with the default settings) -- while testing, use a small `tuning.n_trials` or `tuning.enabled: false`.

## Model evaluation

**Metrics.** `cv.metrics` lists what every run reports; `cv.selection_metric` is the one that matters: tuning maximises it and the scoreboard ranks runs by it. The default is **ROC-AUC**: it scores how well the model *ranks* people by risk, without choosing a threshold, so COMPAS's own 1-10 risk score can be judged the same way. Accuracy, precision, recall and F1 depend on the threshold that turns a probability into a yes/no (0.5 in this pipeline). `fbeta_<beta>` is F-beta: `fbeta_2` counts recall twice as much as precision.

**The scoreboard.** Every `python main.py` adds one row to `results/runs.csv` (`save_scores()` in `src/results.py`): the run id, the data file, the model, whether it was tuned, the encoder and scaler, the model's params (and the tuned ones), the selection metric with its mean and std, the **untuned** score of the same model, the **no-model floor** (a dummy on the same folds) and the score on each fold (`fold_1` ... `fold_5`, from nested CV when tuning is on). The console prints the runs on the same data and metric, best first. The full report of each run is in `results/run_<run id>.txt`.

Two runs can only be compared if they used the **same data, cleaning and folds**: when you change the data file, the cleaning rules or `cv.random_state`, delete `results/runs.csv` and start a new table. (You could make this more automatic.) To compare two runs, from the project root:

```python
import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]
corrected_paired_test(runs.loc[0, folds], runs.loc[1, folds], test_train_ratio=1 / 4)   # rows 0 and 1; 5-fold: 1 / (5 - 1)
```

It returns the mean difference, the folds the first run wins, t and the p-value.

## Feature selection

Set `feature_selection.enabled` in `config.yaml` to add the `"select"` step between preprocessing and the model. `VoteSelector` fits three models on the training rows (a random forest, gradient boosting, and an L1-penalised logistic regression); each votes for the features it finds useful (above-average importance, or a non-zero coefficient), and features with at least `min_votes` votes are kept. Because it learns from the labels, it lives **inside** the Pipeline and is re-fitted in every fold; selecting features on all rows before cross-validation would make the score optimistic. With tuning on, `feature_selection.search_space` is added to the model's search space, so `min_votes` is chosen with the model's hyperparameters and assessed by nested CV. The features the final model keeps are printed at the end of each run.

**XGBoost** is a separate package: see "Installing XGBoost" below if `import xgboost` fails.

## Environment setup

You only need to do this once per machine.

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run.

**When `requirements.txt` changes** (a week adds a package -- e.g. week 5 adds `optuna`), reinstall once, with the venv active: `pip install -r requirements.txt`. Otherwise `python main.py` fails with `ModuleNotFoundError`. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Installing XGBoost

`xgboost` is in `requirements.txt`, so `pip install -r requirements.txt` (venv active) installs it. On its own: `pip install xgboost`. Check it worked, with the venv active:
```bash
python -c "import xgboost; print(xgboost.__version__)"
```

**Windows:** nothing extra: `pip install xgboost` installs a ready-made version. If it fails to build or install, update pip first (`python -m pip install --upgrade pip`) and make sure your Python is 64-bit (`python -c "import struct; print(struct.calcsize('P') * 8)"` should print 64).

**macOS:** XGBoost needs the **OpenMP** library (for running on several CPU cores), which macOS doesn't ship. If `import xgboost` fails with `XGBoostError`, `Library not loaded: libomp.dylib` or similar:
1. install [Homebrew](https://brew.sh) if you don't have it (the command is on its home page);
2. run `brew install libomp`;
3. open a **new** terminal, activate the venv, and try the import again.

Works on both Apple Silicon (M1/M2/M3/M4) and Intel Macs. If you use conda instead of a venv, `conda install -c conda-forge xgboost` installs XGBoost together with its OpenMP library.

**Linux:** on most distributions `pip install xgboost` just works. Two things can go wrong:
- it downloads a very large extra package (`nvidia-nccl-...`, a few hundred MB) for NVIDIA GPUs. If you have no NVIDIA GPU, or the download is too slow, install the CPU-only version instead: `pip install xgboost-cpu` (the code is identical: still `import xgboost`). If you do this, replace `xgboost` with `xgboost-cpu` in your copy of `requirements.txt`, so a reinstall doesn't fetch the large package again.
- a very old distribution may have no ready-made version and try to compile it. Update pip first (`python -m pip install --upgrade pip`); recent releases need glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, Fedora 29+).

**In a notebook (Jupyter / VS Code): `No module named 'xgboost'`, although you installed it.** The notebook runs on a different Python from the one `pip` installed into (e.g. you installed into the venv, but the notebook uses the system Python). Either:
- install from inside the notebook, into the Python it actually uses: run `%pip install xgboost` in a cell, then **restart the kernel**; or
- make your venv available as a kernel, once, with the venv active:
  ```bash
  pip install ipykernel
  python -m ipykernel install --user --name etai --display-name "ETAI (venv)"
  ```
  then choose **ETAI (venv)** as the notebook's kernel. The notebook then sees every package in `requirements.txt`.

To check which Python a notebook is using: `import sys; print(sys.executable)`. In the terminal: `which python` (macOS/Linux) or `where python` (Windows). They should point into the same `venv` folder.

## Environment Troubleshooting

Two Windows issues come up often enough to note here -- if you hit either, this saves you re-diagnosing it from scratch.

**PowerShell blocks the venv activation script, every new terminal.** The `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` line above only fixes it for that one terminal window -- close it and it's back. For a fix that actually sticks across sessions, run this **once**, instead:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
If it still doesn't stick (common on locked-down school/lab machines with a Group Policy that resets it on every logon), skip PowerShell entirely: use **Git Bash** (`source venv/Scripts/activate`) or **cmd.exe** (`venv\Scripts\activate.bat`) instead -- neither is affected by PowerShell's execution policy.

**Windows blocks the terminal/Python from reading or writing files in Documents (or Desktop/Pictures).** Shows up as an "Access is denied" error, or a silent failure to create/update a file, only when the project sits inside one of those folders. Two independent settings can cause this -- check both:
- **Windows Security -> Virus & threat protection -> Manage ransomware protection** -- turn off **Controlled folder access**, or add your terminal/Python/editor to its allowed-apps list.
- **Settings -> Privacy & security -> File system** -- make sure the terminal/Python has access.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, diagnoses and cleans the data (week 3), locks the final test set away, cross-validates preprocessing + model on the development set (week 4), tunes the model's hyperparameters if `tuning.enabled` is on (week 5 -- see "Hyperparameter tuning" above; minutes, not seconds, for the ensembles), and prints, in order:
- the model, the selection metric and the CV design; the size of the development set and of the locked test set (which is **not** used)
- **Default hyperparameters**: the cross-validation score on the selection metric, the train-validation gap (how much better the model looks on rows it trained on: a large gap means it memorised) and the no-model floor
- with tuning on, **Tuning**: the nested cross-validation score (the honest score of the tuned model) and the chosen hyperparameters
- **All metrics**: mean ± std of every metric
- **Scoreboard**: this run next to the earlier runs on the same data and metric, best first
- the final model (refit on all development rows), the features it keeps (with feature selection on) and where the full report was saved

The full detail -- per-fold tables for every metric, the tuning report, the out-of-fold confusion matrix and classification report, the false-positive-rate-by-race comparison with COMPAS's own score -- goes to the results file.

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Push to GitHub via Terminal

Standard workflow, from the project's root folder, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

**If `git push` asks for a password and rejects your normal GitHub password:** GitHub no longer accepts account passwords for git over HTTPS -- you need a **Personal Access Token (PAT)** instead.
1. On GitHub: **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. When `git push` prompts for a password, paste the token instead (username stays your GitHub username).
3. So you're not asked every time: `git config --global credential.helper manager` (Windows, usually already set up by Git for Windows) or `git config --global credential.helper store` (caches it in plaintext -- fine on a personal machine, not a shared one).

Alternative: set up an SSH key once (`ssh-keygen -t ed25519`, then add the public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL (`git@github.com:...`) instead of HTTPS -- no token to manage or renew.

## Dataset

See `data/README.md`.

## Results Analysis

For a detailed overview of the experiments, model comparisons, cross-validation results, and fairness evaluation, see [Results and Analysis](RESULTS.md).

This document tracks the project's progress across different weeks and discusses model performance, generalisation, and the main findings from each experiment.


## Beyond COMPAS: Adapting the Pipeline to Adult Income

### Running the Adult Configuration

The pipeline was extended to support a second dataset, Adult Income, without compromising its compatibility with COMPAS. Instead of maintaining separate implementations, `main.py` now accepts a configuration file through the `--config` argument. The default remains `config.yaml`, so the original COMPAS workflow is unchanged. The Adult pipeline can be executed with:

```bash
python main.py --config config_adult.yaml
```

The new `config_adult.yaml` specifies the Adult dataset, its target variable (`income_over_50k`), sensitive attribute (`sex`), columns to exclude, preprocessing settings, model configuration and output paths. Dataset-specific behaviour is controlled through the configuration rather than hard-coded into the main pipeline.

### Issues Found and Fixes

Adapting the pipeline revealed several assumptions that were specific to COMPAS or to the original dataset structure.

| Issue | Why it occurred | Solution |
|---|---|---|
| `main.py` only supported the COMPAS configuration | The original implementation loaded `config.yaml` directly, assuming it was the only configuration file used by the pipeline. This prevented the same code from being executed with dataset-specific settings for Adult Income. | Added command-line argument support using `argparse`, allowing users to specify a configuration file with `--config`. The default remains `config.yaml`, preserving compatibility with the original COMPAS workflow. |
| Dataset-specific feature engineering | The original pipeline included COMPAS-specific variables and transformations that are not present in Adult Income. | Made the feature engineering conditional on the required columns being available, allowing the Adult dataset to pass through without applying irrelevant transformations. |
| Missing values represented by `?` | Adult uses `?` as a missing-value marker in categorical columns, rather than standard `NaN` values. | Extended the generic cleaning procedure to recognise `?` as a placeholder and convert it into a proper missing value, allowing the configured imputation strategies and missingness indicators to work correctly. |
| Dataset-specific columns and target | The original pipeline assumed COMPAS column names and target-related information. | Moved dataset-specific settings into `config_adult.yaml`, including the target, sensitive attribute and columns to exclude from modelling. |
| Identifier and duplicate handling | Prediction records must retain their original identifiers, and every input row must receive a prediction, including duplicate records. | Kept the identifier separately for the output and reused the cleaning and feature-splitting procedures without applying training-only duplicate removal to prediction data. |
| Prediction data has no target | The unlabelled Adult prediction dataset cannot follow the same target-dependent evaluation workflow as the training data. | Added an optional prediction stage, activated only when a `prediction_path` is provided in the configuration. This keeps the COMPAS workflow independent of the Adult prediction file. |


### Model Comparison and Final Selection

The Adult dataset contains 6,000 labelled observations. A stratified 20% test set (1,200 observations) was reserved, leaving 4,800 observations for model development and cross-validation. Model selection was therefore performed using the development data, while the test set remained separate from the cross-validation and tuning procedures.

The following table summarises the main XGBoost baseline results obtained with feature selection disabled:

| Model configuration | Accuracy | Balanced accuracy | Precision | Recall | F1-score | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost, default parameters, no feature selection (test model) | 0.857 ± 0.017 | 0.767 ± 0.028 | 0.758 ± 0.038 | 0.593 ± 0.050 | 0.665 ± 0.045 | 0.906 ± 0.017 | 0.788 ± 0.030 |
| XGBoost, default parameters, feature selection enabled | **0.864 ± 0.011** | **0.781 ± 0.021** | **0.766 ± 0.035** | **0.622 ± 0.046** | **0.686 ± 0.030** | **0.913 ± 0.016** | **0.805 ± 0.027** |
| XGBoost, tuned parameters, feature selection enabled | 0.855 ± 0.011 | 0.769 ± 0.026 | 0.745 ± 0.035 | 0.603 ± 0.059 | 0.665 ± 0.036 | 0.908 ± 0.016 | 0.792 ± 0.030 |
| Logistic Regression, tuned parameters, feature selection enabled | 0.801 ± 0.018 | 0.810 ± 0.021 | 0.556 ± 0.026 | 0.829 ± 0.031 | 0.666 ± 0.027 | 0.895 ± 0.014 | 0.745 ± 0.026 |
| HistGradientBoosting, default parameters, feature selection enabled | 0.824 ± 0.014 | 0.816 ± 0.018 | 0.600 ± 0.023 | 0.801 ± 0.029 | 0.686 ± 0.024 | 0.905 ± 0.017 | 0.790 ± 0.026 |
| HistGradientBoosting, tuned parameters, feature selection enabled | 0.818 ± 0.017 | 0.819 ± 0.026 | 0.586 ± 0.028 | 0.822 ± 0.061 | 0.683 ± 0.029 | 0.905 ± 0.019 | 0.789 ± 0.031 |
| GradientBoosting, default parameters, feature selection enabled | 0.862 ± 0.011 | 0.774 ± 0.025 | 0.769 ± 0.029 | 0.606 ± 0.054 | 0.676 ± 0.036 | 0.912 ± 0.016 | 0.800 ± 0.031 |
| GradientBoosting, tuned parameters, feature selection enabled | 0.859 ± 0.015 | 0.778 ± 0.030 | 0.745 ± 0.038 | 0.623 ± 0.063 | 0.677 ± 0.043 | 0.907 ± 0.017 | 0.789 ± 0.034 |
| AdaBoost, default parameters, feature selection enabled | 0.850 ± 0.016 | 0.762 ± 0.033 | 0.731 ± 0.037 | 0.593 ± 0.069 | 0.653 ± 0.049 | 0.903 ± 0.016 | 0.776 ± 0.027 |
| AdaBoost, tuned parameters, feature selection enabled | 0.857 ± 0.010 | 0.766 ± 0.026 | 0.762 ± 0.020 | 0.589 ± 0.058 | 0.663 ± 0.037 | 0.905 ± 0.016 | 0.787 ± 0.025 |
| Oblique Random Forest, default parameters, feature selection enabled | 0.854 ± 0.014 | 0.764 ± 0.020 | 0.744 ± 0.040 | 0.593 ± 0.035 | 0.659 ± 0.033 | 0.896 ± 0.016 | 0.775 ± 0.027 |
| Oblique Random Forest, tuned parameters, feature selection enabled | 0.856 ± 0.010 | 0.759 ± 0.021 | 0.768 ± 0.029 | 0.574 ± 0.047 | 0.656 ± 0.031 | 0.900 ± 0.013 | 0.783 ± 0.024 |
| Random Forest, default parameters, feature selection enabled | 0.827 ± 0.011 | 0.794 ± 0.017 | 0.617 ± 0.019 | 0.730 ± 0.031 | 0.668 ± 0.022 | 0.891 ± 0.015 | 0.749 ± 0.026 |
| Random Forest, tuned parameters, feature selection enabled | 0.812 ± 0.013 | 0.815 ± 0.019 | 0.575 ± 0.019 | 0.821 ± 0.035 | 0.676 ± 0.022 | 0.902 ± 0.017 | 0.772 ± 0.034 |
| Decision Tree, tuned parameters, feature selection enabled | 0.775 ± 0.012 | 0.776 ± 0.025 | 0.520 ± 0.016 | 0.779 ± 0.069 | 0.623 ± 0.024 | 0.833 ± 0.036 | 0.610 ± 0.064 |
| Naive Bayes, tuned parameters, feature selection enabled | 0.818 ± 0.007 | 0.671 ± 0.013 | 0.723 ± 0.027 | 0.390 ± 0.027 | 0.506 ± 0.025 | 0.885 ± 0.014 | 0.714 ± 0.021 |

The final configuration selected was **XGBoost with feature selection enabled and hyperparameter tuning disabled**. This configuration was chosen based on the assignment's target-score requirements and the comparison between the evaluated alternatives. Another reason that model was chosen was that most of the models that achieved a good f1-score and acceptable ROC-AUC, didn't reach the requirement for the accuracy, which made them invalid to use, for example Hist Gradient Boosting.

Feature selection was implemented through `VoteSelector`, which combines the feature importance assessments of a Random Forest, Gradient Boosting and L1-regularised Logistic Regression. Features receiving at least the configured minimum number of votes are retained. Since selection is part of the pipeline, it is fitted independently within each cross-validation fold, avoiding information leakage.

**Features retained by the final model:**

numeric__age, numeric__education_num, numeric__capital_gain, numeric__capital_loss, numeric__hours_per_week, categorical__marital_status, categorical__occupation, categorical__relationship


The feature selection process retained only 8 of the original 38 features. Notably, none of the anonymised `v` columns were selected, suggesting that the other features provided more useful predictive information for the model.


### Predictions on Unlabelled Data

After evaluating the selected configuration, the pipeline was refitted using all 6,000 labelled Adult observations. It was then applied to `adult_new.csv`, which contains 2,000 unlabelled records.

The prediction workflow applies the same cleaning, feature engineering and feature-splitting procedures used during training. However, duplicate removal is deliberately skipped so that the number and order of prediction records remain unchanged. The resulting CSV contains the original `id`, the predicted probability of earning more than $50,000 and the corresponding binary prediction (0 or 1).

The resulting file is saved as `adult_predictions.csv` and contains exactly 2,000 predictions. This functionality is optional and is only executed when a prediction file is specified in the configuration; therefore, it does not introduce an additional requirement for COMPAS.

### Lessons Learned

The hardest assumption to identify was that missing values were represented consistently across datasets. In COMPAS, missing values were generally represented using standard missing-value markers, whereas Adult used `?` in several categorical columns. Without explicitly recognising this representation, the pipeline would treat missing values as a legitimate category, preventing the intended imputation and missingness handling. The most important change was therefore to make data cleaning generic and configurable, rather than relying on the original dataset's conventions. Reusing the same preprocessing and modelling pipeline, while keeping dataset-specific decisions in configuration files, made it possible to extend the project to Adult Income without duplicating the original COMPAS implementation.
