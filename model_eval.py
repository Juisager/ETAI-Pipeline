import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]
print(runs[["run", "model", "feature_selection"] + folds])

fs_run = runs.loc[runs["run"] == "20261105_164702"].iloc[0]
no_fs_run = runs.loc[runs["run"] == "20261105_172148"].iloc[0]

result = corrected_paired_test(
    fs_run[folds],
    no_fs_run[folds],
    test_train_ratio=1 / 4
)

print(result)