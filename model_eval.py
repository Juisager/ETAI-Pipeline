import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]
print(runs[["run", "model", "feature_selection", "imputer"] + folds])

median_run = runs.loc[runs["run"] == "20261119_101912"].iloc[0]
mean_run = runs.loc[runs["run"] == "20261119_101955"].iloc[0]
knn_run = runs.loc[runs["run"] == "20261119_102039"].iloc[0]
iterative_run = runs.loc[runs["run"] == "20261119_102124"].iloc[0]
rf_run = runs.loc[runs["run"] == "20261119_102207"].iloc[0]

comparisons = {
    "Mean vs Median": (mean_run, median_run),
    "KNN vs Median": (knn_run, median_run),
    "Iterative vs Median": (iterative_run, median_run),
    "Random Forest vs Median": (rf_run, median_run)
}

for name, (run_a, run_b) in comparisons.items():
    result = corrected_paired_test(
        run_a[folds],
        run_b[folds],
        test_train_ratio=1 / 4
    )

    print(f"\n{name}")
    print(result)