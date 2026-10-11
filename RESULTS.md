# Results and Analysis

This document presents **the experimental results and evaluation of the models developed for the ETAI Pipeline project**. It documents the progression of the experiments across different weeks, comparing model performance and evaluating generalisation using holdout testing and cross-validation.

The models are assessed using accuracy, classification metrics, and train–validation performance gaps. Fairness is also considered by comparing false positive rates across racial groups with those reported by COMPAS.

The goal is to **identify the most suitable model while considering not only predictive performance but also generalisation and potential fairness concerns**. The results and conclusions are presented chronologically to reflect the evolution of the pipeline and its evaluation methodology.

## Results

### Week 2 (16/09) - Decision Tree vs Logistic Regression

| Model | Train accuracy | Test accuracy | Train–test gap |
|---|---:|---:|---:|
| Logistic Regression | 0.680 | **0.678** | +0.002 |
| Decision Tree | 0.680 | 0.668 | +0.012 |

**Best model: Logistic Regression.** It achieved the highest test accuracy (0.678) and had the smallest train–test gap (+0.002), suggesting consistent performance between the training and test sets. Decision Tree achieved a test accuracy of 0.668, with a slightly larger gap of +0.012. However, the difference in test accuracy is only 0.010, so a single train/test split is not sufficient to conclude that Logistic Regression will consistently outperform Decision Tree.

###  Week 3 (23/09) - Data Cleaning

| Model | Train accuracy | Test accuracy | Train–test gap |
|---|---:|---:|---:|
| Logistic Regression | 0.676 | 0.657 | +0.019 |
| Decision Tree | 0.684 | **0.665** | +0.020 |

**Best model: Decision Tree.** It achieves higher test accuracy (66.5% vs. 65.7%) and performs slightly better at identifying recidivists, with a class 1 F1-score of 0.58 compared to 0.56 for Logistic Regression. Both models have a similar train–test gap, so the Decision Tree is the stronger choice based on these results.

### Week 4 (30/09) - Data Preprocessing

| Model | Holdout accuracy (W3) | CV accuracy (mean ± std) | CV train–val gap |
|---|---:|---:|---:|
| Logistic Regression | 0.657 | 0.672 ± 0.013 | +0.003 |
| Decision Tree | **0.665** | **0.675 ± 0.018** | +0.011 |
| Random Forest | — | 0.650 ± 0.018 | +0.083 |

I would trust the cross-validation accuracy more than a single holdout result because it evaluates the model across five different folds, providing a more reliable estimate of its generalisation performance. The Decision Tree has the highest mean validation accuracy (67.5%), slightly outperforming Logistic Regression (67.2%), while Random Forest performs worst and shows a much larger train–validation gap, suggesting overfitting. Therefore, the Week 2 conclusion that the Decision Tree is the best model is broadly supported by cross-validation in Week 3, although its advantage over Logistic Regression is small. The holdout accuracy for Week 3 is unavailable because the test set was deliberately kept separate and was not evaluated.

**Best model: Decision Tree.** It remains the best-performing model, achieving the highest holdout accuracy (66.5%) in Week 2 and the highest mean cross-validation accuracy (67.5%) in Week 3.

### Week 5 (07/10) - Hyperparameter Tuning


| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.695 ± 0.014 | 0.672 ± 0.017 | 0.679 | +0.002 | max_depth=6, min_samples_leaf=101, criterion=gini |
| Logistic regression | 0.672 ± 0.013 | 0.676 ± 0.014 | 0.677 | +0.003 | C=0.0746, solver=lbfgs, class_weight=None |
| Random forest | 0.647 ± 0.016 | 0.679 ± 0.016 | 0.682 | +0.004 | n_estimators=140, max_depth=25, min_samples_leaf=45, max_features=0.5 |

**Best Model**

Random Forest is the best-performing model after hyperparameter tuning, achieving a nested cross-validation accuracy of 0.679 ± 0.016. Tuning produced a substantial improvement compared with its default performance of 0.647 ± 0.016, corresponding to a gain of 0.032. The selected hyperparameters were 140 trees, a maximum depth of 25, a minimum of 45 samples per leaf, and max_features=0.5. 

**Questions**

*Which models gained from tuning, and is the gain larger than the fold-to-fold std?*

Logistic Regression and Random Forest benefited from tuning, while Decision Tree decreased in performance. The improvement for Logistic Regression was only 0.004, which is smaller than its fold-to-fold standard deviation (0.014). Random Forest improved by 0.032, which is larger than its standard deviation (0.016), indicating a more substantial improvement. Decision Tree decreased by 0.023, which is also larger than its standard deviation (0.017).


*Did tuning change which model is best? Which number would you report for your best model, and why?*

Yes. Before tuning, Decision Tree had the highest validation accuracy (0.695), while after tuning, Random Forest became the best model with a nested CV accuracy of 0.679 ± 0.016. I would report 0.679 ± 0.016 for Random Forest because it is the honest estimate obtained from the outer folds of nested cross-validation, rather than the Optuna best-trial score (0.682), which is optimistic because it was used to select the hyperparameters.


