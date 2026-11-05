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

### Week 6 (14/10) - Emsembles + Feature Engineering

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.673 ± 0.015 (+0.002) | 0.677 ± 0.013 (+0.001) | 0.681 | `C=0.00534, solver='liblinear', class_weight=None` |
| Decision tree | 0.605 ± 0.013 (+0.094) | 0.670 ± 0.006 (+0.010) | 0.679 | `max_depth=17, min_samples_leaf=95, criterion='gini'` |
| Random forest | 0.644 ± 0.018 (+0.088) | 0.676 ± 0.011 (+0.013) | 0.681 | `n_estimators=183, max_depth=25, min_samples_leaf=28, max_features=0.5` |
| Oblique random forest | 0.645 ± 0.012 (+0.119) | 0.679 ± 0.012 (+0.014) | 0.682 | `tree__max_depth=5, tree__min_samples_leaf=8, project__n_components=8` |
| AdaBoost | 0.675 ± 0.011 (+0.006) | 0.681 ± 0.014 (+0.007) | 0.684 | `estimator__max_depth=6, n_estimators=68, learning_rate=0.1715` |

**Best model**

**AdaBoost** is the best-performing model based on nested cross-validation, achieving a mean accuracy of 68.1% ± 1.4%. It slightly outperforms the oblique random forest (67.9%), Logistic Regression (67.7%), Decision Tree (67.0%), and Random Forest (67.6%). Its train–validation gap of 0.007 indicates good generalisation, with little evidence of overfitting.


1. Which model would you choose, and is its lead bigger than the fold-to-fold std?

I would choose AdaBoost. Its lead over the oblique random forest is only 0.2 percentage points, much smaller than either model's standard deviation. Therefore, the difference is not large enough to establish a clear winner.

2. Do the chosen hyperparameters match how each model works?

Yes, the hyperparameters are generally consistent with each model's characteristics. AdaBoost uses trees of depth 6, 68 estimators, and a learning rate of 0.1715, combining moderately complex trees with gradual learning. The other models also use regularisation or tree constraints to balance model complexity and generalisation.

3. Which model and which number would you report?

I would report AdaBoost with a nested CV accuracy of 68.1% ± 1.4%. Nested cross-validation provides a more reliable estimate than the best Optuna trial score of 68.4%, which may be optimistic. 

## Week 7 (21/10) - Gradient Boosting + Metrics

| Model | Default: CV F1 mean ± std (gap) | Tuned: nested CV F1 mean ± std | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Hist. Gradient Boosting | 0.626 ± 0.020 (+0.043) | 0.641 ± 0.017 | 0.644 | `max_iter=133, learning_rate=0.2013, max_leaf_nodes=2, min_samples_leaf=165, l2_regularization=8.234e-05` |
| Random Forest | 0.603 ± 0.008 (+0.097) | 0.640 ± 0.017 | 0.644 | `max_depth=6, min_samples_leaf=1, max_features='sqrt'` |
| Logistic Regression | 0.634 ± 0.013 (-0.001) | 0.637 ± 0.010 | 0.638 | `C=0.006725, solver='lbfgs'` |
| Decision Tree | 0.562 ± 0.014 (+0.101) | 0.637 ± 0.013 | 0.634 | `max_depth=2, min_samples_leaf=148, criterion='gini'` |
| Gradient Boosting | 0.610 ± 0.018 (+0.017) | 0.621 ± 0.017 | 0.625 | `n_estimators=327, learning_rate=0.3976, max_depth=1, min_samples_leaf=200, subsample=0.9351` |
| Oblique Random Forest | 0.591 ± 0.009 (+0.132) | 0.617 ± 0.009 | 0.619 | `tree__max_depth=8, tree__min_samples_leaf=73, project__n_components=9` |
| AdaBoost | 0.605 ± 0.019 (+0.007) | 0.615 ± 0.012 | 0.618 | `estimator__max_depth=7, n_estimators=130, learning_rate=0.268` |
| Naive Bayes | 0.434 ± 0.016 (-0.002) | 0.435 ± 0.014 | 0.435 | `var_smoothing=0.00091749` |

**Best model**

I would choose **Hist. Gradient Boosting**, as it achieved the highest nested CV F1 score (0.641 ± 0.017), indicating the best overall balance between precision and recall among the tested models. However, Random Forest (0.640 ± 0.017) and Logistic Regression (0.637 ± 0.010) performed very similarly, so the differences are small; I would validate the final choice on the locked test set before drawing a definitive conclusion.



## Week 8 (04/11) - XGBoost + Feature Selection

| Model | Default: CV F1 mean ± std (gap) | Tuned: nested CV F1 mean ± std | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic Regression | 0.635 ± 0.009 (+0.004) | **0.642 ± 0.009** | 0.642 | `C=0.07459, min_votes=3` |
| Naive Bayes | 0.520 ± 0.096 (+0.012) | 0.636 ± 0.010 | 0.636 | `var_smoothing=5.202e-11, min_votes=1` |
| Hist. Gradient Boosting | 0.615 ± 0.021 (+0.032) | 0.637 ± 0.014 | 0.637 | `max_iter=302, learning_rate=0.02295, max_leaf_nodes=3, min_samples_leaf=9, l2_regularization=0.000135, min_votes=2` |
| Decision Tree | 0.579 ± 0.031 (+0.079) | 0.634 ± 0.018 | 0.634 | `max_depth=8, min_samples_leaf=149, criterion='gini', min_votes=1` |
| Random Forest | 0.583 ± 0.032 (+0.074) | 0.629 ± 0.019 | 0.634 | `max_depth=12, min_samples_leaf=149, max_features='sqrt', min_votes=1` |
| Oblique Random Forest | 0.564 ± 0.034 (+0.080) | 0.615 ± 0.007 | 0.615 | `tree__max_depth=12, tree__min_samples_leaf=149, project__n_components=8, min_votes=2` |
| XGBoost | 0.603 ± 0.017 (+0.017) | 0.613 ± 0.014 | 0.613 | `n_estimators=126, learning_rate=0.2537, max_depth=6, min_child_weight=10.402, subsample=0.578, colsample_bytree=0.578, reg_lambda=0.01494, reg_alpha=2.915, gamma=3.006, min_votes=3` |
| AdaBoost | 0.598 ± 0.018 (+0.009) | 0.611 ± 0.020 | 0.611 | `estimator__max_depth=3, n_estimators=106, learning_rate=0.07310, min_votes=1` |
| Gradient Boosting | 0.602 ± 0.012 (+0.011) | 0.608 ± 0.015 | 0.608 | `n_estimators=302, learning_rate=0.02295, max_depth=2, min_samples_leaf=2, subsample=0.6521, min_votes=2` |


| Model | Feature Selection | Nested CV F1 mean ± std | Precision | Recall | F2 | ROC-AUC | Average Precision |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | Enabled (`min_votes=3`) | 0.613 ± 0.014 | 0.677 | 0.562 | 0.581 | 0.724 | 0.688 |
| XGBoost | Disabled | **0.622 ± 0.014** | 0.664 | **0.586** | **0.600** | **0.726** | **0.689** |

**Best Model** 

The **Logistic Regression** model achieved the best overall performance, with a nested cross-validation F1-score of 0.642 ± 0.009. It also showed very consistent results across folds, with a low standard deviation of 0.009. Using Optuna, the model was tuned with a regularization parameter of C=0.07459 and min_votes=3. Its balanced performance, stability, and relatively simple structure make it the strongest candidate among the evaluated models.

**Is Feature Selection Worth It?**

