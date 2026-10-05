# My Diabetes Model - Simple Project Guide

Last updated: 5 October 2026.

## What did I build?

I built this project to practise machine learning. My model uses 16 inputs, including age and symptom answers, to predict a label from the dataset:

- `0` means Negative.
- `1` means Positive.

**My model is not a medical tool. It cannot confirm or rule out diabetes, recommend treatment, or predict when someone will develop diabetes.**

## Where did I get the data?

I used *Early Stage Diabetes Risk Prediction* (2020), published by the UCI Machine Learning Repository. The answers came from patient questionnaires at Sylhet Diabetes Hospital in Bangladesh.

- Source: https://doi.org/10.24432/C5VG8H
- License: CC BY 4.0. I credit the source when sharing or reusing the data.
- File: `diabetes_data_upload.csv`.
- Original size: 520 rows and 17 columns.
- There are 16 input columns and one answer column, `class`.

I keep `class` out of the inputs so the model cannot see the answer.

## How did I prepare the data?

I checked for missing values using pandas and found none. I removed 269 repeated copies of complete rows, keeping one copy of each. That left 251 rows: 173 Positive and 78 Negative. I did not change the original CSV.

I did this to avoid putting the same recorded example in both training and testing. However, different people can have identical answers. Removing repeats changes how common some profiles are. This was a choice for this experiment, not a rule for every project.

I split the remaining rows:

| Set | Total | Positive | Negative |
|---|---:|---:|---:|
| Training | 200 | 138 | 62 |
| Testing | 51 | 35 | 16 |

I used `train_test_split(test_size=0.20, random_state=42, stratify=y)`. It reserves about 20% for testing and keeps similar label proportions. The fixed random state lets me repeat the split.

## How does my pipeline work?

I used a pipeline to keep the preparation steps and prediction model together:

1. Fill missing ages using the training median.
2. Scale age using settings learned from training data.
3. Fill missing text values using the most common training value in each column.
4. Turn text categories into numbers using one-hot encoding.
5. Send the prepared numbers to the classifier.

I used `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, and `Pipeline`.

The 16 inputs become 31 numerical columns: one for age, plus two for each of the 15 category columns.

The pipeline learns preparation settings from training data only. When predicting, it reuses those settings. The selected tree does not normally need scaling, but I kept it in the shared preparation steps.

## How did I compare models?

I compared logistic regression and a decision tree.

I also used `DummyClassifier(strategy="most_frequent")` as a baseline. It ignores the inputs and always predicts Positive, the most common training label. This shows what a very simple rule can achieve.

For comparison, I used five fold cross-validation:

1. Split the 200 training rows into five groups.
2. Train a fresh pipeline on 160 rows.
3. Check it on the remaining 40 rows.
4. Repeat five times, changing the group used for checking.
5. Calculate the average F1 score.

Each round learns its preparation settings from its own 160 rows. The 51 test rows are not used here.

I used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and `cross_val_score(scoring="f1")`.

| Model | Average cross-validation F1 |
|---|---:|
| Logistic regression | 91.49% |
| Decision tree | 92.26% |

These F1 scores measure performance for the Positive label.

## How did I tune the model?

I used GridSearchCV to try eight combinations of tree settings using the same five folds:

- `max_depth`: 2, 3, 4, or 5. This limits the number of decisions along a tree path.
- `min_samples_leaf`: 1 or 3. This sets the minimum number of training examples in a final leaf.

That means 40 cross-validation training runs.

I selected these tree settings:


```text
max_depth = 3
min_samples_leaf = 1
random_state = 42
```


The best average F1 was 92.26%. These matched the original tree settings, so tuning did not improve the score.

With `refit=True`, the search then trained the selected pipeline on all 200 training rows. I got it using `search.best_estimator_`.

The tree's small lead does not prove it will always be better. The winning validation score may look too good because I used those same scores to choose the winner.

## What results did I get?

I evaluated the selected tree on the 51 test rows:

| Measure | Selected tree | Logistic regression | Always-Positive baseline |
|---|---:|---:|---:|
| Accuracy | 84.31% | 82.35% | 68.63% |
| Precision | 88.57% | 93.33% | 68.63% |
| Recall | 88.57% | 80.00% | 100.00% |
| F1 | 88.57% | 86.15% | 81.40% |

- **Accuracy:** how many predictions were correct overall.
- **Precision:** of the Positive predictions, how many were correct.
- **Recall:** of the actual Positive examples, how many were found.
- **F1:** combines precision and recall. It is not the percentage of correct predictions.

My tree produced this confusion matrix:

| Actual label | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Negative     |        12          |        4           |
| Positive     |        4           |        31          |

My model made 43 correct predictions and eight mistakes: four false positives and four false negatives.

Cross-validation and test scores use different groups of examples, so they do not need to match.

I looked at test results during the lessons before final model selection. Grid search used only training data, but I cannot call this a completely untouched final test.

## What limitations have I identified?

- The data is small and comes from one hospital. Performance elsewhere is unknown.
- There are only 51 test examples. Different splits could give different scores. I have not calculated how uncertain these scores are.
- I have not tested the model on an independent external dataset.
- I have not checked whether it performs equally well for different groups of people.
- I have not checked whether estimated probabilities match real observed rates. They are not proven personal medical risks.
- The saved model is a decision tree. The sigmoid explanation from the lessons applied to logistic regression, not this tree.
- Successful saving and loading does not prove the model is accurate or medically useful.

## What inputs does my model need?

I supply all 16 columns with these exact names:

```text
Age
Gender
Polyuria
Polydipsia
sudden weight loss
weakness
Polyphagia
Genital thrush
visual blurring
Itching
Irritability
delayed healing
partial paresis
muscle stiffness
Alopecia
Obesity
```

Spelling and capital letters matter. The pipeline can fill supported missing values inside columns, but it does not create absent columns.

Unknown categories become zeros for that input because of `handle_unknown="ignore"`. This avoids an encoding error; it does not make the input valid.

My prediction script does not check all possible input mistakes.

## How do I run it?

| File | Purpose |
|---|---|
| `training/train_model.py` | Prepares data, trains, compares, tunes, evaluates, saves, and checks loading. |
| `predict.py` | Loads the saved pipeline and predicts for an invented example without retraining. |
| `models/diabetes_pipeline.joblib` | Stores the trained tree and preparation steps. |
| `data/diabetes_data_upload.csv` | Keeps the original dataset. |

I run these commands from the `diabetes-prediction` folder in PowerShell:

```powershell
# Train again and replace the saved model file.
.\.venv\Scripts\python.exe training/train_model.py

# Load the saved model and predict without retraining.
.\.venv\Scripts\python.exe predict.py
```

I load the saved model separately in my API. I do not import the training script, because importing it would run its training and saving code.

I only load joblib files I created or trust. A harmful file can run code on my computer. I use the same library versions when loading, because other versions may not work correctly.

## How did I check saving and loading?

I compared the outputs before and after loading. The checks returned:

```text
Predictions match after loading: True
Probabilities match after loading: True
```

I used `np.array_equal()` to compare labels and `np.allclose()` to compare probability values. No retraining happened between saving and checking.

The review also tested this using a temporary model file without replacing the existing saved model.

## Which software versions did I use?

My project environment had these versions when checked on 4 October 2026:

| Software | Version |
|---|---|
| Python | 3.13.15 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| joblib | 1.6.0 |

These are the main libraries, not a complete list of every dependency.


