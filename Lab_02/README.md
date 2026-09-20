# Machine Learning-Based Classification of Student GPA Categories

A supervised classification project that predicts a student's grade category (`GradeClass`) from demographic, academic, and extracurricular features. Two tree-based models are trained and compared: a single Decision Tree and a Random Forest.

## Dataset

| | |
|---|---|
| Rows | 2,392 |
| Columns (raw) | 15 |
| Features used | 11 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Source file | `../data/raw.csv` |
| Cleaned file | `../data/processed.csv` |

### Columns

| Column | Type | Description |
|---|---|---|
| `StudentID` | int | Unique identifier (dropped) |
| `Age` | int | 15–18 |
| `Gender` | int | Binary encoded |
| `Ethnicity` | int | Categorical code (dropped) |
| `ParentalEducation` | int | Ordinal, 0–4 |
| `StudyTimeWeekly` | float | Hours per week, 0–20 |
| `Absences` | int | 0–29 |
| `Tutoring` | int | 0 = no, 1 = yes |
| `ParentalSupport` | int | Ordinal, 0–4 |
| `Extracurricular` | int | 0 = no, 1 = yes |
| `Sports` | int | 0 = no, 1 = yes |
| `Music` | int | 0 = no, 1 = yes |
| `Volunteering` | int | 0 = no, 1 = yes |
| `GPA` | float | 0.0–4.0 (dropped — see below) |
| `GradeClass` | float | **Target**, 5 classes (0–4) |

### Target distribution

| Class | Count | Share |
|---|---|---|
| 4.0 | 1,211 | 50.6% |
| 3.0 | 414 | 17.3% |
| 2.0 | 391 | 16.3% |
| 1.0 | 269 | 11.2% |
| 0.0 | 107 | 4.5% |

The target is heavily imbalanced — class 4 alone is half the data. This matters when reading the results below: a model that predicted class 4 for everything would already score ~51% accuracy.

Class labels follow the conventional letter-grade banding used in this dataset (0 = A, 1 = B, 2 = C, 3 = D, 4 = F), where higher codes mean lower GPA. Worth confirming against the original data dictionary if you publish this.

## Preprocessing

Three columns are dropped:

- **`StudentID`** — an identifier with no predictive signal.
- **`GPA`** — `GradeClass` is derived directly from GPA, so keeping it would leak the answer and push accuracy to ~100% for the wrong reason.
- **`Ethnicity`** — removed from the feature set.

No imputation, scaling, or encoding is needed: there are no nulls or duplicates, and every remaining column is already numeric. Tree-based models don't require feature scaling.

## Method

1. Split features `X` (11 columns) from target `y` (`GradeClass`).
2. `train_test_split` with `test_size=0.2`, `random_state=42`, `stratify=y` → 1,913 train / 479 test. Stratifying keeps the class proportions intact in both splits, which matters given the imbalance.
3. Fit `DecisionTreeClassifier(random_state=42)` — default depth, so it grows until leaves are pure.
4. Fit `RandomForestClassifier(n_estimators=100, random_state=42)`.
5. Evaluate both with accuracy, a per-class classification report, and a confusion matrix heatmap.

## Results

| Model | Accuracy |
|---|---|
| Decision Tree | 0.580 |
| **Random Forest** | **0.704** |

The Random Forest gains about 12 percentage points. The unconstrained Decision Tree overfits the training data; averaging 100 trees reduces that variance.

### Decision Tree — per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0.0 | 0.22 | 0.19 | 0.21 | 21 |
| 1.0 | 0.33 | 0.43 | 0.37 | 54 |
| 2.0 | 0.39 | 0.36 | 0.37 | 78 |
| 3.0 | 0.34 | 0.39 | 0.36 | 83 |
| 4.0 | 0.85 | 0.79 | 0.82 | 243 |
| **Macro avg** | 0.43 | 0.43 | 0.43 | 479 |
| **Weighted avg** | 0.60 | 0.58 | 0.59 | 479 |

### Random Forest — per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0.0 | 0.38 | 0.14 | 0.21 | 21 |
| 1.0 | 0.51 | 0.48 | 0.50 | 54 |
| 2.0 | 0.51 | 0.60 | 0.55 | 78 |
| 3.0 | 0.48 | 0.48 | 0.48 | 83 |
| 4.0 | 0.90 | 0.91 | 0.91 | 243 |
| **Macro avg** | 0.56 | 0.52 | 0.53 | 479 |
| **Weighted avg** | 0.70 | 0.70 | 0.70 | 479 |

Both models do well on class 4 and poorly on class 0, which has only 21 test samples. The Random Forest's headline accuracy improves, but its recall on class 0 actually drops to 0.14 — it recovers just 3 of 21 top students. Macro F1 (0.53) is the more honest summary of overall performance here than accuracy (0.70).

### Feature importance (Random Forest)

| Feature | Importance |
|---|---|
| Absences | 0.457 |
| StudyTimeWeekly | 0.188 |
| ParentalSupport | 0.073 |
| ParentalEducation | 0.062 |
| Age | 0.061 |
| Gender | 0.030 |
| Sports | 0.029 |
| Extracurricular | 0.027 |
| Tutoring | 0.027 |
| Music | 0.024 |
| Volunteering | 0.021 |

Absences and weekly study time account for roughly 65% of total importance. The binary activity flags contribute very little individually. Note that impurity-based importance tends to favor high-cardinality and continuous features, so `Absences` and `StudyTimeWeekly` have a structural advantage over the 0/1 columns — permutation importance would give a fairer read.

## Requirements

```
pandas
scikit-learn
matplotlib
seaborn
```

Install with:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Project structure

```
project/
├── data/
│   ├── raw.csv           # original dataset
│   └── processed.csv     # after dropping StudentID, GPA, Ethnicity
└── notebooks/
    └── GPA_Classification.ipynb
```

Paths in the notebook are relative (`../data/raw.csv`), so it expects to be run from the `notebooks/` directory.

## Usage

```bash
jupyter notebook notebooks/GPA_Classification.ipynb
```

Run the cells top to bottom. `random_state=42` is set on the split and both models, so results are reproducible.

## Possible next steps

- **Handle the imbalance** — try `class_weight="balanced"`, or resampling, to improve recall on the minority classes.
- **Tune hyperparameters** — `max_depth` and `min_samples_leaf` on the Decision Tree would curb the overfitting; `GridSearchCV` or `RandomizedSearchCV` on the Random Forest.
- **Cross-validate** — a single 80/20 split on 2,392 rows gives a noisy estimate, especially for class 0 with 21 test samples. `StratifiedKFold` would be more reliable.
- **Report macro F1 alongside accuracy** so minority-class performance isn't hidden.
- **Try other models** — gradient boosting (XGBoost, LightGBM) usually beats a plain Random Forest on tabular data of this shape.
- **Reconsider ordinality** — `GradeClass` is ordered, so misclassifying an A as a B is less wrong than as an F. An ordinal model or a regression on GPA followed by binning might suit the problem better.
