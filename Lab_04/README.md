# Lab 4: Data Quality Assessment & Preprocessing

ARTI 308 - Machine Learning

## Dataset
**Auto MPG** (UCI Machine Learning Repository, license CC BY 4.0).
Source: https://archive.ics.uci.edu/dataset/9/auto+mpg

- 398 rows, 9 columns (mpg, cylinders, displacement, horsepower, weight, acceleration, model_year, origin, car_name).
- The file has no header, values are separated by spaces, and missing values are written as `?`.
- `horsepower` has 6 missing values.

## Folder structure
```
Lab_04/
├── README.md
├── Lab04_Assignment.ipynb
└── data/
    └── auto-mpg.data
```

## How to run
1. Keep the `data` folder next to the notebook.
2. Open `Lab04_Assignment.ipynb` in Jupyter or Colab.
3. Run all cells from top to bottom.

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn.

## Tasks covered
1. **Data quality issues:** data types, missing values, duplicates.
2. **Missing values:** median imputation of `horsepower` (the notebook also shows removing rows and mean imputation for comparison).
3. **Outliers:** detected with the IQR method, then handled by removing or capping.
4. **Normalization:** Min-Max and Z-score.
5. **PCA:** applied only if the features are correlated (|r| > 0.7).

Each task has an **Observation** cell for the written explanation.
