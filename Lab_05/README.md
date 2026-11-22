# Lab 5: Feature Engineering (Classification)

ARTI 308 - Machine Learning

## Dataset
**Talabat-style orders dataset** (`talabat_enhanced_orders2.csv`), provided with the lab.

- 25,000 rows, 23 columns (one row = one food delivery order).
- Target: `Order_Status` (Delivered, Cancelled, In Transit).
- The data is already clean: no missing values and no duplicate rows.

## Folder structure
```
Lab_05/
├── README.md
├── Lab05_Feature_Engineering.ipynb
└── data/
    └── talabat_enhanced_orders2.csv
```

## How to run
1. Keep the `data` folder next to the notebook.
2. Open `Lab05_Feature_Engineering.ipynb` in Jupyter or Colab.
3. Run all cells from top to bottom.

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn.

## What the notebook does
1. Removes leakage columns (`Delivery_Time`, `Delivery_Duration_Minutes`).
2. Creates new features: order hour, day of week, weekend and peak-hour flags, `price_per_item`, a reduced `Item_Name`, and `price_tier`.
3. Trains a Random Forest (with one-hot encoding in a Pipeline) and evaluates it.
4. Shows the feature importances.

## Task
Change `top_k` in `Item_Name_reduced` (3, 5, 6) and compare the accuracy and the top feature importances.

**Result:** the accuracy is about the same for all values (0.39 to 0.40) and the top 5 features do not change, so `top_k` has no clear effect.
