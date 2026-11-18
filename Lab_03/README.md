# Lab 3: Exploratory Data Analysis (EDA)

**Course:** ARTI308 - Machine Learning, Imam Abdulrahman Bin Faisal University

In this lab we apply Exploratory Data Analysis (EDA) to a flower shop sales dataset: we check the data quality, clean it, and look for patterns using summary statistics and visualizations.

## Dataset

- **File:** `data/flowershopdata.csv`
- **Content:** sales records of a flower shop during 2016. Each row is one flower sale.
- **Size:** 106 rows and 11 columns in the original file; 49 rows and 8 columns after cleaning.
- **Source:** *[(add the dataset source / link here)](https://www.kaggle.com/datasets/alimodd/flower-shop-dataset)*

| Column | Description |
|---|---|
| Name | flower name |
| Description | short description of the flower |
| Needs | care needs of the flower |
| Season | blooming season |
| Status | sale status (all records are "Sold") |
| Date | date of the sale |
| Client Name | name of the client |
| Client phone number | phone number of the client |

## Project structure

```
Lab_03/
├── README.md
├── data/
│   └── flowershopdata.csv
└── src/
    └── Lab03_Assignment.ipynb
```

## How to run

1. Install the requirements:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```
2. Open `src/Lab03_Assignment.ipynb` with Jupyter or VS Code.
3. Make sure the notebook reads the data with the path `../data/flowershopdata.csv`.
4. Run all cells from top to bottom (**Kernel > Restart & Run All**).

## What was done

1. **Data types:** checked the type of each column.
2. **Data cleaning:**
   - removed 3 empty columns,
   - converted `Date` to a date type,
   - fixed broken characters (encoding problem) in `Season` and `Description`,
   - made the `Needs` text consistent.
3. **Missing values:** none were found.
4. **Duplicates:** 57 exact duplicate rows were found and removed (106 rows became 49).
5. **Descriptive statistics:** summary of all columns.
6. **Univariate analysis:** sales by flower, by client, by day of the week, and status/needs.
7. **Bivariate analysis:** flower vs. season, flower vs. month, and top clients vs. the flowers they buy.
8. **Correlation:** not possible, because the dataset has no numeric columns.
9. **Time analysis:** monthly and daily number of sales.

## Main findings

- Rose is the best-selling flower (16 sales, about one third of all sales); Zinnia sells the least (1 sale).
- There are 22 clients, and 16 of them bought more than once.
- Sales are highest on Tuesday. April and July are the busiest months, there are no sales in June, and October is the lowest.
- Almost every sales day has only one sale, so there are no big daily peaks.
- `Status` has only one value and `Needs` is almost always "well-drained soil", so these columns give little information.
- Each flower has exactly one blooming season.
