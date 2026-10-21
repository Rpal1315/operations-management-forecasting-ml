# BAMEE342 - Production and Operations Management
## Project: Machine Learning Demand & Price Forecasting Model

This project implements an end-to-end Machine Learning pipeline to forecast demand and prices (`Adjusted Close (INR)`) using supply chain, macroeconomic, and operational disruption indicators from [`Data/Book1.csv`](Data/Book1.csv).

---

## 📌 Project Overview

In Operations Management, accurate forecasting is critical for inventory control, capacity planning, procurement, and risk mitigation. This project models how external supply chain factors—such as raw material indices, freight costs, and operational disruptions—influence product price and demand valuation over a 250-day operational period.

### Key Objectives
1. **Chronological Time-Series Partitioning**: Train using strictly the **first 80%** (Trading Days 1 to 200) and evaluate generalizability on the **last 20%** (Trading Days 201 to 250) without look-ahead bias.
2. **Multi-Model Evaluation**: Benchmark Linear Regression, Ridge Regularization, Random Forest, and Gradient Boosting Regressors.
3. **POM Forecast Error Analysis**: Compute standard Operations Management forecasting metrics (MAD, MSE, RMSE, MAPE, MFE, Tracking Signal).
4. **Structured CSV & Visual Exports**: Generate comprehensive demand vs. forecast tables and publication-ready diagnostic plots.

---

## 📂 Dataset & Features

The dataset (`Data/Book1.csv`) consists of 250 operational trading days:

| Feature / Column | Type | Operations Management Significance |
| :--- | :---: | :--- |
| `Trading Day` | Integer | Chronological operational timeline (Days 1 to 250) |
| `Date` | Timestamp | Parsed to extract seasonal components (`Month`, `Day_of_Week`) |
| `Adjusted Close (INR)` | Target ($A_t$) | Actual demand data / price target variable to forecast |
| `Volume` | Float | Operational market volume / transaction intensity |
| `HRC_Steel_Index` | Float | Hot-Rolled Coil Steel Index (direct chassis & body manufacturing raw material cost) |
| `Lithium_Price_USD` | Integer | Global Lithium commodity price (battery and EV component cost indicator) |
| `Global_Freight_Index` | Integer | Shipping and logistics index (supply chain transport costs) |
| `JLR_Disruption_Shock`| Categorical | Binary indicator (`Yes` = 1, `NO` = 0) of operational/supply chain shocks |

---

## ⚙️ Methodology & Architecture

### 1. Preprocessing & Feature Engineering
- **Whitespace Cleaning**: Strips trailing whitespace from column headers (e.g., `'Trading Day '`).
- **Binary Shock Encoding**: Maps `JLR_Disruption_Shock` values (`'Yes'` $\to 1$, `'NO'` $\to 0$).
- **Calendar Decomposition**: Extracts `Month` and `Day_of_Week` from `Date` to capture cyclical patterns.

### 2. Chronological Split (80% Train / 20% Test)
- **Training Set (First 80%)**: Periods 1 to 200 (14-Oct-2025 to 05-Aug-2026)
- **Testing Set (Last 20%)**: Periods 201 to 250 (06-Aug-2026 to 16-Oct-2026)
- Enforces strict chronological order (`shuffle=False`) to simulate real-world forecasting where future demand is unknown.

### 3. Model Benchmark
Four regression models are trained and tested:
1. **Linear Regression** (Ordinary Least Squares)
2. **Ridge Regression** ($L_2$ Regularized Linear Model)
3. **Random Forest Regressor** (Ensemble of 150 bagged decision trees)
4. **Gradient Boosting Regressor** (Sequentially boosted shallow trees)

---

## 📊 Model Performance on Last 20% Test Set (Days 201–250)

| Model | Train $R^2$ | Test MAE (MAD) | Test RMSE | Test MAPE (%) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Best)** | **0.1555** | **₹29.64** | **₹33.10** | **10.00%** | **-1.6228** |
| **Ridge Regression** | 0.3174 | ₹37.94 | ₹42.64 | 12.82% | -3.3527 |
| **Random Forest** | 0.9887 | ₹40.00 | ₹44.65 | 13.51% | -3.7727 |
| **Gradient Boosting** | 0.9989 | ₹41.31 | ₹46.11 | 13.95% | -4.0898 |

### Why Linear Regression Performed Best on Out-of-Sample Forecasting
During the training window (Days 1–200), prices averaged ₹356.29 (range: ₹293.76 to ₹413.57). In the last 20% testing window (Days 201–250), prices continued their secular downward trend to ₹273.00. 
- **Tree-based models (Random Forest, Gradient Boosting)** partition feature space into piecewise constant steps and *cannot extrapolate* values below the lowest target observed in the training split (₹293.76).
- **Linear models** leverage continuous slope coefficients (e.g., negative coefficients on `Global_Freight_Index`, `Lithium_Price_USD`, and `Trading Day`) to extrapolate downward trends into unseen territory, achieving the lowest MAD (₹29.64) and MAPE (10.00%).

---

## 📐 Operations Management (POM) Error Formulas & Statistics

For actual demand $A_t$ and forecast $F_t$ across $n = 50$ test periods:

| Metric | Formula | Value (Test Set) | Operational Interpretation |
| :--- | :---: | :---: | :--- |
| **Forecast Error ($e_t$)** | $e_t = A_t - F_t$ | — | Raw deviation between actual and forecast |
| **Mean Absolute Deviation (MAD / MAE)** | $\frac{1}{n}\sum \|e_t\|$ | **₹29.64** | Average magnitude of forecast error |
| **Mean Squared Error (MSE)** | $\frac{1}{n}\sum e_t^2$ | **1,095.52** | Penalizes larger forecasting errors |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\text{MSE}}$ | **₹33.10** | Standard deviation of forecast residuals |
| **Mean Absolute Percentage Error (MAPE)** | $\frac{1}{n}\sum \left\|\frac{e_t}{A_t}\right\| \times 100\%$ | **10.00%** | Relative percentage error |
| **Mean Forecast Error (MFE / Bias)** | $\frac{1}{n}\sum e_t$ | **-₹27.68** | Persistent tendency to over- or under-forecast |
| **Tracking Signal (TS)** | $\frac{\sum e_t}{\text{MAD}}$ | **-46.69** | Cumulative drift metric indicating structural trend shift |

---

## 📁 Repository Structure

```text
├── Data/
│   └── Book1.csv                         # Raw 250-day operational dataset
├── output/
│   ├── actual_vs_predicted.png           # Actual demand vs. forecast curve (Test Set)
│   ├── feature_importances.png           # Feature coefficients / importance plot
│   ├── model_comparison.png              # Multi-model R² and MAE benchmark bar charts
│   ├── demand_forecast_and_error.csv     # Complete 250-period dataset with split labels
│   └── demand_forecast_test_error.csv    # Dedicated 50-period test holdout error log
├── demand_forecast_and_error.csv         # Direct root copy of forecast results
├── main.py                               # Core Python executable script
└── README.md                             # Documentation
```

---

## 📄 Output CSV Format

Both `demand_forecast_and_error.csv` and `output/demand_forecast_test_error.csv` contain the following columns:

```csv
Trading_Day,Date,Dataset_Partition,Actual_Demand,Forecast,Forecast_Error,Absolute_Error,Percentage_Error_(%),Squared_Error
200,05-08-2026,Train (First 80%),349.8000,338.5475,11.2525,11.2525,3.2168,126.6196
201,06-08-2026,Test (Last 20%),345.3500,336.7209,8.6291,8.6291,2.4987,74.4613
202,07-08-2026,Test (Last 20%),347.0000,338.3383,8.6617,8.6617,2.4962,75.0257
...
```

---

## 🚀 How to Run

### Requirements
- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`

Install dependencies:
```powershell
pip install pandas numpy scikit-learn matplotlib
```

Execute the forecasting pipeline:
```powershell
python main.py
```

Upon execution, the script will:
1. Load and clean `Data/Book1.csv`.
2. Partition the data (first 80% train, last 20% test).
3. Train all four regression models.
4. Print the performance comparison and POM error metrics.
5. Save high-resolution plots to `output/`.
6. Export the forecast and error CSV files.

