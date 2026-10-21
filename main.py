"""
BAMEE342 - Production and Operations Management
Project: Machine Learning Model to Predict Prices
Dataset: Data/Book1.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def load_and_preprocess_data(file_path):
    """
    Loads and cleans data from CSV.
    Handles column whitespace, date parsing, and categorical encoding.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Strip any leading/trailing spaces from column headers
    df.columns = [c.strip() for c in df.columns]

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    # Encode binary shock feature: 'Yes' -> 1, 'NO' -> 0
    df['JLR_Disruption_Shock'] = (
        df['JLR_Disruption_Shock']
        .astype(str)
        .str.strip()
        .str.upper()
        .map({'YES': 1, 'NO': 0})
        .fillna(0)
        .astype(int)
    )

    # Extract temporal signals
    df['Month'] = df['Date'].dt.month
    df['Day_of_Week'] = df['Date'].dt.dayofweek

    return df


def prepare_features(df):
    """
    Prepares feature matrix X and target y.
    """
    target_col = 'Adjusted Close (INR)'
    feature_cols = [
        'Trading Day',
        'Volume',
        'HRC_Steel_Index',
        'Lithium_Price_USD',
        'Global_Freight_Index',
        'JLR_Disruption_Shock',
        'Month',
        'Day_of_Week'
    ]

    # Ensure all required features are present
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from data.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y, feature_cols, target_col


def evaluate_model(y_true, y_pred):
    """
    Computes key regression evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE (%)': mape
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains multiple ML regression models and compares their performance.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = {}
    fitted_models = {}
    test_predictions = {}

    for name, model in models.items():
        # Fit model on training set (80%)
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predict on train and test
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        test_predictions[name] = y_test_pred

        # Evaluate metrics
        train_metrics = evaluate_model(y_train, y_train_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)

        results[name] = {
            'Train R2': train_metrics['R2'],
            'Test R2': test_metrics['R2'],
            'Test MAE': test_metrics['MAE'],
            'Test RMSE': test_metrics['RMSE'],
            'Test MAPE (%)': test_metrics['MAPE (%)']
        }

    results_df = pd.DataFrame(results).T
    return fitted_models, results_df, test_predictions


def plot_results(y_test, test_predictions, best_model_name, best_model, feature_names, test_trading_days=None, output_dir='output'):
    """
    Generates and saves performance and diagnostic plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Actual vs Predicted for the Best Model
    plt.figure(figsize=(10, 6))
    y_test_arr = np.array(y_test)
    best_preds = test_predictions[best_model_name]
    x_axis = test_trading_days if test_trading_days is not None else np.arange(len(y_test_arr))
    x_label = 'Trading Day' if test_trading_days is not None else 'Test Sample Index'

    plt.plot(x_axis, y_test_arr, marker='o', label='Actual Demand / Price (INR)', color='#1f77b4', linewidth=2)
    plt.plot(x_axis, best_preds, marker='s', linestyle='--', label=f'Forecast ({best_model_name})', color='#ff7f0e', linewidth=2)
    plt.title(f'Actual vs Forecast Demand on Test Set (Last 20% Holdout)\nBest Model: {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Adjusted Close Price / Demand (INR)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    pred_plot_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()

    # 2. Feature Importances or Coefficients
    plt.figure(figsize=(9, 5))
    if hasattr(best_model, 'feature_importances_'):
        weights = best_model.feature_importances_
        sorted_idx = np.argsort(weights)
        plt.barh(np.array(feature_names)[sorted_idx], weights[sorted_idx], color='#2ca02c')
        plt.title(f'Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
    elif hasattr(best_model, 'coef_'):
        weights = best_model.coef_
        sorted_idx = np.argsort(np.abs(weights))
        colors = ['#d62728' if w < 0 else '#2ca02c' for w in weights[sorted_idx]]
        plt.barh(np.array(feature_names)[sorted_idx], weights[sorted_idx], color=colors)
        plt.title(f'Feature Coefficients (Weights) - {best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Regression Coefficient Value', fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    feat_plot_path = os.path.join(output_dir, 'feature_importances.png')
    plt.savefig(feat_plot_path, dpi=300)
    plt.close()

    # 3. Model Comparison (R2 and MAE)
    metrics_summary = []
    for model_name, preds in test_predictions.items():
        m = evaluate_model(y_test, preds)
        metrics_summary.append({'Model': model_name, 'R2': m['R2'], 'MAE': m['MAE']})
    comp_df = pd.DataFrame(metrics_summary)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(comp_df))
    width = 0.35

    rects1 = ax1.bar(x - width/2, comp_df['R2'], width, label='R^2 Score', color='#3470a3')
    rects2 = ax2.bar(x + width/2, comp_df['MAE'], width, label='MAE (INR)', color='#e26d5c')

    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_ylabel('R^2 Score', color='#3470a3', fontsize=12)
    ax2.set_ylabel('MAE (INR) (Lower is better)', color='#e26d5c', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_df['Model'], fontsize=11)
    plt.title('Model Performance Comparison (Last 20% Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()

    print(f"\n[+] Visualization plots saved to directory: '{output_dir}/'")


def export_forecast_csv(df, X_train, X_test, y_train, y_test, best_model, output_dir='output', base_dir='.'):
    """
    Generates and saves CSV files containing:
      - Demand data (Actual demand / adjusted close price)
      - Forecast (Model predicted value)
      - Forecast error (Actual - Forecast)
      - Absolute error (|Actual - Forecast|)
      - Percentage error (%)
      - Squared error
    Saves both the 20% holdout test set and the complete dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    test_indices = X_test.index
    train_indices = X_train.index

    y_test_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)

    # 1. Dedicated Test Set Results (20% Holdout)
    test_results_df = pd.DataFrame({
        'Trading_Day': df.loc[test_indices, 'Trading Day'],
        'Date': df.loc[test_indices, 'Date'].dt.strftime('%d-%m-%Y'),
        'Actual_Demand': df.loc[test_indices, 'Adjusted Close (INR)'],
        'Forecast': y_test_pred,
        'Forecast_Error': df.loc[test_indices, 'Adjusted Close (INR)'] - y_test_pred,
        'Absolute_Error': np.abs(df.loc[test_indices, 'Adjusted Close (INR)'] - y_test_pred),
        'Percentage_Error_(%)': (np.abs(df.loc[test_indices, 'Adjusted Close (INR)'] - y_test_pred) / df.loc[test_indices, 'Adjusted Close (INR)']) * 100,
        'Squared_Error': (df.loc[test_indices, 'Adjusted Close (INR)'] - y_test_pred) ** 2
    }).sort_values('Trading_Day').reset_index(drop=True)

    test_csv_path = os.path.join(output_dir, 'demand_forecast_test_error.csv')
    test_results_df.round(4).to_csv(test_csv_path, index=False)

    # 2. Complete Dataset Results (Train + Test with Partition Indicator)
    full_df = pd.DataFrame({
        'Trading_Day': df['Trading Day'],
        'Date': df['Date'].dt.strftime('%d-%m-%Y'),
        'Dataset_Partition': 'Train (First 80%)',
        'Actual_Demand': df['Adjusted Close (INR)'],
        'Forecast': np.nan,
        'Forecast_Error': np.nan,
        'Absolute_Error': np.nan,
        'Percentage_Error_(%)': np.nan,
        'Squared_Error': np.nan
    })

    full_df.loc[train_indices, 'Forecast'] = y_train_pred
    full_df.loc[test_indices, 'Forecast'] = y_test_pred
    full_df.loc[test_indices, 'Dataset_Partition'] = 'Test (Last 20%)'

    full_df['Forecast_Error'] = full_df['Actual_Demand'] - full_df['Forecast']
    full_df['Absolute_Error'] = np.abs(full_df['Forecast_Error'])
    full_df['Percentage_Error_(%)'] = (full_df['Absolute_Error'] / full_df['Actual_Demand']) * 100
    full_df['Squared_Error'] = full_df['Forecast_Error'] ** 2

    full_results_df = full_df.sort_values('Trading_Day').reset_index(drop=True)

    full_csv_path = os.path.join(output_dir, 'demand_forecast_and_error.csv')
    root_csv_path = os.path.join(base_dir, 'demand_forecast_and_error.csv')

    full_results_df.round(4).to_csv(full_csv_path, index=False)
    full_results_df.round(4).to_csv(root_csv_path, index=False)

    print("\n" + "=" * 70)
    print(" FORECAST CSV EXPORT SUMMARY ")
    print("=" * 70)
    print(f"[+] Saved complete dataset forecast & error CSV to:")
    print(f"    - {full_csv_path}")
    print(f"    - {root_csv_path}")
    print(f"[+] Saved test set forecast & error CSV to:")
    print(f"    - {test_csv_path}")

    # POM Summary Statistics for the Test Set
    mad = test_results_df['Absolute_Error'].mean()
    mse = test_results_df['Squared_Error'].mean()
    rmse = np.sqrt(mse)
    mape = test_results_df['Percentage_Error_(%)'].mean()
    mfe = test_results_df['Forecast_Error'].mean()
    tracking_signal = test_results_df['Forecast_Error'].sum() / mad if mad > 0 else 0

    print("\n" + "-" * 70)
    print(" POM FORECAST ACCURACY & ERROR ANALYSIS (TEST SET - 50 PERIODS) ")
    print("-" * 70)
    print(f"  Mean Absolute Deviation (MAD / MAE):  INR {mad:.2f}")
    print(f"  Mean Squared Error (MSE):             {mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE):       INR {rmse:.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE):{mape:.2f}%")
    print(f"  Mean Forecast Error (MFE / Bias):     INR {mfe:.2f}")
    print(f"  Tracking Signal:                      {tracking_signal:.2f}")
    print("-" * 70)

    print("\nPreview of CSV Output (First 5 rows of test set):")
    print(test_results_df.head().to_string(index=False, formatters={
        'Actual_Demand': '{:.2f}'.format,
        'Forecast': '{:.2f}'.format,
        'Forecast_Error': '{:.2f}'.format,
        'Absolute_Error': '{:.2f}'.format,
        'Percentage_Error_(%)': '{:.2f}%'.format,
        'Squared_Error': '{:.2f}'.format
    }))

    return full_results_df, test_results_df


def main():
    # Path configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'Data', 'Book1.csv')

    print("=" * 70)
    print(" BAMEE342 - POM PROJECT: MACHINE LEARNING PRICE PREDICTION MODEL ")
    print("=" * 70)
    print(f"Loading data from: {data_path}")

    # Step 1: Load and preprocess data
    df = load_and_preprocess_data(data_path)
    print(f"Dataset successfully loaded. Total records: {len(df)}")
    print("Features extracted: Date parsed, JLR Shock encoded, calendar features added.")

    # Step 2: Prepare features and target
    X, y, feature_names, target_name = prepare_features(df)
    print(f"Features: {feature_names}")
    print(f"Target variable: {target_name}")

    # Step 3: Chronological Split (First 80% train, Last 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    train_days_min = df.loc[X_train.index, 'Trading Day'].min()
    train_days_max = df.loc[X_train.index, 'Trading Day'].max()
    test_days_min = df.loc[X_test.index, 'Trading Day'].min()
    test_days_max = df.loc[X_test.index, 'Trading Day'].max()

    print(f"\nData Split (Chronological: First 80% Train, Last 20% Test):")
    print(f"  Training set (First 80%): Trading Days {train_days_min} to {train_days_max} ({len(X_train)} samples, {len(X_train)/len(X)*100:.1f}%)")
    print(f"  Testing set  (Last 20%):  Trading Days {test_days_min} to {test_days_max} ({len(X_test)} samples, {len(X_test)/len(X)*100:.1f}%)")

    # Step 4: Train and Evaluate Models
    print("\nTraining and evaluating models on chronological split...")
    fitted_models, results_df, test_predictions = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    print("\n" + "=" * 70)
    print(" MODEL PERFORMANCE SUMMARY (EVALUATED ON LAST 20% TEST SET) ")
    print("=" * 70)
    print(results_df.to_string(formatters={
        'Train R2': '{:.4f}'.format,
        'Test R2': '{:.4f}'.format,
        'Test MAE': '{:.2f}'.format,
        'Test RMSE': '{:.2f}'.format,
        'Test MAPE (%)': '{:.2f}%'.format
    }))

    # Identify best model based on minimum Test MAE (standard criterion for forecasting accuracy)
    best_model_name = results_df['Test MAE'].idxmin()
    best_model = fitted_models[best_model_name]
    best_mae = results_df.loc[best_model_name, 'Test MAE']
    best_rmse = results_df.loc[best_model_name, 'Test RMSE']
    best_mape = results_df.loc[best_model_name, 'Test MAPE (%)']
    best_r2 = results_df.loc[best_model_name, 'Test R2']

    print("\n" + "-" * 70)
    print(f"[*] BEST PERFORMING MODEL: {best_model_name.upper()}")
    print(f"    - Test MAE (MAD):   INR {best_mae:.2f}")
    print(f"    - Test RMSE:        INR {best_rmse:.2f}")
    print(f"    - Test MAPE:        {best_mape:.2f}%")
    print(f"    - Test R^2 Score:   {best_r2:.4f}")
    print("-" * 70)

    # Step 5: Feature Importances / Coefficients
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nFeature Importances ({best_model_name}):")
        feat_imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
        for feat, imp in feat_imp.items():
            print(f"  {feat:<22} : {imp*100:6.2f}%")
    elif hasattr(best_model, 'coef_'):
        print(f"\nFeature Coefficients ({best_model_name}):")
        feat_coef = pd.Series(best_model.coef_, index=feature_names).sort_values(ascending=False)
        for feat, coef in feat_coef.items():
            print(f"  {feat:<22} : {coef:10.5f}")
        print(f"  {'Intercept':<22} : {best_model.intercept_:10.5f}")

    # Step 6: Sample Actual vs Predicted Prices Table
    sample_df = pd.DataFrame({
        'Trading Day': df.loc[X_test.index[:10], 'Trading Day'].values,
        'Actual (INR)': y_test.values[:10],
        'Predicted (INR)': test_predictions[best_model_name][:10],
        'Abs Error (INR)': np.abs(y_test.values[:10] - test_predictions[best_model_name][:10])
    })
    print("\nSample Test Predictions (First 10 test periods):")
    print(sample_df.to_string(index=False, formatters={
        'Trading Day': '{:d}'.format,
        'Actual (INR)': '{:.2f}'.format,
        'Predicted (INR)': '{:.2f}'.format,
        'Abs Error (INR)': '{:.2f}'.format
    }))

    # Step 7: Plot Results
    output_dir = os.path.join(base_dir, 'output')
    test_trading_days = df.loc[X_test.index, 'Trading Day'].values
    plot_results(y_test, test_predictions, best_model_name, best_model, feature_names, test_trading_days=test_trading_days, output_dir=output_dir)

    # Step 8: Export Forecast and Error CSV
    export_forecast_csv(df, X_train, X_test, y_train, y_test, best_model, output_dir=output_dir, base_dir=base_dir)

    print("\nExecution complete.")


if __name__ == '__main__':
    main()
