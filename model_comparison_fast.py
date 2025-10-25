"""
Model Comparison: Linear Regression vs MLP Backpropagation (Optimized Version)
Prediksi Harga Rumah menggunakan California Housing Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(data_path):
    """Load dan preprocess data"""
    print("=" * 60)
    print("LOADING DAN PREPROCESSING DATA")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Pilih fitur numerik
    numeric_features = ['longitude', 'latitude', 'housing_median_age', 
                       'total_rooms', 'total_bedrooms', 'population', 
                       'households', 'median_income']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(df[numeric_features]), 
                     columns=numeric_features)
    y = df['median_house_value']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train Linear Regression"""
    print("\n" + "=" * 60)
    print("TRAINING LINEAR REGRESSION")
    print("=" * 60)
    
    start_time = time.time()
    
    # Train model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_test = lr_model.predict(X_test)
    
    training_time = time.time() - start_time
    
    # Evaluasi
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results = {
        'model': lr_model,
        'training_time': training_time,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'predictions': y_pred_test
    }
    
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    return results

def train_mlp_backpropagation(X_train, X_test, y_train, y_test):
    """Train MLP dengan backpropagation"""
    print("\n" + "=" * 60)
    print("TRAINING MLP BACKPROPAGATION")
    print("=" * 60)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # MLP dengan arsitektur optimal
    start_time = time.time()
    
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 50),  # Arsitektur sedang
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,  # Reduced iterations for faster training
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    
    mlp_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_test = mlp_model.predict(X_test_scaled)
    
    training_time = time.time() - start_time
    
    # Evaluasi
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results = {
        'model': mlp_model,
        'scaler': scaler,
        'training_time': training_time,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'predictions': y_pred_test,
        'n_iterations': mlp_model.n_iter_
    }
    
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Iterations: {mlp_model.n_iter_}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    return results

def compare_and_visualize(lr_results, mlp_results, y_test):
    """Bandingkan model dan buat visualisasi"""
    print("\n" + "=" * 80)
    print("PERBANDINGAN MODEL")
    print("=" * 80)
    
    # Comparison table
    comparison_data = {
        'Metric': ['Training Time (s)', 'Test R²', 'Test RMSE', 'Test MAE'],
        'Linear Regression': [
            f"{lr_results['training_time']:.4f}",
            f"{lr_results['test_r2']:.4f}",
            f"{lr_results['test_rmse']:.2f}",
            f"{lr_results['test_mae']:.2f}"
        ],
        'MLP Backpropagation': [
            f"{mlp_results['training_time']:.4f}",
            f"{mlp_results['test_r2']:.4f}",
            f"{mlp_results['test_rmse']:.2f}",
            f"{mlp_results['test_mae']:.2f}"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Determine best model
    if lr_results['test_rmse'] < mlp_results['test_rmse']:
        best_model = "Linear Regression"
        best_rmse = lr_results['test_rmse']
        best_r2 = lr_results['test_r2']
    else:
        best_model = "MLP Backpropagation"
        best_rmse = mlp_results['test_rmse']
        best_r2 = mlp_results['test_r2']
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Best RMSE: {best_rmse:.2f}")
    print(f"   Best R²: {best_r2:.4f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison: Linear Regression vs MLP Backpropagation', 
                 fontsize=14, fontweight='bold')
    
    # 1. Performance Metrics
    metrics = ['R²', 'RMSE (×1000)', 'MAE (×1000)', 'Time (s)']
    lr_values = [lr_results['test_r2'], 
                lr_results['test_rmse']/1000, 
                lr_results['test_mae']/1000, 
                lr_results['training_time']]
    mlp_values = [mlp_results['test_r2'], 
                 mlp_results['test_rmse']/1000, 
                 mlp_results['test_mae']/1000, 
                 mlp_results['training_time']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, lr_values, width, label='Linear Regression', color='skyblue')
    ax1.bar(x + width/2, mlp_values, width, label='MLP Backpropagation', color='lightcoral')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    
    # 2. Prediction vs Actual - Linear Regression
    ax2 = axes[0, 1]
    ax2.scatter(y_test, lr_results['predictions'], alpha=0.6, color='blue')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Linear Regression: Prediction vs Actual')
    
    # 3. Prediction vs Actual - MLP
    ax3 = axes[1, 0]
    ax3.scatter(y_test, mlp_results['predictions'], alpha=0.6, color='red')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title('MLP Backpropagation: Prediction vs Actual')
    
    # 4. Residuals Comparison
    ax4 = axes[1, 1]
    lr_residuals = y_test - lr_results['predictions']
    mlp_residuals = y_test - mlp_results['predictions']
    
    ax4.hist(lr_residuals, bins=50, alpha=0.7, label='Linear Regression', color='skyblue')
    ax4.hist(mlp_residuals, bins=50, alpha=0.7, label='MLP Backpropagation', color='lightcoral')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residuals Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(r'c:\deep learning\model_comparison_fast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Visualizations saved as 'model_comparison_fast.png'")
    
    return best_model, best_rmse, best_r2

def main():
    """Main function"""
    print("🚀 Model Comparison: Linear Regression vs MLP Backpropagation (Fast Version)")
    print("=" * 80)
    
    try:
        # Load data
        data_path = r"c:\deep learning\housing1.csv"
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
        
        # Train models
        lr_results = train_linear_regression(X_train, X_test, y_train, y_test)
        mlp_results = train_mlp_backpropagation(X_train, X_test, y_train, y_test)
        
        # Compare and visualize
        best_model, best_rmse, best_r2 = compare_and_visualize(lr_results, mlp_results, y_test)
        
        print("\n" + "=" * 80)
        print("🎉 COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Summary
        print("\nSUMMARY:")
        print(f"✓ Linear Regression - RMSE: {lr_results['test_rmse']:.2f}, R²: {lr_results['test_r2']:.4f}")
        print(f"✓ MLP Backpropagation - RMSE: {mlp_results['test_rmse']:.2f}, R²: {mlp_results['test_r2']:.4f}")
        print(f"🏆 Winner: {best_model}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()