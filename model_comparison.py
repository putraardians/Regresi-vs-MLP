"""
Model Comparison: Linear Regression vs MLP Backpropagation
Prediksi Harga Rumah menggunakan California Housing Dataset

Author: Deep Learning Project
Date: October 25, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ModelComparison:
    """
    Kelas untuk membandingkan performa Linear Regression vs MLP Backpropagation
    """
    
    def __init__(self, data_path):
        """
        Initialize dengan path ke dataset
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler_X = None
        self.scaler_y = None
        self.results = {}
        
    def load_and_preprocess_data(self, test_size=0.2, random_state=42):
        """
        Load dataset dan preprocessing yang konsisten untuk kedua model
        """
        print("="*60)
        print("LOADING DAN PREPROCESSING DATA")
        print("="*60)
        
        # Load dataset
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values sebelum preprocessing:\n{self.df.isnull().sum()}")
        
        # Handle missing values dengan mean imputation
        imputer = SimpleImputer(strategy='mean')
        
        # Pilih fitur numerik (exclude target dan categorical features)
        numeric_features = ['longitude', 'latitude', 'housing_median_age', 
                          'total_rooms', 'total_bedrooms', 'population', 
                          'households', 'median_income']
        
        # Pastikan semua fitur ada di dataset
        available_features = [col for col in numeric_features if col in self.df.columns]
        print(f"Available numeric features: {available_features}")
        
        # Prepare X and y
        X = self.df[available_features].copy()
        y = self.df['median_house_value'].copy()
        
        # Handle missing values
        X = pd.DataFrame(imputer.fit_transform(X), columns=available_features)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set shape: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Test set shape: X={self.X_test.shape}, y={self.y_test.shape}")
        
        # Feature scaling (akan digunakan untuk MLP, opsional untuk Linear Regression)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        print("✓ Data berhasil dimuat dan dipreprocess")
        
    def train_linear_regression(self):
        """
        Train Linear Regression model
        """
        print("\n" + "="*60)
        print("TRAINING LINEAR REGRESSION")
        print("="*60)
        
        start_time = time.time()
        
        # Linear Regression tidak memerlukan scaling, tapi kita akan menggunakan data original
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = lr_model.predict(self.X_train)
        y_pred_test = lr_model.predict(self.X_test)
        
        training_time = time.time() - start_time
        
        # Evaluasi
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(self.y_train, y_pred_train)
        
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        # Simpan hasil
        self.results['Linear_Regression'] = {
            'model': lr_model,
            'training_time': training_time,
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test,
            'train_mae': train_mae,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'coefficients': lr_model.coef_,
            'intercept': lr_model.intercept_
        }
        
        print(f"Training time: {training_time:.4f} seconds")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print("✓ Linear Regression training selesai")
        
    def train_mlp_backpropagation(self):
        """
        Train MLP dengan backpropagation (berbagai arsitektur)
        """
        print("\n" + "="*60)
        print("TRAINING MLP BACKPROPAGATION")
        print("="*60)
        
        # Scale data untuk MLP
        X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        X_test_scaled = self.scaler_X.transform(self.X_test)
        
        # Berbagai arsitektur MLP untuk dibandingkan
        mlp_architectures = {
            'MLP_Small': (50,),
            'MLP_Medium': (100, 50),
            'MLP_Large': (100, 100, 50),
            'MLP_Deep': (150, 100, 50, 25)
        }
        
        best_mlp = None
        best_score = -np.inf
        
        for name, hidden_layers in mlp_architectures.items():
            print(f"\nTraining {name} - Architecture: {hidden_layers}")
            
            start_time = time.time()
            
            # Train MLP
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.0001,  # L2 regularization
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
            
            mlp.fit(X_train_scaled, self.y_train)
            
            training_time = time.time() - start_time
            
            # Predictions
            y_pred_train = mlp.predict(X_train_scaled)
            y_pred_test = mlp.predict(X_test_scaled)
            
            # Evaluasi
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(self.y_train, y_pred_train)
            
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            # Simpan hasil
            self.results[name] = {
                'model': mlp,
                'architecture': hidden_layers,
                'training_time': training_time,
                'n_iter': mlp.n_iter_,
                'train_predictions': y_pred_train,
                'test_predictions': y_pred_test,
                'train_mae': train_mae,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'test_mae': test_mae,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            }
            
            print(f"  Training time: {training_time:.4f}s")
            print(f"  Iterations: {mlp.n_iter_}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.2f}")
            
            # Track best model
            if test_r2 > best_score:
                best_score = test_r2
                best_mlp = name
        
        print(f"\n✓ Best MLP model: {best_mlp} (R² = {best_score:.4f})")
        
    def compare_models(self):
        """
        Bandingkan semua model dan tampilkan hasil
        """
        print("\n" + "="*80)
        print("PERBANDINGAN MODEL")
        print("="*80)
        
        # Buat comparison table
        comparison_data = []
        
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Training Time (s)': f"{result['training_time']:.4f}",
                'Train R²': f"{result['train_r2']:.4f}",
                'Test R²': f"{result['test_r2']:.4f}",
                'Test MAE': f"{result['test_mae']:.2f}",
                'Test RMSE': f"{result['test_rmse']:.2f}",
                'Overfitting': f"{result['train_r2'] - result['test_r2']:.4f}"
            })
        
        # Display table
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Find best model
        best_model = min(self.results.keys(), 
                        key=lambda x: self.results[x]['test_rmse'])
        print(f"\n🏆 Best Model (lowest RMSE): {best_model}")
        print(f"   Test RMSE: {self.results[best_model]['test_rmse']:.2f}")
        print(f"   Test R²: {self.results[best_model]['test_r2']:.4f}")
        
    def visualize_results(self):
        """
        Visualisasi hasil perbandingan model
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Setup plotting
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Comparison: Linear Regression vs MLP Backpropagation', 
                     fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        models = list(self.results.keys())
        test_r2_scores = [self.results[m]['test_r2'] for m in models]
        test_rmse_scores = [self.results[m]['test_rmse'] for m in models]
        training_times = [self.results[m]['training_time'] for m in models]
        overfitting = [self.results[m]['train_r2'] - self.results[m]['test_r2'] for m in models]
        
        # 1. R² Score Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, test_r2_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'plum'])
        ax1.set_title('Test R² Score Comparison')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(test_r2_scores):
            ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. RMSE Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, test_rmse_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'plum'])
        ax2.set_title('Test RMSE Comparison')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        for i, v in enumerate(test_rmse_scores):
            ax2.text(i, v + 1000, f'{v:.0f}', ha='center', va='bottom')
        
        # 3. Training Time Comparison
        ax3 = axes[0, 2]
        bars3 = ax3.bar(models, training_times, color=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'plum'])
        ax3.set_title('Training Time Comparison')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        for i, v in enumerate(training_times):
            ax3.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        # 4. Overfitting Analysis
        ax4 = axes[1, 0]
        colors = ['red' if x > 0.1 else 'green' if x < 0.05 else 'orange' for x in overfitting]
        bars4 = ax4.bar(models, overfitting, color=colors)
        ax4.set_title('Overfitting Analysis (Train R² - Test R²)')
        ax4.set_ylabel('Overfitting Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High Overfitting')
        ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate Overfitting')
        ax4.legend()
        for i, v in enumerate(overfitting):
            ax4.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom')
        
        # 5. Prediction vs Actual (Best Model)
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['test_rmse'])
        ax5 = axes[1, 1]
        y_pred = self.results[best_model]['test_predictions']
        ax5.scatter(self.y_test, y_pred, alpha=0.6, color='blue')
        ax5.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax5.set_xlabel('Actual Values')
        ax5.set_ylabel('Predicted Values')
        ax5.set_title(f'Prediction vs Actual ({best_model})')
        
        # 6. Residuals Plot (Best Model)
        ax6 = axes[1, 2]
        residuals = self.y_test - y_pred
        ax6.scatter(y_pred, residuals, alpha=0.6, color='purple')
        ax6.axhline(y=0, color='red', linestyle='--')
        ax6.set_xlabel('Predicted Values')
        ax6.set_ylabel('Residuals')
        ax6.set_title(f'Residuals Plot ({best_model})')
        
        plt.tight_layout()
        plt.savefig(r'c:\deep learning\model_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizations saved as 'model_comparison_results.png'")
        
    def save_models(self):
        """
        Simpan model terbaik
        """
        best_model_name = min(self.results.keys(), 
                             key=lambda x: self.results[x]['test_rmse'])
        best_model = self.results[best_model_name]['model']
        
        # Save model
        joblib.dump(best_model, r'c:\deep learning\best_model.pkl')
        joblib.dump(self.scaler_X, r'c:\deep learning\scaler_X.pkl')
        
        print(f"\n✓ Best model ({best_model_name}) saved as 'best_model.pkl'")
        print("✓ Scaler saved as 'scaler_X.pkl'")

def main():
    """
    Main function untuk menjalankan seluruh comparison
    """
    print("🚀 Starting Model Comparison: Linear Regression vs MLP Backpropagation")
    print("=" * 80)
    
    # Initialize comparison
    data_path = r"c:\deep learning\housing1.csv"
    comparison = ModelComparison(data_path)
    
    try:
        # Step 1: Load and preprocess data
        comparison.load_and_preprocess_data()
        
        # Step 2: Train Linear Regression
        comparison.train_linear_regression()
        
        # Step 3: Train MLP Backpropagation
        comparison.train_mlp_backpropagation()
        
        # Step 4: Compare models
        comparison.compare_models()
        
        # Step 5: Visualize results
        comparison.visualize_results()
        
        # Step 6: Save best model
        comparison.save_models()
        
        print("\n" + "="*80)
        print("🎉 MODEL COMPARISON COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during model comparison: {str(e)}")
        print("Please check the data path and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()