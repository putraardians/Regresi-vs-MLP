from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
import time
from sklearn.impute import SimpleImputer

data_url = "C:\deep learning\housing1.csv"
# use the actual CSV file (comma-separated) and avoid backslash escapes in the string
data_url = r"D:\KULIAH SMT 7\Deep learning tugas\housing.csv"
# the file has a header row and is comma-separated, so let pandas parse it normally
raw_df = pd.read_csv(data_url, sep=",", header=0)

# Use pandas to select numeric features and one-hot encode the categorical column
# target column in this dataset is 'median_house_value'
if 'median_house_value' not in raw_df.columns:
	raise RuntimeError("Expected 'median_house_value' column in CSV")

# select numeric feature columns (exclude target)
all_numeric_cols = list(raw_df.select_dtypes(include=[np.number]).columns.drop('median_house_value'))

# choose exactly 5 input features for the model
preferred = ['median_income', 'total_rooms', 'total_bedrooms', 'population', 'households']
numeric_cols = [c for c in preferred if c in all_numeric_cols]
if len(numeric_cols) < 5:
	for c in all_numeric_cols:
		if c not in numeric_cols:
			numeric_cols.append(c)
		if len(numeric_cols) == 5:
			break

# Build X using only the selected 5 numeric features (no one-hot to keep input dim=5)
X_df = raw_df[numeric_cols].reset_index(drop=True)

y = raw_df['median_house_value'].values
X = X_df.values

# fill missing values (NaN) using mean imputation for numeric features
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.metrics import mean_absolute_error,mean_squared_error

	# Run multiple experiments with different hidden-layer architectures
experiments = [
	("Percobaan 1", (16,)),
	("Percobaan 2", (16, 16)),
	("Percobaan 3", (16, 16, 32)),
]

results = []
for name, arch in experiments:
	print("\n=== {} : hidden_layer_sizes={} ===".format(name, arch))
	mlp = MLPRegressor(hidden_layer_sizes=arch, activation='relu', solver='adam', random_state=1,
					   max_iter=2000, early_stopping=True, n_iter_no_change=20)

	# capture convergence warnings during fit
	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter('always')
		t0 = time.time()
		mlp.fit(X_train, y_train)
		t1 = time.time()

	# determine whether training hit the max_iter (likely not converged)
	hit_max_iter = getattr(mlp, 'n_iter_', 0) >= mlp.max_iter

	# check if any ConvergenceWarning was raised
	conv_warnings = [ww for ww in w if issubclass(ww.category, ConvergenceWarning)]

	y_pred = mlp.predict(X_test)

	mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
	mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
	rmse = np.sqrt(mse)

	print(f"time (s): {t1-t0:.2f}")
	print(f"n_iter_: {getattr(mlp,'n_iter_', None)}; hit_max_iter={hit_max_iter}")
	print(f"convergence_warnings: {len(conv_warnings)}")
	print("MAE:", mae)
	print("MSE:", mse)
	print("RMSE:", rmse)

	results.append({
		'name': name,
		'arch': arch,
		'time_s': t1 - t0,
		'n_iter': getattr(mlp, 'n_iter_', None),
		'hit_max_iter': hit_max_iter,
		'n_conv_warnings': len(conv_warnings),
		'mae': mae,
		'mse': mse,
		'rmse': rmse,
	})

print('\n=== Summary ===')
for r in results:
	print(r)

# regression coefficients
# print('Coefficients: ', reg.coef_)

# # variance score: 1 means perfect prediction
# print('Variance score: {}'.format(reg.score(X_test, y_test)))

# # regression coefficients
# print('Coefficients: ', reg.coef_)

# # variance score: 1 means perfect prediction
# print('Variance score: {}'.format(reg.score(X_test, y_test)))