import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
import torch.nn as nn

from torchviz import make_dot

#x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
#y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

data_url = r"C:\deep learning\housing1.csv"
raw_df = pd.read_csv(data_url)

# Handle missing values
raw_df = raw_df.dropna()

X = raw_df.drop(['median_house_value'], axis=1).select_dtypes(include=[np.number]).values
y = raw_df['median_house_value'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

model = nn.Sequential(
    nn.Linear(len(X_train[0]), 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

print(model)



X_train_ten = torch.tensor(X_train, dtype=torch.float32)
y_train_ten = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_ten = torch.tensor(X_test, dtype=torch.float32)
y_test_ten = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


# pred_y = model(X_train_ten)
# dot = make_dot(pred_y, params=dict(model.named_parameters()))
# dot.render("simple_nn_graph", format="png", view=True) 

# criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# loss function and optimizer
criterion = nn.MSELoss()  # mean square error
#optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(5000):
    pred_y = model(X_train_ten)
    # Compute and print loss
    loss = criterion(pred_y, y_train_ten)
    # Zero gradients, perform a backward pass, 
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        # Calculate R² score (accuracy for regression)
        with torch.no_grad():
            y_pred_train = model(X_train_ten)
            ss_res = torch.sum((y_train_ten - y_pred_train) ** 2)
            ss_tot = torch.sum((y_train_ten - torch.mean(y_train_ten)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
        print('epoch {}, loss {:.6f}, R² score: {:.4f}'.format(epoch, loss.item(), r2_score.item()))
    if (loss.item()<=0.1):
        break


model.eval()
y_pred_scaled = model(X_test_ten).detach().numpy()
# Denormalize predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled)
# mse = criterion(y_pred, y_test_ten)
# mse = float(mse)

# y_pred = reg.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_pred))

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)


if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use the GPU.")
else:
    print("CUDA is not available. PyTorch will use the CPU.")

# new_var = Variable(torch.Tensor([[4.0]]))
# pred_y = our_model(new_var)
# print("predict (after training)", 4, our_model(new_var).item())