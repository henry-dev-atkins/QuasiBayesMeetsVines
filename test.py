import pandas as pd
from model.QBVM import QBV
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = datasets.load_wine(as_frame=True)['data']

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = QBV(init_dist='Cauchy', perm_count=20, train_frac=0.8, seed=42)
model.fit(X, y, theta_iterations=2)

model.save_model('qb_vine_model')

model = QBV().load_model('qb_vine_model')
print(f"X shape: {X.shape}")

preds_df = model.predict(X)
preds = preds_df.squeeze()

error = preds - y
print(f"STD of preds is: {preds.std()}")
print(f"STD of y is: {y.std()}")
print(abs(error).mean())

plt.figure(figsize=(10, 6))
plt.scatter(y, preds, label='Predictions vs Actual', color='blue', alpha=0.5)
plt.title('Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()