import pandas as pd
from model.QBVM import QBV
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

data = datasets.load_wine(as_frame=True)['data']
mms = MinMaxScaler()
data = pd.DataFrame(mms.fit_transform(data), columns=data.columns)[:100]

model = QBV(init_dist='Cauchy', perm_count=10, train_frac=0.8, seed=42)
model.fit(data)

test_data = pd.read_csv('datasets/wine.data')

test_dens, cop_dens_xy = model.predict(test_data)
print("Test Densities:", test_dens)
print("Copula Log-Likelihood:", cop_dens_xy)

model.save_model('qb_vine_model.pkl')
print("Model saved as qb_vine_model.pkl")