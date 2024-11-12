import pandas as pd
from model.QBVM import QBV
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

data = datasets.load_wine(as_frame=True)['data']
mms = MinMaxScaler(feature_range=(0.00001, 0.99999))
data = pd.DataFrame(mms.fit_transform(data), columns=data.columns)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = QBV(init_dist='Cauchy', perm_count=2, train_frac=0.8, seed=42)
model.fit(X, y)

model.save_model('_tmp_model/qb_vine_model.pkl')
print("Model saved as _tmp_model/qb_vine_model.pkl")
model = QBV().load_model('_tmp_model/qb_vine_model.pkl')

test_dens, cop_dens_xy = model.predict(data)
print("Test Densities:", test_dens)
print("Copula Log-Likelihood:", cop_dens_xy)