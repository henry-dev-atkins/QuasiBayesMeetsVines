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
##model.fit(X, y)
# model.save_model('qb_vine_model')


model = QBV().load_model('qb_vine_model')

preds = model.predict(data)
print("Test Densities:", preds['marginal_densities'])
print("joint_densities:", preds['joint_densities'])
print("Scores:", preds['final_scores'])