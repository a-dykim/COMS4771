import sklearn.neural_network as sknn
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.io as sio
import csv

mat_contents = sio.loadmat(file_name='/Users/a_dykim/Google_drive/Columbia/Fall2018/Classes/ML/MSdata.mat')
test_x = mat_contents['testx']
train_x = mat_contents['trainx']
train_y = mat_contents['trainy']

scaler = StandardScaler()
scaler.fit(train_x)
x_train_scaled = scaler.transform(train_x)
x_test_scaled = scaler.transform(test_x)
nnr = sknn.MLPRegressor()
nnr.fit(x_train_scaled, train_y)
nn_predict_y_adam= nnr.predict(x_test_scaled)
