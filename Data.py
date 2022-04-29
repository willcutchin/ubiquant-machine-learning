import pandas as pd
import numpy as np
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

batch_size = 5

class DataClean:

    def __init__(self, path):
        self.path = path

    def get_data(self):
        df = pd.read_csv(self.path)
        return df

    def manipulate(self, data):
        # print(data.shape)
        testdata = data.iloc[:, :-5]
        # print(testData.shape)
        return testdata

    def get_targets(self, data):
        # targets = data[:, :-1]
        data = data.iloc[:, 150]
        return data

    def get_train(self, data):
        x = data.iloc[:, :-1]
        x = x.astype(float)
        # print(x.dtypes)
        print(len(data.columns))
        return x

    def to_numpy(self, x, y):
        newX = x.to_numpy()
        newY = y.to_numpy()
        return newX, newY

    def train_test(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
        x_train, x_test, y_train, y_test

        return x_train, x_test, y_train, y_test

    def data_to_tensors(self, xtrain, xtest, ytrain, ytest, b_size):

        train_features = torch.Tensor(xtrain)
        train_targets = torch.Tensor(ytrain)

        test_features = torch.Tensor(xtest)
        test_targets = torch.Tensor(ytest)

        train = TensorDataset(train_features, train_targets)
        # val = TensorDataset(val_features, val_targets)
        test = TensorDataset(test_features, test_targets)

        train_loader = DataLoader(train, batch_size=b_size, shuffle=False)
        # val_loader = DataLoader(val, batch_size=b_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=b_size, shuffle=False)
        test_loader_one = DataLoader(test, batch_size=1, shuffle=False)

        return train_loader, test_loader, test_loader_one
