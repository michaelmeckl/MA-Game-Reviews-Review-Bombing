import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import evaluate
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ReviewNeuralNetwork(nn.Module):
    # based on https://andrew-muller.medium.com/video-game-review-analysis-3c7602184668
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.sequential = nn.Sequential(
            nn.Linear(500, 250),
            nn.ReLU(),
            self.dropout,
            nn.Linear(250, 125),
            nn.ReLU(),
            self.dropout,
            nn.Linear(125, 250),
            nn.ReLU(),
            self.dropout,
            nn.Linear(250, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.sequential(x)
        return logits


class MLP(nn.Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, 10)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(10, 8)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(8, 1)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Sigmoid()

    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X
