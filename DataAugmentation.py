import numpy as np
import pandas as pd


class Reverter():
    def __init__(self, random_state=None):
        self.random_state = random_state
        
    def fit(self, X, y):
        return self
    
    def transform(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_aug, y_aug = X.copy(), y.copy()
        X_aug.iloc[:, 3:16] = -X_aug.iloc[:, 3:16]
        X_aug.iloc[:, 5] = -X_aug.iloc[:, 5]
        X_aug['sequence'] += X['sequence'].max()+1
        X = pd.concat([X, X_aug], axis=0, ignore_index=True)
        y = np.concatenate([y, y_aug], axis=0)
        return X, y


class MultPerturb():
    def __init__(self, random_state=None):
        self.random_state = random_state
        
    def fit(self, X, y):
        return self
    
    def transform(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_aug, y_aug = X.copy(), y.copy()
        X_aug.iloc[:, 3:16] *= (0.9 + 0.2*rng.random(X_aug.iloc[:, 3:16].shape))
        X_aug['sequence'] += X['sequence'].max()+1
        X = pd.concat([X, X_aug], axis=0, ignore_index=True)
        y = np.concatenate([y, y_aug], axis=0)
        return X, y