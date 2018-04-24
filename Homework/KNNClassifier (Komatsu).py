# Pat Deken
# HWP04

import numpy as np
import pandas as pd

class KNNClassifier:
    
    def __init__(self, X, y, K):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_observations = len(self.y)
        self.classes = np.unique(self.y)
        self.K = int(K)
        
    def predict(self, X):
        X = np.array(X)
        predictions = []
        for r in X:
            row=r[:]
            distances = np.sqrt(np.sum((row-self.X)**2, axis=1))
            idx = np.argsort(distances)[:self.K]
            knn_labels = self.y[idx]
            knn_distances = distances[idx]
            best_dist = 0
            best_class = 0
            best_count = 0
            for clss in self.classes:
                temp_count = np.sum(clss==knn_labels)
                temp_distances = np.sum(knn_distances[clss==knn_labels])
                if temp_count > best_count:
                    best_dist = temp_distances
                    best_class = clss
                    best_count = temp_count
                elif (temp_count == best_count) & (temp_distances < best_dist):
                    best_dist = temp_distances
                    best_class = clss
                    best_count = temp_count
            predictions.append(best_class)
        return np.array(predictions)
    
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        true_values = np.sum(self.predict(X)==y)
        return true_values / len(y)
    
    def confusion_matrix(self, X, y):
        X = np.array(X)
        y = np.array(y)
        list_o_lists = []
        for actual in self.classes:
            row = []
            for pred in self.classes:
                temp_entry = np.sum( (self.predict(X)==pred) & 
                                     (y==actual ))
                row.append(temp_entry)
            list_o_lists.append(row)
        cm = pd.DataFrame(list_o_lists)
        pred_str = []
        actual_str = []
        for clss in self.classes:
            pred_str.append('pred_'+ str(clss))
            actual_str.append('actual_' + str(clss))
        cm.columns = pred_str
        cm.index = actual_str
        return cm