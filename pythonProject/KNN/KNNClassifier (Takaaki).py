# Takaaki Komatsu
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
    
    # testing code 1
np.random.seed(1548)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X, y = make_classification(n_samples=2000, n_features=10, n_informative=4,
 n_clusters_per_class=1,class_sep=0.5,n_classes=6 )
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout,
 test_size=0.5)
rng = range(1,100)
val_acc = []
for K in rng:
 temp_mod = KNNClassifier(X_train, y_train, K)
 val_acc.append(temp_mod.score(X_val,y_val))

plt.close()
plt.plot(rng, val_acc)
plt.show()

knn_mod = KNNClassifier(X_train, y_train, 10)
print("Training Accuracy:", knn_mod.score(X_train, y_train))
print("Testing Accuracy: ", knn_mod.score(X_test, y_test))
print(knn_mod.predict(X_test[:20,:]))
print(y_test[:20])    
    
#    # testing code 2
#from ClassificationPlotter import plot_regions
#np.random.seed(1204)
#X = np.random.uniform(0,10,40).reshape(20,2)
#y = np.random.choice(['a','b','c','d'],20)
#knn_mod_3 = KNNClassifier(X,y,3)
#print(knn_mod_3.score(X,y))
#print(knn_mod_3.predict(X))
#plot_regions(knn_mod_3, X, y, 500)
#knn_mod_4 = KNNClassifier(X,y,4)
#print(knn_mod_4.score(X,y))
#print(knn_mod_4.predict(X))
#plot_regions(knn_mod_4, X, y, 500)