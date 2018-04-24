import numpy as np

class KNNClassifier:
    
    def __init__(self, X, y, K):
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = np.unique(y)
        self.K = K
        
    def predict(self, X):
        
        X = np.array(X)
        predictions = []
        
        for i in range(X.shape[0]):
            
            row = X[i,:]
            
            distances = np.sum((row - self.X)**2, axis=1)**0.5
            
            idx = np.argsort(distances)[:self.K]
            
            knn_labels = self.y[idx]
            knn_distances = distances[idx]
                        
            cur_dist = 0
            cur_class = 0
            cur_count = 0
            
            
            #for j in range(len(self.classes)):
            
            for label in self.classes:
                
                sel = (knn_labels == label)
                temp_count = np.sum(sel)
                temp_dist = np.sum(knn_distances[sel])
                
                if(temp_count > cur_count):
                    cur_dist = temp_dist
                    cur_class = label
                    cur_count = temp_count
                    
                if( (temp_count == cur_count) and (temp_dist < cur_dist )):
                    cur_dist = temp_dist
                    cur_class = label
                    cur_count = temp_count
                    
            predictions.append(cur_class)
                
        return np.array(predictions)
          
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.predict(X)
        
        return np.sum(y_pred == y) / len(y)
            
            
#import sklearn.datasets as skds
#from sklearn.neighbors import KNeighborsClassifier
#
#X, y = skds.make_blobs(n_samples=5000, n_features=4, centers=4, cluster_std=6)
#
#for i in range(1,30):
#    
#    print('K =', i)
#    knn = KNNClassifier(X, y, i)
#    print(knn.score(X,y))
#    
#    knn2 = KNeighborsClassifier(i)
#    knn2.fit(X,y)
#    print(knn2.score(X,y))
#    print()
#    
#from ClassificationPlotter import plot_regions
#
#np.random.seed(1204)
#X = np.random.uniform(0,10,40).reshape(20,2)
#y = np.random.choice(['a','b','c','d'],20)
#
#knn_mod_3 = KNNClassifier(X,y,3)
#print(knn_mod_3.score(X,y))
#print(knn_mod_3.predict(X))
#
#plot_regions(knn_mod_3, X, y, 500)
#
#knn_mod_4 = KNNClassifier(X,y,4)
#print(knn_mod_4.score(X,y))
#print(knn_mod_4.predict(X))
#
#plot_regions(knn_mod_4, X, y, 500)
#

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




#knn_mod_3 = KNNClassifier(X,y,3)
#print(knn_mod_3.score(X,y))
#print(knn_mod_3.predict(X))