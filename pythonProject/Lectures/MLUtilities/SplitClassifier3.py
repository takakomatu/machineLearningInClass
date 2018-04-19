# Takaaki Komatsu

import numpy as np

class SplitClassifier:#class

    def __init__(self, X, y):#constructor
        self.data = np.array(X)
        self.labels = np.array(y)
        self.size = len(y) # equivalent to self.data.shape[0], 
#       self.size is number of observations = 20, ten 0s and ten 1s
 
        # Find and order possible label categories
        self.classes = sorted(set(self.labels)) #[0,1]

        # Initialize the Training Accuracy to 0
        self.accuracy = 0

        # Iterate over each axis/feature
        # range(self.data.shape[1]) gives us number of column
        for i in range(self.data.shape[1]): #self.data.shape[1] =2

            # Obtain sorted list of feature values                        
            col_values = self.data[:,i].copy()
            col_values.sort()
            # Iterate over each observation
            for j in range(self.size): #self.size =20, ten 0s and ten 1s
                
                # Select values below the current observation
                sel = self.data[:,i] <= col_values[j] # col_values has numbers from low to high
                #sel = self.data.iloc[:,i].values <= col_values[j]
#self.labels[sel]=['a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b' 'b' 'b' 'b' 'b' 'b']
                # Determine the number correctly classified, assuming
                # that the lower class is class[0]
                n_correct = (np.sum(self.labels[sel] == self.classes[0]) + 
                             np.sum(self.labels[~sel] == self.classes[1]))
                #                bool_array = np.array([True, True, False, True, False])
                #                my_array = np.array([1,2,3,4,5])
                #
                #                sub_array = my_array[bool_array]
                #                print(sub_array) these return [1 2 4]
                                
                #Determine the accuracy of the current cut
                temp_acc=n_correct / self.size # 11/20
#                print("sadfdsaf",temp_acc)=print("sdflfdffsf" + str(temp_acc))
                cur_acc=max(temp_acc, 1-temp_acc)
                #If new cut is an improvement, update attributes
                if cur_acc >= self.accuracy:
                    self.accuracy = cur_acc
                    self.feature = i# decide which axis we should draw a line along
                    
                    if(j==len(col_values)-1) : # if j==19, j is the last one
                        self.threshold = col_values[j]
                    else:#these only happens if we changed the accuracy.
                        self.threshold=0.5*(col_values[j]+col_values[j+1])
                    if cur_acc==temp_acc: # we dont reverse the labels
                        self.lower_class=self.classes[0]
                        self.upper_class=self.classes[1]
                    else: # we reverse the labels
                        self.lower_class=self.classes[1]
                        self.upper_class=self.classes[0]

    def predict(self, X):
        # Create inner function to classify an individual observation
        #classifyObject, row is from feature table
        def classify_obs(row):
            if row[self.feature] <= self.threshold:
                return self.lower_class# return either one label such as a, b, 0, 1
            else:
                return self.upper_class# return either one label such as a, b, 0, 1     

        # Convert X to a NumPy array
        X = np.array(X)
        
        # Apply classify_obs to rows of X
        return np.apply_along_axis(classify_obs, 1, X)#does classify_obs take row of X as an argument??
    #1 means were applying classify_obs to every row, 0 would mean to every column
#    np.apply_along_axis takes and apply classify_obs that to every row such as [0.94233555  0.72765208] 
    def score(self, X, y): # for test data
        
        X = np.array(X)
        y = np.array(y)
        
        predictions = self.predict(X)
        num_correct = np.sum(predictions == y)
        
        return num_correct / len(y)
        
    def summary(self):
        print('+----------------------------+')
        print('|  Split Classifier Summary  |')
        print('+----------------------------+')
        print('Number of training observations:', self.size)
        print('Axis/Feature split:', self.feature)
        print('Threshold value:', self.threshold)
        print('Predicted class below threshold:', self.lower_class)
        print('Predicted class above threshold:', self.upper_class)
        print('Training accuracy:', self.accuracy, '\n')
        
#//////////////////     
x=[[ 1,  1],
 [ 2,  2],
 [ 3 , 3],
 [ 4 , 4],
 [ 5 , 5],
 [ 6 , 6],
 [ 7 , 7],
 [ 8  ,8],
 [ 9  ,9],
 [ 10 , 10],
 [ 11 , 11],
 [ 12  , 12],
 [ 13 , 13],
 [ 14 , 14 ],
 [ 15 , 15],
 [ 16 ,  16],
 [ 17 , 17],
 [ 18  ,18 ],
 [ 19 , 19],
 [ 20  ,20]]
print(x)
y=["0"]*10 + ["1"]*10# 10 copies of label, outputs

split = SplitClassifier(x,y)

x2=[[ 2,  2],
 [ 4, 4],
 [ 6, 6],
 [ 8 , 8],
 [ 10 , 10],
 [ 12, 12],
 [ 14, 14],
 [ 16,16],
 [ 18 ,18],
 [ 20, 20],
 [ 22 ,22],
 [ 24  , 24],
 [ 26, 26],
 [ 28, 28 ],
 [ 30 ,30],
 [ 32,  32],
 [ 34,34],
 [ 36,36 ],
 [ 38 ,38],
 [ 40,40]]
split2=SplitClassifier(x2,y)
split2.summary() #Training accuracy: 1.0 ????
print("What was predicted was: ",split2.predict(x)) 
#If we are using the same X, or training data, the machine will have 100% accuracy??

print(split2.score(x,y))