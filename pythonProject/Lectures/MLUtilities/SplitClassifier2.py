import numpy as np

class SplitClassifier:

    def __init__(self, X, y):
        self.data = np.array(X)
        self.labels = np.array(y)
        self.size = len(y) # equivalent to self.data.shape[0], 
#        self.size is number of observations = 20, ten 0s and ten 1s
 
        # Find and order possible label categories
        self.classes = sorted(set(self.labels)) #[0,1]

        # Initialize the Training Accuracy to 0
        self.accuracy = 0
        
        # Iterate over each axis/feature
        # range(self.data.shape[1]) gives us number of column
        for i in range(self.data.shape[1]):

            # Obtain sorted list of feature values                        
            col_values = self.data[:,i].copy()
            col_values.sort()
           
            # Iterate over each observation
            for j in range(self.size): #self.size =20, ten 0s and ten 1s
                
                # Select values below the current observation
                sel = self.data[:,i] <= col_values[j]
                #sel = self.data.iloc[:,i].values <= col_values[j]
                
                # Determine the number correctly classified, assuming
                # that the lower class is class[0]
                n_correct = (np.sum(self.labels[sel] == self.classes[0]) + 
                             np.sum(self.labels[~sel] == self.classes[1]))
                #n_correct = np.sum(self.labels[sel] == self.classes[0])
#                n_correct += self.labels[~sel]==self.classes[0]
                
                #Determine the accuracy of the current cut
                temp_acc=n_correct / self.size # 11/20
#                print("sadfdsaf",temp_acc)=print("sdflfdffsf" + str(temp_acc))
                cur_acc=max(temp_acc, 1-temp_acc)
                #If new cut is an improvement, update attributes
                if cur_acc >= self.accuracy:
                    self.accuracy = cur_acc
                    self.feature = i 
                    
                    if(j==len(col_values)-1) : # if j==19, j is the last one
                        self.threshold = col_values[j]
                    else:
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
        def classify_obs(row): #how does row probably look like?
            if row[self.feature] <= self.threshold:
                return self.lower_class#could return both self.classes[0] and self.classes[1]???
            else:
                return self.upper_class#could return both self.classes[0] and self.classes[1]???              

        # Convert X to a NumPy array
        X = np.array(X)
        
        # Apply classify_obs to rows of X
        return np.apply_along_axis(classify_obs, 1, X)#doesnt classify_obs take X as an argument
    #1 means were applying classify_obs to every row, 0 would mean to every column
#    np.apply_along_axis takes and apply classify_obs that to every row such as [ 0.94233555  0.72765208] 
        #apply_along_axis(func1d, axis, arr, *args, **kwargs) 
        #Apply a function to 1-D slices along the given axis.    
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
x=np.random.uniform(0,10,40).reshape(20,2)
print(x)
print(x.shape[1])
y=["a"]*10 + ["b"]*10# 10 copies of label 

split = SplitClassifier(x,y)

split.summary()
print(split.predict(x))


print(split.score(x,y))