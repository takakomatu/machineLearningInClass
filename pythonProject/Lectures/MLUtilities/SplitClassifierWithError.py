import numpy as np

class SplitClassifier:

    def __init__(self, X, y):
        self.data = np.array(X)
        self.labels = np.array(y)
        self.size = len(y)
 
        # Find and order possible label categories
        self.classes = sorted(set(self.labels))

        # Initialize the Training Accuracy to 0
        self.accuracy = 0
        
        # Iterate over each axis/feature
        # range(self.data.shape[1]) gives us number of column
        for i in range(self.data.shape[1]):

            # Obtain sorted list of feature values                        
            col_values = self.data[:,i].copy()
            col_values.sort()
           
            # Iterate over each observation
            for j in range(self.size):
                
                # Select values below the current observation
                sel = self.data[:,i] <= col_values[j]
                #sel = self.data.iloc[:,i].values <= col_values[j]
                
                # Determine the number correctly classified, assuming
                # that the lower class is class[0]
                n_correct = (np.sum(self.labels[sel] == self.classes[0]) + 
                             np.sum(self.labels[~sel] == self.classes[1]))
                #n_correct = np.sum(self.labels[sel] == self.classes[0])
                n_correct += self.labels[~sel]==self.classes[0]
                
                #Determine the accuracy of the current cut
                temp_acc=n_correct / self.size
                cur_acc=max(temp_acc, 1-temp_acc)
                
                #If new cut is an improvement, update attributes
                if cur_acc >= self.accuracy:
                    self.accuracy = cur_acc
                    self.axis = i
                    
                    if(j==len(col_values)-1) :
                        self.threshold = col_values[j]
                    else:
                        self.threshold=0.5*(col_values[j]+col_values[j+1])
                    if cur_acc==temp_acc:
                        self.lower_class=self.classes[0]
                        self.upper_class=self.classes[1]
                    else:
                        self.lower_class=self.classes[1]
                        self.upper_class=self.classes[0]
#//////////////////     
x=np.random.uniform(0,1,40).reshape(20,2)

y=[0]*10 + [1]*10# 10 copies of label 

split = SplitClassifier(x,y)
print(split)
print(split.axis)
print(split.threshold)
print(split.accuracy)