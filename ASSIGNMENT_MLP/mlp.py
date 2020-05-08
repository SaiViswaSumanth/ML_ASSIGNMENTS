

#import necessarey libraries
import numpy as mp
import pandas as da
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score

#dataset features determine if the ground is rock or mine(binary format)
df = pd.read_csv('sonarnew.csv')
X = df.drop(['Class'], axis='columns')
y = df.Class




#splitting of data to train and test 80 to 20 percent respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)



#arrays to matrices for easier numerical computations
X_train = mp.asmatrix(pd.DataFrame(X_train), dtype = 'float64')
y_train = mp.asmatrix(pd.DataFrame(y_train), dtype = 'float64')
X_test = mp.asmatrix(pd.DataFrame(X_test), dtype = 'float64')
y_test = mp.array(pd.DataFrame(y_test), dtype = 'float64')



training_inputs = X_train
labels = y_train



#preceptron class for training and testing
class Neuron(object):

    def __init__(self, no_of_feature, max_epoch=100, learning_rate=0.01):
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.weight = mp.zeros(no_of_feature + 1)
        print(self.weight.shape)
        
#prediction fucntion  

    def prediction(self, inputs):
        summ = mp.dot(inputs, self.weight[1:]) + self.weight[0]
        d=self.step(summ)
        return d
        
#activation fucntion is a step function 

    def step(self,summ):
        if summ > 0:
            activation= 1
        else:
            activation = 0
        return activation
    
#training fucntion

    def train(self, training_inputs, labels):
        for i in range(self.max_epoch):
            for inputs, label in zip(training_inputs, labels):
                predict = self.prediction(inputs)
                self.weight[1:] += mp.array(self.learning_rate * (label[0] - predict) * mp.array(inputs)[0])[0]
                self.weight[0] += mp.array(self.learning_rate * (label[0] - predict))[0]
            print("Epoch no\t"+str(i+1),"\tLearning rate\t"+str(self.learning_rate),"\nNew weight"+str(self.weight)) 
            





#passing arguments as no of features
perceptron = Neuron(60)

#start train
perceptron.train(training_inputs, labels)


#saving of predicted output to a variable
a=mp.array([])
x=a

for i in range(X_test.shape[0]):
    output=perceptron.prediction(X_test[i])
    x=mp.append(x,[output],axis=0)
    
print("predicted test output\n",x)

#printing of accuracy metrics
a=confusion_matrix(y_test,x)
print("Confusion Matrix\n"+str(confusion_matrix(y_test, x)))
tn=a[0][0]
fp=a[0][1]
fn=a[1][0]
tp=a[1][1]


print("Accuracy _score\t"+str(accuracy_score(y_test, x)))
a=(tp+tn)/(tp+tn+fp+fn)
print("Accuracy using formula",a)
print("Recall Score\t"+str(recall_score(y_test, x, average='binary')))
r=tp/(tp+fn)
print("Recall using formula",r)
print("Precision score\t"+str(precision_score(y_test, x, average='binary')))
p=tp/(tp+fp)
print("precision using formula",p)



#visualization of confusion matrix
labels = [0,1]
cm = confusion_matrix(y_test, x,labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier\n')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


