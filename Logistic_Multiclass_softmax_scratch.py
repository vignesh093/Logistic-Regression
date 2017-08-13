import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class Multiclass_Logistic(object):

    def __init__(self,n_iter,lear_rate,reg_rate):
        self.n_iter=n_iter
        self.lear_rate=lear_rate
        self.reg_rate=reg_rate

    def split_test_train(self,indata,split_ratio):
        train_size=int(split_ratio*len(indata))
        train_data=indata.iloc[0:train_size,1:10].values #take column 1 to 9(except C1)
        train_label=indata.iloc[0:train_size,10].values #take labels
        test_data=indata.iloc[(train_size+1):len(indata),1:10].values
        test_label = indata.iloc[(train_size + 1):len(indata),10].values
        #No standardization here
        for j in range((train_data.shape[1])):
           train_data[:, j] = (train_data[:, j] - train_data[:, j].mean())/train_data[:, j].std()

        for j in range((test_data.shape[1])):
            test_data[:, j] = (test_data[:, j] - test_data[:, j].mean())/test_data[:, j].std()
        return train_data,train_label,test_data,test_label

    def softmax(self,x,w,b):

        linear_equ=np.dot(x,w)#(143,9)(6,9)
        linear_com_equ=np.add(linear_equ,b)
        num=np.exp(linear_equ)

        denom=np.sum(num,axis=1)

        softmax=num.T/denom
        return softmax.T

    def gradient(self,x,y,add_intercept=True):
        if add_intercept:
            intercept = np.ones((x.shape[0], 1))
            x = np.hstack((intercept, x))

        combined_ll=[]
        w=np.zeros((x.shape[1],y.shape[1])) #(number of features,number of class) 9,6
        print(w.shape)
        b=np.zeros((y.shape[1])) #(number of class) 9,
        for step in range(self.n_iter):
            #p_y_given_x = self.softmax(x,w,b)
            p_y_given_x = self.softmax(x, w,b)
            d_y = y - p_y_given_x
            #w += self.lear_rate * np.dot(x.T, d_y) - self.lear_rate * self.reg_rate * w
            b -= self.lear_rate * np.mean(d_y, axis=0)

            grad = (-1 / x.shape[0]) * np.dot(x.T, (y - p_y_given_x)) + self.reg_rate * w
            w = w - (self.lear_rate * grad)
            if step % 100 == 0:
                #ll=self.negative_log_likelihood(x, y, w, b)
                loss = (-1 / x.shape[0]) * np.sum(y * np.log(p_y_given_x)) + (self.reg_rate / 2) * np.sum(w * w)  # We then find the loss of the probabilities
                combined_ll.append(loss)
        plot_cost = 1
        if (plot_cost == 1):
            plt.plot(combined_ll)
            plt.title("Likelihood")
            plt.show()
        return w,b

    def predict_test(self,test_data,test_label,w,b,add_intercept=True):
        if add_intercept:
            intercept = np.ones((test_data.shape[0], 1))
            test_data = np.hstack((intercept, test_data))

        softmax_out = self.softmax(test_data,w,b)
        result=softmax_out.argmax(axis=1)
        print(result)
        count = 0
        for i in range(result.shape[0]):
            predicted=int(result[i])
            print(predicted,",",test_label[i])
            if (predicted == int(test_label[i])):
                count = count + 1
            else:
                print("predicted is ", predicted)

        print("accuracy is ", float(count / test_data.shape[0]) * 100)

    def negative_log_likelihood(self,x,y,w,b):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = self.softmax(x,w,b)
        cross_entropy = - np.mean(np.sum(y * np.log(sigmoid_activation) +(1 - y) * np.log(1 - sigmoid_activation),axis=1))

        return cross_entropy


sf=Multiclass_Logistic(80000,0.001,0.01)
glass=pd.read_csv("D:\\Users\\vignesh.i\\Desktop\\glass_dataset.csv",names=['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','F1'])
glass['F1']=glass['F1'].replace([1,2,3,5,6,7], [0,1,2,3,4,5])
glass_data = shuffle(glass,random_state =4)
train_data,train_label,test_data,test_label=sf.split_test_train(glass_data,0.8)
train_label=(np.arange(np.max(train_label) + 1) == train_label[:, None]).astype(float)

w,b=sf.gradient(train_data,train_label)
#(9,6)
x = np.array([[1, 1, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 0]])
y = np.array([[1, 0],
                 [1, 0],
                 [1, 0],
                 [0, 1],
                 [0, 1],
                 [0, 1]])
#w,b=sf.gradient(x,y)
#print(w,b)
sf.predict_test(test_data,test_label,w,b)
#(number of features,number of class)
#(6,2) (2,2)
