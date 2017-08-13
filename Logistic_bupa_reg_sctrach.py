import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.utils import shuffle

warnings.filterwarnings("error")
class sfo(object):
    def split_test_train(self,indata,split_ratio):
        train_size=int(split_ratio*len(indata))
        train_data=indata.iloc[0:train_size,1:5].values #consider only the columns from 1 to 4(C2,C3,C4,C5)
        train_data=train_data.astype(float)
        train_label=indata.iloc[0:train_size,6].values

        test_data = indata.iloc[(train_size + 1):len(indata), 1:5].values
        test_data = test_data.astype(float) #convert all columns to float
        test_label = indata.iloc[(train_size + 1):len(indata), 6].values
        #standardize all the columns
        #without standardization it would throw ooverflow or divide by zero exception because of exponential
        for j in range((train_data.shape[1])):
            train_data[:, 0:4] = (train_data[:, 0:4] - train_data[:, 0:4].mean())/train_data[:, 0:4].std()
        print(train_data)

        for j in range((test_data.shape[1])):
            test_data[:, 0:4] = (test_data[:, 0:4] - test_data[:, 0:4].mean())/test_data[:, 0:4].std()
        return train_data,train_label,test_data,test_label



    def sigmoid(self,inX):
        return 1.0 / (1 + np.exp(-inX))

    def logistic_regression(self,features, target, num_steps, learning_rate, add_intercept=True):
        cost_history=[]
        if add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
        weights = np.ones(features.shape[1])

        for step in range(num_steps):
            scores = np.dot(features, weights)
            predictions = self.sigmoid(scores)

            # Update weights with gradient
            output_error_signal = target - predictions
            if(step == 0):
                print("step is ",step)
                gradient = np.dot(features.T, output_error_signal)
            else:
                gradient = np.dot(features.T, output_error_signal)+(learning_rate/features.shape[0])*weights
            weights += learning_rate *gradient

            # Print log-likelihood every so often
            if step % 1 == 0:
                param2=np.log(1 - predictions)
                param1=np.multiply(1 - target, param2)
               # cost = ((-np.multiply(target, np.log(predictions)) - param1).sum(axis=0) / features.shape[0])

                #reg = (learning_rate / 2 * features.shape[0]) * np.sum(np.power(weights[:, 1:weights.shape[1]], 2))
                reg = (learning_rate / 2 * features.shape[0]) * np.sum(np.power(weights, 2))
                cost = ((-np.multiply(target, np.log(predictions)) - param1).sum(axis=0) / features.shape[0])+reg#with regularization
                cost_history.append(cost)

        plot_cost = 1
        if (plot_cost == 1):
            plt.plot(cost_history)
            plt.title("Cost")
            plt.show()
        return weights

    def predict_test(self,test_data,test_label,theta,add_intercept=True):
        if add_intercept:
            intercept = np.ones((test_data.shape[0], 1))
            test_data = np.hstack((intercept, test_data))

        score = np.dot(test_data, theta)
        out_val = self.sigmoid(score)
        count=0
        for i in range(out_val.shape[0]):
            if(out_val[i] >= 0.5):
                predicted=1
            else:
                predicted=0
            if(predicted==int(test_label[i])):
                count=count+1
            else:
                print("predicted is ",predicted)

        print("accuracy is ",float(count/test_data.shape[0])*100)

bupa=pd.read_csv("D:\\Users\\vignesh.i\\Desktop\\logistic_bupa.csv",names=['C1','C2','C3','C4','C5','C6','F1'])
bupa['F1']=bupa['F1'].replace([1, 2], [0,1]) #replace 1 with 0 and 2 with 1 because python accepts the categorical variable as 0 and 1 only
bupa.info()
sf=sfo()
train_data,train_label,test_data,test_label=sf.split_test_train(bupa,0.8)


theta=sf.logistic_regression(train_data,train_label,1000,0.001)
sf.predict_test(test_data,test_label,theta)

print(theta)
