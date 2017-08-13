import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.utils import shuffle

warnings.filterwarnings("error")
class sfo(object):
    def split_test_train(self,indata,split_ratio):
        train_size=int(split_ratio*len(indata))
        train_data=indata.iloc[0:train_size,2:4].values #take column 2,3 alone(gpa,rank)
        train_label=indata.iloc[0:train_size,0].values #take labels
        test_data=indata.iloc[(train_size+1):len(indata),2:4].values
        test_label = indata.iloc[(train_size + 1):len(indata), 0].values
        #No standardization here
        #for j in range((train_data.shape[1])):
        #   train_data[:, 0] = (train_data[:, 0] - train_data[:, 0].mean())/train_data[:, 0].std()

       # for j in range((test_data.shape[1])):
        #    test_data[:, 0] = (test_data[:, 0] - test_data[:, 0].mean())/test_data[:, 0].std()
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
            gradient = np.dot(features.T, output_error_signal)
            weights += learning_rate * gradient


            if step % 1 == 0:
                cost = ((-np.multiply(target, np.log(predictions)) - np.multiply(1 - target, np.log(1 - predictions))).sum(axis=0) / 268)
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
               # print("actual is ",test_label[i])

        print("accuracy is ",float(count/test_data.shape[0])*100)

bin_data=pd.read_csv("D:\\Users\\vignesh.i\\Downloads\\binary.csv")
bin_data = shuffle(bin_data,random_state =4) #Shuffles the data. random_state is for repeatitiveness of the data
sf=sfo()
train_data,train_label,test_data,test_label=sf.split_test_train(bin_data,0.67)
theta=sf.logistic_regression(train_data,train_label,1000,0.001)
sf.predict_test(test_data,test_label,theta)
#accuracy 77
