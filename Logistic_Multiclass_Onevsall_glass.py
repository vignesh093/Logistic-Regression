import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.utils import shuffle

warnings.filterwarnings("error")
class sfo(object):
    def split_test_train(self, indata, split_ratio):
        train_size = int(split_ratio * len(indata))
        train_data = indata.iloc[0:train_size, 1:10].values  # take column 1 to 9(except C1)
        train_label = indata.iloc[0:train_size, 10].values  # take labels
        test_data = indata.iloc[(train_size + 1):len(indata), 1:10].values
        test_label = indata.iloc[(train_size + 1):len(indata), 10].values
        # No standardization here
        for j in range((train_data.shape[1])):
            train_data[:, j] = (train_data[:, j] - train_data[:, j].mean()) / train_data[:, j].std()

        for j in range((test_data.shape[1])):
            test_data[:, j] = (test_data[:, j] - test_data[:, j].mean()) / test_data[:, j].std()
        return train_data, train_label, test_data, test_label

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

            # Print log-likelihood every so often
            if step % 1 == 0:
                param2=np.log(1 - predictions)
                param1=np.multiply(1 - target, param2)
                cost = ((-np.multiply(target, np.log(predictions)) - param1).sum(axis=0) / 232)
                cost_history.append(cost)

        plot_cost = 1
        if (plot_cost == 1):
            plt.plot(cost_history)
            plt.title("Cost")
            #plt.show()
        return weights

    def predict_test(self,test_data,theta,add_intercept=True):
        if add_intercept:
            intercept = np.ones((test_data.shape[0], 1))
            test_data = np.hstack((intercept, test_data))

        score = np.dot(test_data, theta)
        out_val = self.sigmoid(score)
        return out_val
        #print("accuracy is ",float(count/test_data.shape[0])*100)

    def final_predict_accuracy(self,final_pred,test_label):
        count=0
        for i in range(final_pred.shape[0]):
            if (final_pred[i] == int(test_label[i])):
                count = count + 1
            else:
                print("predicted is ", final_pred[i])
        print("accuracy is ", float(count / final_pred.shape[0]) * 100)

glass=pd.read_csv("D:\\Users\\vignesh.i\\Desktop\\glass_dataset.csv",names=['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','F1'])
 #replace 1 with 0 and 2 with 1 because python accepts the categorical variable as 0 and 1 only
glass['F1']=glass['F1'].replace([1,2,3,5,6,7], [0,1,2,3,4,5])
glass_data = shuffle(glass,random_state =4)
sf=sfo()
train_data,train_label,test_data,test_label=sf.split_test_train(glass_data,0.63)


#The idea here is to to train the model equal to number of classes we have
#Here we have six classes from 0 to 5. For creating the first model we create a train label with 0 for class0 and 1 for other classes
#For creating the second model we create another train label with 0 for class1 and 1 for other classes and so on till the last model
#which is created with 0 for class5 and 1 for other classes
train_label0=np.where(train_label==0,0,1)
train_label1=np.where(train_label==1,0,1)
train_label2=np.where(train_label==2,0,1)
train_label3=np.where(train_label==3,0,1)
train_label4=np.where(train_label==4,0,1)
train_label5=np.where(train_label==5,0,1)


#For each train label created run a logistic gradient to get the theta(weight) values
theta0=sf.logistic_regression(train_data,train_label0,1000,0.001)
theta1=sf.logistic_regression(train_data,train_label1,1000,0.001)
theta2=sf.logistic_regression(train_data,train_label2,1000,0.001)
theta3=sf.logistic_regression(train_data,train_label3,1000,0.001)
theta4=sf.logistic_regression(train_data,train_label4,1000,0.001)
theta5=sf.logistic_regression(train_data,train_label5,1000,0.001)

#For each theta created above we need to predict and get the probabilites
#Assume we have 70 test dataset, so after running and concatenating the predict result, we would get 70 rows with 6 columns(number of
#classes) so it is (70,6) where the first row is the probability result for all 6 classes for the first test dataset
# for example [ 0.04939404  0.44295143  0.87754321  0.98378622  0.85855848  0.99621084]
#here the first probability is for class 0 and last for class5,Now look for the positive result(0.50) since for every positive we
#have deemed it as 0 and 1 otherwise. Here we have two 0.04939404  0.44295143 the highest is second so we finalize the predicted class
#for this test data row to be 1(since 0.44295143 is at index 1)
#If we dont have any values <0.5 then we have to look for confidence interval, we cannot compute the interval here, so we wil have to use
#packages
predict0=sf.predict_test(test_data,theta0)
predict1=sf.predict_test(test_data,theta1)
predict2=sf.predict_test(test_data,theta2)
predict3=sf.predict_test(test_data,theta3)
predict4=sf.predict_test(test_data,theta4)
predict5=sf.predict_test(test_data,theta5)
#print("predict0 is ",predict0)
#print("predict0 shape ",predict0.shape[0])
#print("predict1 is ",predict1)
#print("predict1 shape ",predict1.shape)
final_predict=np.concatenate((predict0.reshape(predict0.shape[0],1),predict1.reshape(predict1.shape[0],1),
                              predict2.reshape(predict2.shape[0],1),predict3.reshape(predict3.shape[0],1),
                              predict4.reshape(predict4.shape[0],1),predict5.reshape(predict5.shape[0],1)),axis=1)
print(final_predict)
#final_pred=np.argmax(final_predict,axis=1)

print(np.where(final_predict < 0.51,"correct","false"))
print(test_label)
#sf.final_predict_accuracy(final_pred,test_label)


