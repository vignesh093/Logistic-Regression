import numpy as np
import pandas as pd

class LogisticRegression(object):


    def split_test_train(self,X, Y, splitsize):
        splitintsize=int(splitsize*X.shape[0])
        for j in range((X.shape[1])):
            X[:, j] = (X[:, j] - X[:, j].mean())

        Y=np.reshape(Y,(X.shape[0],1))

        trainset_copy=np.append(X,Y,axis=1)
        np.random.shuffle(trainset_copy)
        print(trainset_copy.shape)
        training_data,training_class_values,test_data,test_class_values=trainset_copy[:splitintsize,:4]\
            ,trainset_copy[:splitintsize,4],trainset_copy[splitintsize:,:4],trainset_copy[splitintsize:,4]

        return training_data,training_class_values,test_data,test_class_values

    def sigmoid(self,scores):
        return (1/(1+np.exp(-scores)))



    def reggradientdescnt(self,training_data,training_class_values,theta,learning_rate=12):
        m=len(training_class_values)


        for i in range(10):
            score = np.dot(training_data, theta)
            h = self.sigmoid(score)
            theta=theta-(learning_rate*(1/m*(training_data.T.dot(h-training_class_values))))
        return theta

    def test_model(self,test_data,test_class_values,theta):
        score=np.dot(test_data,theta)
        out_val = self.sigmoid(score)

        count=0
        for i in range(out_val.shape[0]):
            if(out_val[i] >= 0.5):
                predicted=1
            else:
                predicted=0
            if(predicted==int(test_class_values[i])):
                count=count+1

        print("accuracy is ",float(count/test_data.shape[0])*100)


iris = pd.read_csv('D:\\Users\\vignesh.i\\Desktop\\iris.csv')

X = iris.iloc[0:100,:4].values
Y = iris.iloc[0:100,4].values
Y=np.where(Y == 'Iris-setosa',0,(np.where(Y == 'Iris-versicolor',1,2 )))
lg=LogisticRegression()
training_data,training_class_values,test_data,test_class_values=lg.split_test_train(X,Y,0.7)
theta=np.ones(training_data.shape[1])
print(training_class_values.shape)
#lg.regCostFunction(training_data,training_class_values,theta)
theta=lg.reggradientdescnt(training_data,training_class_values,theta)
lg.reggradientdescnt(test_data,test_class_values,theta)
lg.test_model(test_data,test_class_values,theta)
