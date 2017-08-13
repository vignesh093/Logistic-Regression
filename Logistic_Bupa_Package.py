import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from scipy.stats import pearsonr
from scipy.stats.distributions import chi2
import statsmodels.formula.api as smf

bupa=pd.read_csv("D:\\Users\\vignesh.i\\Desktop\\logistic_bupa.csv",names=['C1','C2','C3','C4','C5','C6','F1'])

bupa['F1']=bupa['F1'].replace([1, 2], [0,1])
#replace 1 with 0 and 2 with 1 because python accepts the categorical variable as 0 and 1 only
bupa['intercept']=1.0 #add intercept
cols_to_keep = ['intercept','C1','C2','C3','C4','C5','C6','F1']
data=bupa[cols_to_keep]
pd.set_option('display.width', 100)
pd.set_option('precision', 10)
description = bupa.describe()

correlations = bupa.corr(method='pearson')
print(pearsonr(bupa['C1'].values,bupa['C2'].values))
print(pearsonr(bupa['C2'].values,bupa['C3'].values))
#print(pearsonr(bupa['C3'].values,bupa['C4'].values))
#print(pearsonr(bupa['C4'].values,bupa['C5'].values))
#print(pearsonr(bupa['C1'].values,bupa['C5'].values))
print(pearsonr(bupa['C2'].values,bupa['C5'].values))
#print(pearsonr(bupa['C3'].values,bupa['C5'].values))
#print(pearsonr(bupa['C1'].values,bupa['C4'].values))
print(pearsonr(bupa['C1'].values,bupa['C6'].values))
print(pearsonr(bupa['C2'].values,bupa['C6'].values))
print(pearsonr(bupa['C3'].values,bupa['C6'].values))
print(pearsonr(bupa['C4'].values,bupa['C6'].values))
print(pearsonr(bupa['C5'].values,bupa['C6'].values))
print(pearsonr(bupa['C2'].values,bupa['C4'].values)) #pearsonr would give a correlation value and a p value decide based on that.
#we have only considered C2,C3,C4,C5 and omitted C1,C6 because of correlation
#also if two columns are correlated, run the algo by removing first column and then run the algo by removing second column and check for
#decrease in deviance and decide based on that
#here removing C2 by keeping C1 gives less deviance but less accuarcy as well than removing C1 and keeping C2 but gives good accuarcy

train, test = train_test_split(bupa, train_size = 0.67,random_state =0)
#train_cols=data.columns[2:6]
train_cols = ['intercept','C1','C2','C3','C4','C5','C6']

print(train_cols)
logit = sm.GLM(train['F1'], train[train_cols],family=sm.families.Binomial())
#logit=smf.Logit(train['F1'], train[train_cols])
result = logit.fit()
expected=test['F1']
predicted=result.predict(test[train_cols])
count=0
for i in range(expected.size):
    if(predicted.iloc[i] >= 0.5):
        predicted_val=1
    else:
        predicted_val=0

    if(expected.iloc[i]==predicted_val):
        count=count+1
    else:
        print(predicted.iloc[i])
print("accuracy ",count/len(test))
print(result.summary())
#print(result.pvalues)
p=6
AIC= -2*(-135.14)+2*(p+1)
print("AIC is ",AIC)
print(chi2.sf((310.60-238),1))

#How to proceed with the model

#Goodness of fit:
