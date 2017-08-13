import pandas as pd
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from scipy.stats import pearsonr
from scipy.stats.distributions import chi2
bina=pd.read_csv("D:\\Users\\vignesh.i\\Downloads\\binary.csv")
bina['intercept']=1.0 #add intercept column
cols_to_keep = ['admit', 'intercept','gpa','rank']
data=bina[cols_to_keep]#contains the index of the columns that we need to keep
#bina.info() #gets the info
pd.set_option('display.width', 100)
pd.set_option('precision', 10)
description = bina.describe()
#print(description)
class_counts = bina.groupby('admit').size() #gives the class count
#print(class_counts)
bina.boxplot()
correlations = bina.corr(method='pearson')
#print(correlations)
#plt.show()
#bina['gre']=(bina['gre']-bina['gre'].mean())/bina['gre'].std()
print(pearsonr(bina['rank'].values,bina['gre'].values))
print(pearsonr(bina['gpa'].values,bina['gre'].values))
#with the correlation test it seems like gre,rank and gre,gpa is correlated. So it is good to remove gre
train, test = train_test_split(bina, train_size = 0.67,random_state =0)
train_cols=data.columns[1:] #omit gre and consider only gpa and rank
train_cols=[ 'intercept','gpa','rank']
logit = sm.GLM(train['admit'], train[train_cols],family=sm.families.Binomial()) #run GLM
result = logit.fit()
expected=test['admit']
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
print(chi2.sf((331.71-309.53),2)) #subtract null and residual deviance and give that to chi square with df as 1, if the p value is less
#thn 0.05 then the model is good fit
#gives accuracy 71
