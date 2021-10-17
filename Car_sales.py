
"""
@author: khare
"""

## Name- SATYAM RAJ KHARE



import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

#Importing dataset
df= pd.read_csv(r"\ToyotaCorolla.csv")

#preprocessing & EDA
df.columns

df.drop(['Id', 'Model','Mfg_Month', 'Mfg_Year','Fuel_Type','Met_Color', 'Color', 'Automatic','Cylinders', 'Mfr_Guarantee',
       'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2','Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
       'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio','Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim',
       'Radio_cassette', 'Tow_Bar'],axis=1,inplace=True)

df.isna().sum() # no missing values
df.duplicated().sum() # 1 duplicated value found
df.drop_duplicates(keep="first",inplace=True) # duplicate value removed

des=df.describe()
Skewness=df.skew()
Kurtosis=df.kurt()

# Jointplot

sns.jointplot(x=df['Age_08_04'], y=df['Price'])
sns.jointplot(x=df['KM'], y=df['Price'])
sns.jointplot(x=df['HP'], y=df['Price'])
sns.jointplot(x=df['Doors'], y=df['Price'])
sns.jointplot(x=df['Gears'], y=df['Price'])
sns.jointplot(x=df['Quarterly_Tax'], y=df['Price'])
sns.jointplot(x=df['Weight'], y=df['Price'])
sns.jointplot(x=df['cc'], y=df['Price'])

# Correlation matrix 
corr=df.corr()

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(df['Price'], dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(df.iloc[:, :])
                             

#Outlier treatment
sns.boxplot(data=df,orient=True)
#conda install -c conda-forge feature_engine
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['KM'])
df['KM'] = winsor.fit_transform(df[['KM']])


winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['cc'])
df['cc'] = winsor.fit_transform(df[['cc']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(data=df,orient=True);plt.title('Boxplot');plt.show()

### Normalization
def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

m=df.iloc[:,0]
df = norm_func(df.iloc[:,1:])
df['Price']= m
df = df.iloc[:,[8,0,1,2,3,4,5,6,7]]

############### preparing model considering all the variables ######################## 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price~  Age_08_04 + HP + KM + Doors +Gears + Quarterly_Tax + cc + Weight', data = df).fit() # regression model

# Summary
ml1.summary()
# p-values for Doors are more than 0.05

pred = ml1.predict(df)

###Error calculation
# residual values 
resid = pred - df.Price
###RMSE value## 
rmse1 = np.sqrt(np.mean(resid * resid))
rmse1 

##R-squared value##
Rsquared1 =ml1.rsquared 
Rsquared1 




####Added Vraiable Plot###
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(ml1)
# Door is near to zero has influence on other features

#####Influence Index Plots####
# Checking whether data has any influential values 
sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 221,960 is showing high influence so we can exclude that entire row

df = df.drop(df.index[[221,960]])

#####Model_2 after removing Influencial values  #####               
ml2 = smf.ols('Price~  Age_08_04 + HP + KM + Doors +Gears + Quarterly_Tax + cc + Weight', data = df).fit()    

# Summary
ml2.summary()
# Droping index 221,960 dosen't change p values of Door feature.

pred = ml2.predict(df)

###Error calculation
# residual values 
resid = pred - df.Price
# RMSE value 
rmse2 = np.sqrt(np.mean(resid * resid))
rmse2 

##R-squared value
Rsquared2 =ml2.rsquared 
Rsquared2 



##### VIF ##########
# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_Age = smf.ols('Age_08_04 ~ HP + KM + Doors +Gears + Quarterly_Tax + cc + Weight', data = df).fit().rsquared  
vif_Age = 1/(1 - rsq_Age) 

rsq_HP = smf.ols('HP ~ Age_08_04  + KM + Doors +Gears + Quarterly_Tax + cc + Weight', data = df).fit().rsquared  
vif_HP = 1/(1 - rsq_HP)

rsq_KM = smf.ols('KM ~ Age_08_04  + HP + Doors +Gears + Quarterly_Tax + cc + Weight', data = df).fit().rsquared  
vif_KM = 1/(1 - rsq_KM) 

rsq_Doors= smf.ols('Doors~ Age_08_04  + HP + KM +Gears + Quarterly_Tax + cc + Weight', data = df).fit().rsquared  
vif_Doors = 1/(1 - rsq_Doors ) 

rsq_Gears = smf.ols('Gears ~ Age_08_04  + HP + KM + Doors + Quarterly_Tax + cc + Weight', data = df).fit().rsquared  
vif_Gears = 1/(1 - rsq_Gears ) 

rsq_Tax= smf.ols('Quarterly_Tax ~ Age_08_04  + HP + KM +Gears +Doors + cc + Weight', data = df).fit().rsquared  
vif_Tax = 1/(1 - rsq_Tax) 

rsq_Weight = smf.ols('Weight ~ Age_08_04  + HP + KM +Gears +Doors + cc +  Quarterly_Tax', data = df).fit().rsquared  
vif_Weight = 1/(1 - rsq_Weight) 

rsq_cc = smf.ols('cc ~ Age_08_04  + HP + KM +Gears +Doors + Weight +  Quarterly_Tax', data = df).fit().rsquared  
vif_cc = 1/(1 - rsq_cc) 

# Storing vif values in a data frame
d1 = {'Variables':['Age_08_04','HP','KM','Doors','Gears','Quarterly_Tax', 'Weight','cc'], 'VIF':[vif_Age, vif_HP, vif_KM ,vif_Doors,vif_Gears,vif_Tax,vif_Weight,vif_cc]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# VIF of all values is lower than 10 



##################### Final model #####################
# In Final model we drop Doors feature,as in Partial Regression Plot it is near to Zero.


final_ml = smf.ols('Price~  Age_08_04 + HP + KM +Gears + Quarterly_Tax + cc + Weight',data=df).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(df)

###Error calculation
# residual values 
resid = pred - df.Price
# RMSE value 
rmse3 = np.sqrt(np.mean(resid * resid))
rmse3 

##R-squared value
Rsquared3 =ml1.rsquared 
Rsquared3 

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

### Residuals vs Fitted plot##
sns.residplot(x = pred, y = df.Price, lowess = True)
plt.xlaoel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()


##### Influencial Plot ####
sm.graphics.influence_plot(final_ml)

##### Added Variable Plot ##
sm.graphics.plot_partregress_grid(final_ml)


########################## Model Building #############################
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size = 0.2,random_state=0) # 20% test data

# preparing the model on train data 
model = smf.ols('Price~  Age_08_04 + HP + KM +Gears + Quarterly_Tax + cc + Weight', data = df_train).fit()

# prediction on test data set 
test_pred = model.predict(df_test)

# test residual values 
test_resid = test_pred - df_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model.predict(df_train)

# train residual values 
train_resid  = train_pred - df_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

##R-squared value
Rsquared4 =model.rsquared 
Rsquared4 

# RMSE Table
data1 = {"MODEL":pd.Series(["ml1", "ml2", "Final_ml","model(test)" ,"model(train)"]), "RMSE":pd.Series([rmse1, rmse2, rmse3,test_rmse,train_rmse])}
table_rmse = pd.DataFrame(data1)
table_rmse

#R-squared Table
data2 = {"MODEL":pd.Series(["ml1", "ml2", "Final_ml","model"]), "R-squared":pd.Series([Rsquared1, Rsquared2, Rsquared3,Rsquared4])}
table_rsqr = pd.DataFrame(data2)
table_rsqr
