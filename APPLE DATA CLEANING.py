# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:19:54 2024

@author: Sindhuja
"""

## APPLE STOCK PRICE  DATA CLEANING #

import pandas as pd 
apple = pd.read_csv(r"C:/Users/Sindhuja/Downloads/archive (3)/AppleStockPrice.csv")

apple.info()
apple.isna()
apple.describe
apple.shape
apple.dtypes

#FIRST MOMENT BUSINESS DESCISION
apple.Open.mean()  #22.620120899782133
apple.Open.median() #0.5332589999999999
apple.Open.mode() # 0.354911

apple.High.mean()   #22.86500082670661
apple.High.median() #0.5385044999999999
apple.High.mode()   #0.372768

apple.Low.mean()   #22.386529776416122
apple.Low.median() #0.524554
apple.Low.mode()   #0.357143

apple.Close.mean()   #22.63555764624183
apple.Close.median() #0.532366
apple.Close.mode()   #0.399554


apple.Adjustmentclose.mean()   #21.818456419389978
apple.Adjustmentclose.median() #0.433894
apple.Adjustmentclose.mode()   #0.085177

apple.Volume.mean()   #317648338.4259259
apple.Volume.median() #205307200.0
apple.Volume.mode()   #246400000

#SECOND MOMENT BUSINESS DESCISION 

apple.Open.var()  #2175.9363556120497
apple.Open.std()  # 46.646932971118794
range = max(apple.Open) - min(apple.Open)
range #236.430331


apple.High.var() #2223.321029338258
apple.High.std() #47.15210524820984
range = max(apple.High) - min(apple.High)
range #236.430331

apple.Low.var() #2132.5585339946047
apple.Low.std() #46.179633324601056
range = max(apple.Low) - min(apple.Low)
range #236.430331

apple.Close.var()  #2179.739571761954
apple.Close.std()  #46.687681156403066
range = max(apple.Close) - min(apple.Close)
range #234.7709


apple.Adjustmentclose.var() #2136.6919937400426
apple.Adjustmentclose.std() #46.224365801382746
range = max(apple.Adjustmentclose) - min(apple.Adjustmentclose)
range #234.51066699999998

apple.Volume.var() #1.1245946667632862e+17
apple.Volume.std() #335349767.6700084
range = max(apple.Volume) - min(apple.Volume)
range # 7421640800


#THIRD MOMENT BUSINESS DESCISION
apple.Open.skew() #2.511195098493477
apple.High.skew() #2.5093943120780087
apple.Low.skew()  #2.513312152604319
apple.Close.skew() #2.511256672009384
apple.Adjustmentclose.skew() #2.5577229494128093
apple.Volume.skew() #3.565099424968885
#FOURTH MOMENT BUSINESS DESCISION
apple.Open.kurt() #5.31175244092767
apple.High.kurt() #5.29580132400263
apple.Low.kurt()  #5.29580132400263
apple.Close.kurt() #5.31122540556577
apple.Adjustmentclose.kurt() #5.31122540556577
apple.Volume.kurt() #5.31122540556577



#FINDING THE DUPLICATES IN THE GIVEN DATA SET 
duplicate = apple.duplicated()
print(duplicate)
print("Number of duplicates :", sum(duplicate))
#PRESENCE OF NO DUPLICATES
#FINDING THE NULL VALUES
apple.isna().sum()
from sklearn.impute import SimpleImputer
#Reporting the no null values
# REPORTING NO NULL VALUES


#DETECTING THE OUTLIERS
import seaborn as sns
import numpy as np
sns.boxplot(apple.Open)
sns.boxplot(apple.High)
sns.boxplot(apple.Low)
sns.boxplot(apple.Close)
sns.boxplot(apple.Adjustmentclose)
sns.boxplot(apple.Volume)
#PRESENCE OF OUTLIERS FOR ALL THE COLUMNS IN DATASET

#TREATING THE OUTLIERS FOR THE APPLE OPEN 
IQR = apple['Open'].quantile(0.75) - apple['Open'].quantile(0.25)

lower_limit = apple['Open'].quantile(0.25) - (IQR * 1.5)
Upper_limit = apple['Open'].quantile(0.75) + (IQR * 1.5)

#REMOVING OUTLIERS BY USING WINSORIZATION
#PIP INSTALL FEATURE_ENGINE  # INSTALL THE PACKAGES 
from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'iqr',
                        tail = 'both',
                        fold = 1.5, 
                        variables = ['Open'])
apple = winsor_iqr.fit_transform(apple[['Open']])
sns.boxplot(apple.Open)


#TREATING THE OUTLIERS FOR THE APPLE HIGH
IQR = apple['High'].quantile(0.75) - apple['High'].quantile(0.25)

lower_limit = apple['High'].quantile(0.25) - (IQR * 1.5)
upper_limit = apple['High'].quantile(0.75) - (IQR * 1.5)


#REMOVING THE OUTLIERS FROM THE APPLE HIGH USEING WINSORIZATION TECHNIQUE 
#pip install feature engine # install the packages
from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'iqr',
                        tail = 'both',
                        fold = 1.5,
                        variables = ['High'])
apple = winsor_iqr.fit_transform(apple[['High']])
sns.boxplot(apple.High)

#TREATING THE OUTLIERS FOR THE APPLE LOW
IQR = apple['Low'].quantile(0.25) - apple['Low'].quantile(0.75)

lower_limit = apple['Low'].quantile(0.25) - (IQR * 1.5)
upper_limit = apple['Low'].quantile(0.75) - (IQR * 1.5)

#TREATING THE OUTLIERS WITH THE WINSORIZER FOR LOW
#import feature engine winsorizer  #install the packages 
from feature_engine.outliers import Winsorizer 
winsor_iqr = Winsorizer(capping_method = 'iqr',
                        tail = 'both',
                        fold = 1.5,
                        variables = ['Low'])
apple = winsor_iqr.fit_transform(apple[['Low']])
sns.boxplot(apple.Low)

print(apple.columns)
#TREATING THE OUTLIERS FOR THE CLOSE
IQR = apple['Close'].quantile(0.75) - apple['Close'].quantile(0.25)

# Define the lower and upper limits using the correct IQR
lower_limit = apple['Close'].quantile(0.25) - (IQR * 1.5)
upper_limit = apple['Close'].quantile(0.75) + (IQR * 1.5)

# Treating outliers with winsorization
from feature_engine.outliers import Winsorizer

Winsor_iqr = Winsorizer(capping_method='iqr', 
                        tail='both', 
                        fold=1.5, 
                        variables=['Close'])

# Apply Winsorizer and overwrite the 'Close' column
apple['Close'] = Winsor_iqr.fit_transform(apple[['Close']])

# Visualize using boxplot
import seaborn as sns
sns.boxplot(x=apple['Close'])


#TREATING THE OUTLIERS FOR THE Adjustmentclose
IQR = apple['Adjustmentclose'].quantile(0.75) - apple['Adjustmentclose'].quantile(0.25)
#Define the lower and upper limits using the correct IQR
lower_limit = apple['Adjustmentclose'].quantile(0.25) - (IQR * 1.5)
upper_limit = apple['Adjustmentclose'].quantile(0.75) - (IQR * 1.5)

# TREATING THE OUTLIERS USING WINSORIZATION

from feature_engine.outliers import Winsorizer
Winsor_iqr = Winsorizer(capping_method = 'iqr',
                        tail = 'both',
                        fold = 1.5,
                        variables= ['Adjustmentclose'])

#Apply winsorizer
apple['Adjustmentclose'] = Winsor_iqr.fit_transform(apple[['Adjustmentclose']])
sns.boxplot(apple.Adjustmentclose)

#TREATING OUTLIERS FOR THE VOLUME
IQR = apple['Volume'].quantile(0.25) - apple['Volume'].quantile(0.75)
#DEFINING THE UPPER AND LOWER LIMITS USING IQR
lower_limit = apple['Volume'].quantile(0.25) - (IQR *1.5)
upper_limit = apple['Volume'].quantile(0.75) - (IQR *1.5)

#TREATING THE OUTLIERS USING WINSORIZATION
from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'iqr',
                        tail = 'both',
                        fold = 1.5,
                        variables= ['Volume'])
apple['Volume'] = winsor_iqr.fit_transform(apple[['Volume']])
sns.boxplot(apple.Volume)


#TRANSFORMATION 
#NORMAL QUANTILE PLOT 
import scipy.stats as stats
import pylab
import numpy as np

#CHECKING WHETHER THE DATA IS NORMALLY DISTRIBUTED 
#TRANSFORMING THE DATA
stats.probplot(apple.Open, dist = "norm",plot = pylab)

stats.probplot(apple.Close, dist = "norm",plot = pylab )

stats.probplot(apple.High,dist = "norm", plot = pylab)

stats.probplot(apple.Low,dist = "norm", plot = pylab)

stats.probplot(apple.Adjustmentclose,dist = "norm", plot = pylab)

stats.probplot(apple.Volume,dist = "norm", plot = pylab)

#TRANSFORMATION TO MAKE THE OPEN VARIABLE NORMAL
stats.probplot(np.log(apple.Open), dist = "norm", plot = pylab)
stats.probplot(np.log(apple.Close), dist = "norm", plot = pylab)
stats.probplot(np.log(apple.High), dist = "norm", plot = pylab)
stats.probplot(np.log(apple.Low), dist = "norm", plot = pylab)
stats.probplot(np.log(apple.Adjustmentclose), dist = "norm", plot = pylab)
#INFERENCES ABOUT THE TRANSFORMATION FOR APPLE STOCK PRICES 
#Log transformations significantly improve the normality of Apple stock prices across all categories 
#(Open, Close, High, Low, Adjustment Close). 
#Handling zero values in the Volume column requires additional preprocessing steps, 
#Such as adding a small constant before applying the log transformation.
volume_mean = apple['Volume'][apple['Volume'] > 0].mean()  # Calculate mean of non-zero volumes
apple['Volume'] = apple['Volume'].replace(0, volume_mean)  # Replace zeroes with the mean
apple['Log_Volume'] = np.log(apple['Volume'])  # Apply log transformation
stats.probplot(np.log(apple.Volume), dist = "norm",plot = pylab )

#STANDARDIZATION AND NORMALIZATION
from sklearn.preprocessing import StandardScaler
a = apple.describe()
#intialise scalar 
scaler = StandardScaler()

from sklearn.preprocessing import MiniMaxscaler
minmaxscale = MinMaxScaler()

# Exclude non-numeric columns like dates and categorical data
apple_numeric = apple.select_dtypes(include=[np.number])

# Initialize the RobustScaler model
from sklearn.preprocessing import RobustScaler
robust_model = RobustScaler()

# Apply scaling to only the numeric columns
df_robust = robust_model.fit_transform(apple_numeric)

# Convert the result back to a DataFrame
dataset_robust = pd.DataFrame(df_robust, columns=apple_numeric.columns)

# Get the summary statistics
res_robust = dataset_robust.describe()

# Display the result
print(res_robust)
# CHECK FOR THE INFERENCES ON ROBUST MODEL 



#UNVARIATE ANALYSIS 
import matplotlib.pyplot as plt
import numpy as np
#HISTOGRAM 
plt.hist(apple.Open)
plt.hist(apple.Open,color = 'Blue')
#Data is more concentrated on right side tail is extended towards the left side

plt.hist(apple.Close)
plt.hist(apple.Close, color = "Green")
#Date is more concentrated on right side and tail extended towards the left side

plt.hist(apple.High)
plt.hist(apple.High, color = "Red")
#Date is more concentrated on right side and tail extended towards the left side

plt.hist(apple.Low)
plt.hist(apple.Low, color = "Blue")
#Date is more concentrated on right side and tail extended towards the left side

plt.hist(apple.Volume)
plt.hist(apple.Volume, color = "Yellow")
#Data is concentrated towards the right side with less tail extension

plt.hist(apple.Adjustmentclose)
plt.hist(apple.Adjustmentclose, color = "Red")
#Date is more concentrated on right side and tail extended towards the left side

#HISTOGRAM USING SEABORN
import seaborn as sns
sns.distplot(apple.Open, color = "Green")
#DATA IS MORE CONCENTRATED ON RIGHT SIDE, 
sns.distplot(apple.Close, color = "Red")
# DATA IS CONCENTRATED MORE ON RIGHT SIDE FLAT TAIL IS EXTENDED TOWARDS THE LEFT SIDE
sns.distplot(apple.High, color = "Violet")
# DATA IS CONCENTRATED MORE ON RIGHT SIDE, FLAT TILE IS EXTENDED TOWARDS THE LEFT SIDE
sns.distplot(apple.Low, color = "orange")
# DATA IS CONCENTRATED MORE ON RIGHT SIDE, FLAT TILE IS EXTENDED TOWARDS THE LEFT SIDE
sns.distplot(apple.Volume, color = "yellow")
# A SHARP PEAK OF CURVE IS SITUATED ON THE RIGHT SIDE, FLAT EXTENDED TOWARDS THE LEFT SIDE
sns.distplot(apple.Adjustmentclose, color = "Black")
# DATA IS DISTRIBUTED TOWARDS THE RIGHT SIDE, FLAT ELONGATED TAIL TOWARDS THE LEFT SIDE.

#KERNEL DENSITY PLOT
sns.kdeplot(apple.Open, color = "Red",fill = "Red")
#SHARP PEAK AND THICK TAILS POSITIVE KURTOSIS LEPTOKURTIC DISTRIBUTION
#If the mean is greater than the median, the data is skewed to the right (positive skewness).
#Skewness: If skewness > 0, the data is positively skewed (tail is on the left side).
#Kurtosis: High kurtosis (>3) indicates a leptokurtic distribution (sharp peak and fat tails).
sns.kdeplot(apple.Close, color = "Black", fill ='black')
#SHARP PEAK THICK TAILS, POSITIVE KURTOSIS LEPTOKURTIC DISTRIBUTION
#Mean vs. Median: If the mean is greater than the median, the distribution is positively skewed.
#Skewness: Positive skewness suggests a distribution tailing to the left.
#Kurtosis: High kurtosis suggests leptokurtic behavior with thicker tails.
sns.kdeplot(apple.High, color = "Green", fill = "Green")
#SHARP PEAK THICK TAILS, POSITIVE KURTOSIS LEPTOKURTIC DISTRIBUTION
#Mean vs Median: Indicates the direction of skewness.
#Skewness: Positive skew with extended tail towards the left.
#Kurtosis: Again, high kurtosis suggests a sharp peak with thicker tails.
sns.kdeplot(apple.Low, color = "Orange", fill = "orange")
#SHARP PEAK THICK TAILS, POSITIVE KURTOSIS LEPTOKURTIC DISTRIBUTION
#Mean vs Median: Helps in confirming the skewness.
#Skewness: Positive skew suggests that lower values are more spread out.
#Kurtosis: Leptokurtic distribution implies a sharp peak with longer tails.
sns.kdeplot(apple.Volume, color = "Yellow", fill = "Yellow")
#SHARP PEAK TAILS ARE NOT MUCH WIDE
#Mean vs. Median: If the mean is higher, this shows positive skew.
#Skewness: Likely positively skewed with a sharp peak.
#Kurtosis: Lower kurtosis here may indicate a more moderate distribution of volumes.
#Inference: A sharp peak with shorter tails, indicating most volumes fall into a concentrated range.
sns.kdeplot(apple.Adjustmentclose,color = "Violet", fill = "Violet")
#SHARP PEAK , WIDE TAILS POSITIVE KURTOSIS LEPTOKURTIC DISTRIBUTION
#Mean vs. Median: To be used for checking skewness.
#Skewness: Positive skew with a wider distribution.
#Kurtosis: Leptokurtic distribution with high kurtosis suggests thick tails and sharp peaks.
#Inference: Right-skewed, indicating the adjusted closing prices have a long left tail and concentrated high values.


#SCATTER PLOT 
Open = apple['Open']
Close = apple['Close']

#CREATE SCATTER PLOT
plt.scatter(Open,Close)
plt.Open('x-axislabel')
plt.Close('y-axislabel')
plt.tittle("scatter plot")
plt.show()
# SCATTER PLOT BETWEEN THE OPEN AND CLOSE SHOWS A THE POSITIVE CORRELATION, WHEN OPEN PRICES INCREASES 
#CLOSING PRICE IS TENDS TO DECREASE,RELATIONSHIP BETWEEN THE OPEN AND CLOSE PRICES APPEARS TO HAVE STRONG POSITIVE RELATIONSHIP

#SCATTER PLOT FOR THE HIGH AND LOW 
High = apple['High']
Low = apple['Low']
#SCATTER PLOT 
plt.scatter(High,Low)
plt.xlabel('X-axis label')
plt.ylabel('y-axis label')
plt.title("scatter plot")
plt.show()
#ITS A LINEAR POSITIVE RELATIONSHIP , WHICH MEANS HIGH PRICE OF STOCK INCREASES LOW PRICE OF THE STOCK ALSO INCREASES 
#THIS MEANS THAT STOCKS THAT REACHES HIGH VALUES IN A DAY, LIKELY TO FALL LOW

#SCATTER PLOT FOR THE VOLUME AND ADJUSTMENT CLOSE 
Volume = apple["Volume"]
Adjustmentclose = apple["Adjustmentclose"]
 #SCATTER PLOT
plt.scatter(Volume,Adjustmentclose)
plt.xlabel('x-axis label')
plt.ylabel('y-axis label')
plt.title("scatter plot")
plt.show()
# MODERATE NEGATIVE CORRELATION /NO CORRELATION

#EDA AFTER PREPROCESSING
#FIRST MOMENT BUSINESS DESCISION
apple.Open.mean() #12.157471763616556 
apple.Open.median() #0.5332589999999999
apple.Open.mode() #50.505135

apple.High.mean()   #12.249603520424834  
apple.High.median() #0.5385044999999999
apple.High.mode()   #50.792968

apple.Low.mean()   #12.03147055131173 
apple.Low.median() #0.524554
apple.Low.mode()   #49.969754

apple.Close.mean()   #12.151956361882718 
apple.Close.median() #0.532366
apple.Close.mode()   #50.46183


apple.Adjustmentclose.mean() # 10.598132996663942
apple.Adjustmentclose.median() # 0.433894
apple.Adjustmentclose.mode() #43.0255

apple.Volume.mean()   #288799902.1423384 
apple.Volume.median() #205307200.0
apple.Volume.mode()   #826279300

#SECOND MOMENT BUSINESS DESCISION 

apple.Open.var() #330.49710032265176 
apple.Open.std() #18.179579211924896
range = max(apple.Open) - min(apple.Open)
range #7421640800


apple.High.var() #334.71062688229125 
apple.High.std() #18.295098438715524
range = max(apple.High) - min(apple.High)
range #7421640800

apple.Low.var() #323.9780749899931 
apple.Low.std() #17.999390961640707
range = max(apple.Low) - min(apple.Low)
range #7421640800

apple.Close.var() #330.132018320629 
apple.Close.std() #18.16953544592236
range = max(apple.Close) - min(apple.Close)
range #50.4127225


apple.Adjustmentclose.var() #251.10110189762497
apple.Adjustmentclose.std() #15.846169944110311
range = max(apple.Adjustmentclose) - min(apple.Adjustmentclose)
range #42.98764437500001

apple.Volume.var() #5.38858293054846e+16
apple.Volume.std() #232133214.56759393
range = max(apple.Volume) - min(apple.Volume)
range
#826279300

#THIRD MOMENT BUSINESS DESCISION
apple.Open.skew() # 1.2758068889519592 
apple.High.skew() # 1.2733708469622893 
apple.Low.skew() # 1.2750060388305642 
apple.Close.skew() #1.2752624630854874 
apple.Adjustmentclose.skew() #1.2470507218519753
apple.Volume.skew() # 1.1180690951927026 
#FOURTH MOMENT BUSINESS DESCISION
apple.Open.kurt() # 0.007216413401252009 
apple.High.kurt() # 0.0001751562091159009 
apple.Low.kurt() # 0.0036970817004324807 
apple.Close.kurt() #0.005280529033025783 
apple.Adjustmentclose.kurt() #-0.10910493581358205
apple.Volume.kurt() # 0.15639908513868894 

#INFERENCES BEFORE AND AFTER PREPROCESSING 
#BEFORE PREPROCESSING: The data was highly dispersed, with strong positive skew and high kurtosis values,
#indicating extreme values and outliers, especially for the Volume column.
#AFTER PREPROCESSING: The data became more normalized. There was a reduction in skewness,
#kurtosis, variance, and standard deviation across all variables, indicating more compact and balanced distributions. 
#However, some skewness remains.

import pandas as pd

# If df_robust is a NumPy array, convert it to a DataFrame
df_robust_df = pd.DataFrame(df_robust)

# Now you can save it to a CSV file
df_robust_df.to_csv("C:\\Users\\Sindhuja\\Downloads\\APPLE_NEW.csv", index=True)
