from numpy.core.fromnumeric import transpose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.util
import os
import datetime

SampleData=pd.read_csv(os.path.dirname(__file__) + '\\' + 'ExampleStockData.csv')
print(SampleData.head())

Ticker = ""

#wait for user to pick an accurate ticker
while Ticker == "":
    print("Please enter a stock name from the following list:")
    i = 0
    printval = ""
    for a in np.unique(SampleData["ticker"]):
        i+=1
        if i < 3:
            printval = printval + a + " ,   " 
        elif i == 3:
            print(printval + a)
            printval = ""
            i = 0 
    inputval = input()
    if inputval in SampleData["ticker"].to_numpy():
        Ticker = inputval
    else:
        print("Please enter a value on the list")
filteredData = SampleData[SampleData['ticker'] == Ticker]
filteredData['date'] = pd.to_datetime(filteredData['date'])
filteredData['date']=filteredData['date'].apply(lambda x:  x.toordinal())
filteredData = filteredData.sort_values(by=['date'],ascending= True)
print(filteredData.head())

#simple linear equation
# m = slope, c = y intercept 
##### x*m + c = y
#apply matrix variables to the equation X = [[series of dates],[series of 1s]], B = [[slope(B0)],[intercept(B1)]],Y = [[series of opening prices]] 
##### X * B = Y
#apply the X transpose to the left of both sides
##### (X^T) * X * B = (X^T) *Y  
#apply the inverse of the left side to both sides 
##### B = [(X^T) * X]^-1 (X^T) * Y

#get cols as numpy arrays
datecol = filteredData['date'].to_numpy(np.matrix).astype(np.double)
opencol = filteredData['open'].to_numpy(np.matrix).astype(np.double)

#Create X matrix 
#first col is the date values 
#second col is filled with 1s for the y intercept
X = np.c_[datecol,np.ones([datecol.size,1])]
Y = np.c_[opencol]
XT = X.transpose()

## B = [(X^T) * X]^-1 (X^T) * Y
B = (np.matmul(np.linalg.inv(np.matmul(XT, X)) ,np.matmul(XT , Y)))

#Generate Equation Values
Xline = [] 
Yline = []
for i in range(datecol.size):
    Xline.append(datecol[i])
    Yline.append(B[0][0]*datecol[i] + B[1][0])

#Plot Everything
plt.plot_date(datecol,opencol)
plt.plot(Xline,Yline)
plt.show()
