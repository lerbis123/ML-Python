from numpy.core.fromnumeric import transpose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.util
import os
import datetime
#os.path.dirname(__file__)
SampleData=pd.read_csv('C:\\Users\\james\\Documents\\Python\\Machine Learning Fun\\Linear Regression' + '\\' + 'ExampleStockData.csv')
print(SampleData.head())

Ticker = ""


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
def formatasDate(x):
    splitdate = x.split('-')
    return datetime.date(int(splitdate[0]),int(splitdate[1]),int(splitdate[2]))
filteredData['date'] = pd.to_datetime(filteredData['date'])
filteredData['date']=filteredData['date'].apply(lambda x:  x.toordinal())
filteredData = filteredData.sort_values(by=['date'],ascending= True)
print(filteredData.head())

#TO Do
#Add code to do simple linear regression 
LRequation = lambda x,m,b : x * m + b
## X*m = Y
## X^T * X * B = X^T *Y  
## B = [X^T * X]^-1 * Y

datecol = filteredData['date'].to_numpy(np.matrix).astype(np.double)
opencol = filteredData['open'].to_numpy(np.matrix).astype(np.double)

X = np.c_[datecol,np.ones([datecol.size,1])]
Y = np.c_[opencol]
XT = X.transpose()

B = (np.matmul(np.linalg.inv(np.matmul(XT, X)) ,np.matmul(XT , Y)))

Xline = [] 
Yline = []
for i in range(datecol.size):
    Xline.append(datecol[i])
    Yline.append(B[0][0]*datecol[i] + B[1][0])

plt.plot_date(datecol,opencol)
plt.plot(Xline,Yline)
plt.show()
