from numpy.core.fromnumeric import transpose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas._libs.tslibs.timestamps import Timestamp
import pandas.util
import os
import datetime

SampleData=pd.read_csv(os.path.dirname(__file__) + '\\' + 'ExampleStockData.csv')
print(SampleData.head())

Ticker = ""

#Wait for user to pick an accurate ticker
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

#filter out all data that doesnt have to do with chosen ticker
filteredData = SampleData[SampleData['ticker'] == Ticker]
filteredData['date'] = pd.to_datetime(filteredData['date'])

#create epoch column
def applyepoch(x):
    ts = Timestamp(x)
    return datetime.datetime(year=ts.year,month=ts.month, day=ts.day).timestamp()
epoch = filteredData['date'].apply(applyepoch)
filteredData["epoch"] = epoch
filteredData = filteredData.sort_values(by=['epoch'],ascending= True)
print(filteredData.head())

#Normalization Function
def Normalize(data: np.array):
    temp1 = np.array(data)
    MyMin,MyMax =  temp1.min(),temp1.max()
    retlist = np.array([])
    for d in range(temp1.size):
       retlist = np.append(retlist, (temp1[d] - MyMin)/(MyMax-MyMin))
    return retlist,MyMax,MyMin

#Get Normalization of epoch col
NormAndInfo= Normalize(filteredData['epoch'].to_numpy(np.matrix).astype(np.double))
NormEpochCol = NormAndInfo[0]
opencol = filteredData['open'].to_numpy(np.matrix).astype(np.double)

#Returns Lambda function of Line of best fit
def GetLineOfBestFit(Xcol,Ycol):
    #Linear Equation How I found it
    # m = slope, c = y intercept 
    ##### x*m + c = y
    #apply matrix variables to the equation X = [[series of dates],[series of 1s]], B = [[slope(B0)],[intercept(B1)]],Y = [[series of opening prices]] 
    ##### X * B = Y
    #apply the X transpose to the left of both sides
    ##### (X^T) * X * B = (X^T) *Y  
    #apply the inverse of the left side to both sides 
    ##### B = [(X^T) * X]^-1 (X^T) * Y
    
    #Create X matrix 
    #first col is the date values 
    #second col is filled with 1s for the y intercept
    X = np.c_[Xcol,np.ones([Xcol.size,1])]
    Y = np.c_[Ycol]
    XT = X.transpose()

    ## B = [(X^T) * X]^-1 (X^T) * Y
    B = (np.matmul(np.linalg.inv(np.matmul(XT, X)) ,np.matmul(XT , Y)))
    return lambda x: B[0][0]*x + B[1][0]

#Our Line Of Best Fit
LOBF= GetLineOfBestFit(NormEpochCol,opencol)

#Generate Equation Values
Xline = [] 
Yline = []
for i in range(NormEpochCol.size):
    Xline.append(filteredData["date"].to_numpy()[i])
    Yline.append(LOBF(NormEpochCol[i]))


#Plot Everything
plt.plot(filteredData["date"].to_numpy(),opencol)
plt.plot(Xline,Yline)
plt.show()

#Stock Cost Date Guessing
def isdate(string:str):
        dmy = ret.split("/")
        isnum = True
        for num in dmy:
            isnum=isnum and num.isnumeric()
        if len(dmy) == 3 and isnum:
            return True
        else:
            return False
NotFinished = True
while (NotFinished):
    ret = input("Please Enter A Date To Guess the stock price at (Numeric Format: Year/Month/day)")

    if isdate(ret):
        daymonthyear = ret.split("/")
        #Need these to normalize user entry to fit existing data
        NormalizationMax = NormAndInfo[1]
        NormalizationMin = NormAndInfo[2]

        #change user entry to epoch and apply normalization
        def getSingleEpoch(y, m, d):
            return datetime.datetime(year=int(y),month=int(m), day=int(d)).timestamp()
        res = (getSingleEpoch(daymonthyear[0],daymonthyear[1],daymonthyear[2]) - NormalizationMin)/(NormalizationMax -NormalizationMin)

        #plug normalized epoch into line of best fit function and print results
        print("The Open Should be at: $" +str(LOBF(res)))
    else:
        print("Entry Invalid")
    YN=input("Would you like another guess (Y/N)")
    if(YN=="N"):
        NotFinished = False