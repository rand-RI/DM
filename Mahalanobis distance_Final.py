#stage 1: IMPORT DATA AND ASSIGN VARIABLE CODE
import numpy as np 
from pylab import *
from pandas.io.data import *
ticker1 = ['^GSPC']  #select data and insert into matrix
stock_data1 = get_data_yahoo(ticker1,start='1/1/1980',end='1/1/2009')

ticker2 = ['^NYA']
stock_data2 = get_data_yahoo(ticker2,start='1/1/1980',end='1/1/2009')


x = stock_data1['Adj Close']    #select adjusted closing price values


y = stock_data2['Adj Close']    #select adjusted closing price values


#stage 2
z=x.join(y) #join x and y
z_cov=z.cov()   #covariance of x and y
z_invcov=inv(z_cov)  #invcovariance of x and y

#stage 3
x_mean= x.mean()    #mean of x
y_mean= y.mean()    #mean of y
xy_mean= x_mean,y_mean

x_diff= x.subtract(x_mean)  #subtract mean from x
x_diff_matrix= x_diff.as_matrix()   #express in array


y_diff= y.subtract(y_mean)  #subtract mean from y
y_diff_matrix= y_diff.as_matrix()   #express in array

xy_diff=x_diff.join(y_diff)     #join x_diff and y_diff
xy_diff_matrix= xy_diff.as_matrix()     #express in array

#stage 4 formula building


md = []                         #build formula for Mahalanobis distance
for i in range(len(xy_diff_matrix)):
    md.append(np.sqrt(np.dot(np.dot(np.transpose(xy_diff_matrix[i]),z_invcov),xy_diff_matrix[i])))
md



def MahalanobisDist(x, y):
    z=x.join(y)
    z_cov=z.cov()
    z_invcov=inv(z_cov)
    x_mean= x.mean()
    y_mean= y.mean()
    xy_mean= x_mean,y_mean
    x_diff= x.subtract(x_mean)
    x_diff_matrix= x_diff.as_matrix()
    y_diff= y.subtract(y_mean)
    y_diff_matrix= y_diff.as_matrix()
    xy_diff=x_diff.join(y_diff)
    xy_diff_matrix= xy_diff.as_matrix()    
    
    md = []
    for i in range(len(xy_diff_matrix)):
        md.append((np.dot(np.dot(np.transpose(xy_diff_matrix[i]),z_invcov),xy_diff_matrix[i])))
    return md
    
MahalanobisDist(x,y)



 
    






    
    
    
    
    
