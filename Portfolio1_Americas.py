#stage 1: IMPORT DATA AND ASSIGN VARIABLE CODE
import numpy as np 
from pylab import *
from pandas.io.data import *

ticker1 = ['^BVSP']  #select data and insert into matrix
stock_data1 = get_data_yahoo(ticker1,start='1/1/1980',end='1/1/2014')

ticker2 = ['^IPSA']
stock_data2 = get_data_yahoo(ticker2,start='1/1/1980',end='1/1/2014')\

ticker3 = ['CLIBX']  #select data and insert into matrix
stock_data3 = get_data_yahoo(ticker3,start='1/1/1980',end='1/1/2014')

ticker4 = ['^MXX']
stock_data4 = get_data_yahoo(ticker4,start='1/1/1980',end='1/1/2014')

ticker5 = ['EPU']  #select data and insert into matrix
stock_data5 = get_data_yahoo(ticker5,start='1/1/1980',end='1/1/2014')


a = stock_data1['Adj Close'] 
b = stock_data2['Adj Close'] 
c = stock_data3['Adj Close'] 
d = stock_data4['Adj Close'] 
e = stock_data5['Adj Close'] 

#stage 2 build inverse cov matrix
f=a.join(b)
g=f.join(c)
h=g.join(d)
j=h.join(e)

j_cov=j.cov()
j_invcov=inv(j_cov)

#stage 3 calculate mean differences
a_mean= a.mean()    
b_mean= b.mean()
c_mean= c.mean()    
d_mean= d.mean()
e_mean= e.mean()    
abcde_mean= a_mean,b_mean,c_mean,d_mean,e_mean


a_diff= a.subtract(a_mean)  
a_diff_matrix= a_diff.as_matrix()   

b_diff= b.subtract(b_mean)  
b_diff_matrix= b_diff.as_matrix()   

c_diff= c.subtract(c_mean)  
c_diff_matrix= c_diff.as_matrix()  

d_diff= d.subtract(d_mean)  
d_diff_matrix= d_diff.as_matrix()   

e_diff= e.subtract(e_mean)  
e_diff_matrix= e_diff.as_matrix()   

ab_diff=a_diff.join(b_diff)
abc_diff=ab_diff.join(c_diff)
abcd_diff=abc_diff.join(d_diff)
abcde_diff=abcd_diff.join(e_diff)
abcde_diff_matrix= abcde_diff.as_matrix()

#stage 4 build formula

md = []                         
for i in range(len(abcde_diff_matrix)):
    md.append((np.dot(np.dot(np.transpose(abcde_diff_matrix[i]),j_invcov),abcde_diff_matrix[i])))
md


def MahalanobisDist(x, y):
    f=a.join(b)
    g=f.join(c)
    h=g.join(d)
    j=h.join(e)
    j_cov=j.cov()
    j_invcov=inv(j_cov)
    a_mean= a.mean()   
    b_mean= b.mean()
    c_mean= c.mean()    
    d_mean= d.mean()
    e_mean= e.mean()    
    abcde_mean= a_mean,b_mean,c_mean,d_mean,e_mean
    a_diff= a.subtract(a_mean)  
    a_diff_matrix= a_diff.as_matrix()   
    b_diff= b.subtract(b_mean)  
    b_diff_matrix= b_diff.as_matrix()   
    c_diff= c.subtract(c_mean)  
    c_diff_matrix= c_diff.as_matrix()   
    d_diff= d.subtract(d_mean) 
    d_diff_matrix= d_diff.as_matrix()  
    e_diff= e.subtract(e_mean)  
    e_diff_matrix= e_diff.as_matrix()   
    ab_diff=a_diff.join(b_diff)
    abc_diff=ab_diff.join(c_diff)
    abcd_diff=abc_diff.join(d_diff)
    abcde_diff=abcd_diff.join(e_diff)
    abcde_diff_matrix= abcde_diff.as_matrix()  
    
    md = []                        
    for i in range(len(abcde_diff_matrix)):
        md.append((np.dot(np.dot(np.transpose(abcde_diff_matrix[i]),j_invcov),abcde_diff_matrix[i])))
    return md    
    
MahalanobisDist(x,y)    
   
#satge 6 plot 
import matplotlib.pyplot as plt
plt.plot(MahalanobisDist(x,y))





