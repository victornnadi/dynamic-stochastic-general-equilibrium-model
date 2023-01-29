# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 00:32:33 2022

@author: HI
"""

import numpy as np
#conda install -c anaconda scipy
import scipy as sci
import pandas as pd
# pip install matplotlib.pyplot
import matplotlib.pyplot as plt

#Question 1 - Finding the policy coefficients
def equations(vars):
    Nck,Nkk,Ncz,Nkz = vars
    eq1 = Nck-Nkk*Nck*1 - Nkk*0.0135
    eq2 = Ncz-Nkz*Nck*1 - Nkz*0.0135 - Ncz*0.979*1 + 0.0199
    eq3 = 1-Nkk*Nck*0.0972 - Nkk*0.9853
    eq4 = 0-Nkz*Ncz*0.0972 - Nkz*0.9853 - Ncz*0.979*0.0972 + 0.1237
    return (eq1,eq2,eq3,eq4)

Nck,Nkk,Ncz,Nkz = sci.optimize.fsolve(equations, (1,1,1,1))
print (equations)
    
#Question 2 - Simulating Consumption Ct and Capital Kt under Random Shocks
# Generate random numbers. First step: set the mean and standard deviation
mean = 0
std = 0.085
# Generate the random shocks and store in a variable called 'numbers' (comment out when done?)
numbers = np.random.normal(mean, std, size=1000)
print(numbers[:10])
#dataframe for shocks and responses(comment out when done?)
Main_df = pd.DataFrame(numbers)
#python is case sensitive,main_df is not equal to Main_df
print(Main_df)
#rename the shocks column
Main_df.rename(columns={0: "e"})  #why does this not work? 
#Adding the next column, Technology Zt = e*rho. First define rho
rho = 0.979
Main_df['Zt'] = Main_df[0]*rho
print(Main_df) #KeyError: 'e'; the column name change did not persist, why?

#Create columns for Nck,Ncz,Nkk,Nkz ratios 
Main_df['Nck'] = Nck/Main_df[0]
Main_df['Ncz'] = Ncz/Main_df[0]
Main_df['Nkk'] = Nkk/Main_df[0]
Main_df['Nkz'] = Nkz/Main_df[0]

#Adding another column for Capital Kt = e*Nkz
Main_df['Kt'] = Main_df[0]*Main_df['Nkz']
#Adding another column for Consumption Ct = Nck*Kt + Ncz*Zt
Main_df['Ct'] = Main_df['Kt']*Main_df['Nck'] + Main_df['Zt']*Main_df['Ncz']
#Standard deviation of Zt, Kt and Ct
stdZt = np.std(Main_df['Zt'])
stdKt = np.std(Main_df['Kt'])
stdCt = np.std(Main_df['Ct'])
#Correlation of Kt and Zt to Zt respectively
CorrKtZt = np.correlate(Main_df['Kt'],Main_df['Zt'])
CorrCtZt = np.correlate(Main_df['Ct'],Main_df['Zt'])

#Question 3 - Simulating Consumption Ct and Capital Kt under One-time Shock
Q3_df = pd.DataFrame()
#Creating column for error e
e = [0.085]
o=0 #has nothing to do with positioning in the list, just a counter
while o < 999:
    eNext = e[o]*rho
    e.append(eNext)
    o += 1
Q3_df['e'] = e
#initialize first value for Zt
print(Q3_df)

#Create all values for the Zt column
Q3_df['Zt'] = Q3_df['e']*rho

#create row for Nkz,Nkc,Ncz, Nck
#create row for Nkz
Q3_df['Nkz'] = Q3_df['e']*0.811
#create row for Nkz
Q3_df['Nkk'] = Q3_df['e']*11.65
#create row for Ncz
Q3_df['Ncz'] = Q3_df['e']*6.75
#create row for Nck
Q3_df['Nck'] = Q3_df['e']*5.46

#Initialize Kt in a list
Kt = [Q3_df.at[0,'Nkz']*Q3_df.at[0,'e']]
# Create remaining values for the Kt in the list
n=1
while n < 1000:
    ktNext = Kt[n-1]*Q3_df.at[n,'Nkk'] + Q3_df.at[n,'Nkz']*Q3_df.at[n,'Zt']
    Kt.append(ktNext)
    n += 1 
# create column for Kt in dataframe Q3_df
Q3_df['Kt'] = Kt

# Create column for consumption
Q3_df['Ct'] = Q3_df['Nck']*Q3_df['Kt'] + Q3_df['Ncz']*Q3_df['Zt'] 


# Creating the plots for Zt, Kt and Ct
plt.plot(Q3_df['Ct'], label='Ct')
plt.plot(Q3_df['Kt'], label='Kt')
plt.plot(Q3_df['Zt'], label='Zt')

# Question 4: run the regression of Ct on Kt and Zt
import statsmodels.formula.api as smf
regQ4 = 'Ct~Kt+Zt'
regQ4Out = smf.ols(regQ4,Main_df).fit()
print(regQ4Out.summary())

# I think the timeframe is too much for the graph to show the fluctuation in the data, here's a code with the dataset truncated at 40 periods for a better visualization.
# Q3_dftest = Q3_df.copy() #makes a deep copy of Q3_df (deep means change made in original will not affect this one)
# Q3_dftest = Q3_dftest.drop(labels=range(40,1000), axis = 0)
# print(Q3_dftest)

# plt.plot(Q3_dftest['Ct'])
# plt.plot(Q3_dftest['Kt'])
# plt.plot(Q3_dftest['Zt'])