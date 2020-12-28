#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:40:22 2020

@author: ubuntu
"""

import csv
attributes=[['Sunny','Cloudy','Rainy'],
            ['Warm','Cold'],
            ['Normal','High'],
            ['Strong','Weak'],
            ['Warm','Cool'],
            ['Same','Change']]

total_attributes=len(attributes)
print("\nTotal number of attributes is:",total_attributes)
print("The most specific hypothesis:['0','0','0','0','0','0']")
print("The most general hypothesis:['?','?','?','?','?','?']")
a=[]
print("\n the given training Data set is:")
with open('EnjoySport.csv','r') as cfile:
    for row in csv.reader(cfile):
        a.append(row)
        print(row)
print("\nTotal number of records is:",len(a))
print("The initial Hypothesis is:")
hypothesis=['0']*total_attributes
print(hypothesis)

#comparing with tarining examples of given dataset
for i in range(0,len(a)):
    if a[i][total_attributes]=='Yes':
        for j in range(0,total_attributes):
            if hypothesis[j]=='0' or hypothesis[j]==a[i][j]:
                hypothesis[j]=a[i][j]
            else:
                hypothesis[j]='?'
        print("\n Hypothesis for training example n0 {} is:\n" .format(i+1),hypothesis)
print("\n The maximally specific Hypothesis for s given training examples:")
print(hypothesis)
                
                
                
                
                
                
                
                
                
    


