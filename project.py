# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:35:05 2019

@author: ASUS
"""
from flask import Flask ,redirect, url_for, request 
app = Flask(__name__) 

def preprocessing():
    import numpy as np
    import pandas as pd
    df=pd.read_csv(r'C:\Users\ASUS\Desktop\DATA SCIENSE\easy.csv')
    input_df=df.iloc[:,0:12]
    target_df=df.iloc[:,13]
    input_arr=input_df.to_numpy()
    #encoding stable and unstable to -1,1
    target_df[target_df=="unstable"]=-1
    target_df[target_df=="stable"]=1
    target_arr=target_df.to_numpy()
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(input_arr,target_arr,random_state=0)
    return x_train,x_test,y_train,y_test


def train(num):
    x_train,x_test,y_train,y_test=preprocessing()
    import numpy as np
    #num=int(input("Enter the no.of Records You want to Train this model on:"))
    x =x_train.copy()
    np.random.seed(8)
    w=np.random.uniform(low=0,high=1,size=(12,2),)
    t=y_train.copy()
    lrate= 0.025
    e=1
    D=[0,0]
    #print('learning rate of this epoch is',lrate)
    while(e<=1):
        #print('Epoch is',e)  
        for i in range(num):
            for j in range(2):
                temp=0
                for k in range(12):
                    temp = temp + ((w[k,j]-x[i,k])**2)
                D[j]=temp
            #print("D is ",D)
            if(D[0]<D[1]):
                J=0
                J_value=1
            else:
                J=1
                J_value=-1
            #print('winning unit is',J,"target is",t[i])
            #print('weight updation ...')
            if J_value==t[i]:
                for m in range(12):
                    w[m,J]=w[m,J] + (lrate *(x[i,m]-w[m,J]))
            else:
                for m in range(12):
                    w[m,J]=w[m,J] - (lrate *(x[i,m]-w[m,J]))
            #print('Updated weights',w)        
        e=e+1
        lrate = lrate/2
        #print(' updated learning rate after ',e,' epoch is',lrate)
    return w,x_test,y_test
def test(num,num_2):
    w,x_test,y_test=train(num)
    x =x_test.copy()
    t=y_test.copy()
    count=0
    e=1
    D=[0,0]
    #print('learning rate of this epoch is',lrate)
    while(e<=1):
        #print('Epoch is',e)  
        for i in range(num_2):
            for j in range(2):
                temp=0
                for k in range(12):
                    temp = temp + ((w[k,j]-x[i,k])**2)
                D[j]=temp        
            if(D[0]<D[1]):
                J=0  
                J_value=1            
            else:
                J=1
                J_value=-1
            if(J_value==t[i]):
                count+=1
        #print("Accuracy is: ",(count/num_2)*100)  
        #print(count)
        e=e+1      
        return((count/num_2)*100)
 
@app.route("/trainmodel",methods=['POST','GET']) 
def web(): 
    print("Calling train Function Web")
    train = int(request.args.get('train')  )
    test_1 = int(request.args.get('test')  )
    return str( test(train,test_1))
    
if __name__ == '__main__': 
    app.run(debug = True)        