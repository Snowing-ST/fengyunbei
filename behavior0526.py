# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:35:07 2018
behavior 数据预处理
需改成类下次可以调用
@author: situ
"""

import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

#os.chdir("C:/Users/steven.wang/Desktop/风云杯/data")
#behaviorA = pd.read_csv("trainA/train_behavior_A.csv")
#behaviorB = pd.read_csv("trainB/train_behavior_B.csv")

def beha_pre(behaviorA,behaviorB):
    print("begin to compute the missing rate......")
    #只有var1——var10的变量缺失值少于50%
    missing = np.array(behaviorA.apply(lambda x: sum(x.isnull())/len(behaviorA)))

#对于缺失太多的变量，缺失90%直接删
    print("begin to delete the heavily missing variables......")
    miss90 = list(behaviorA.columns[missing>0.8])
    try:
        behaviorA.drop(miss90,axis=1,inplace = True) 
    except:
        pass
    try:
        behaviorB.drop(miss90,axis=1,inplace = True) 
    except:
        pass
    
#先做前10个变量的处理
#var1年龄，分段 18-30 30-40 40以上
#var3	var4	var5	var6 很多类别的定性变量
#var2 var7	var8	var9为0-1 看0-1的比例，比例相差过大就删
#var10 除6.8%的缺失值外，其他都是1，删除
    print("begin to deal with var1-var10......")
    behavior = [behaviorA,behaviorB]
    for b in behavior:
        b["q_var1"] = 0
        b["q_var1"][b["var1"]>29]=1
        b["q_var1"][b["var1"]>39]=2
    
    def topNpop(series,n=10):
        v = series.value_counts()
        return sum(v[:n-1])/sum(v)
#    
#    #查看分类变量前10类所占比例
#    for data in behavior:
#        data = data.ix[:,["var3","var4","var5","var6"]]
#        print(data.apply(lambda x:topNpop(x,10)))
    
    #var3存在1%的缺失，用众数填补
#    for b in behavior:
#        mode = behaviorA["var3"].value_counts().index[0]
#        b["var3"].fillna(mode,inplace=True) 
    
    #var4 var6 类别多且不集中，直接删掉  
    for b in behavior:
        b.drop(["var4","var6"],axis=1,inplace=True)
     #var7	var8	var9中，0或1都占了88%以上。。还是不删了，删了就没变量了。。。
#    for b in behavior:
#        var2_9 = b.ix[:,["var2","var7","var8","var9"]]
#        print(var2_9.apply(lambda x:topNpop(x,1)))
    #behaviorB居然没有var10，醉了。。
    behaviorA.drop("var10",axis=1,inplace=True)
      
    print("begin to deal with year variables......")
    #年份类的，var16	var17	var19 ;var16var17换成距今年份，缺失值补零;var19 全是6月1日 直接删
    for b in behavior:
        b["var17"] = 2018-b["var17"] 
        b["var16"] = b["var16"].astype(str)
        b["var16"] = b["var16"].str.slice(0,4) #存在少数不规则数据
        b["var16"][b["var16"]=="nan"]= np.nan
        b["var16"] = 2018-b["var16"].astype(float)
        b["var16"][b["var16"]==2018]=0
        b.drop("var19",axis=1,inplace=True) 
    
    print("begin to imputate the float and get dummy the object")
    #int类float类变量用0填补
#    def impute_int_float(x):
#        if x.dtype=="float64" or x.dtype=="int64":
#            return x.fillna(np.median(x))
#        else:
#            return x
    
#    for i in range(behaviorA.shape[1]):
#        behaviorA.ix[:,i] = impute_int_float(behaviorA.ix[:,i])
#    for i in range(behaviorB.shape[1]):
#        behaviorB.ix[:,i] = impute_int_float(behaviorB.ix[:,i])
                      
    #object 类，先删除超多类别且不集中的，其他one-hot    
    #只有var11 var12  var13 var18 是object
    for b in behavior:
        b.drop("var13",axis=1,inplace=True) #有好几百类。。。
        
    behaviorA = pd.get_dummies(behaviorA,columns=['var3','var5','var11','var12','var18'])
    behaviorB = pd.get_dummies(behaviorB,columns=['var3','var5','var11','var12','var18'])
    return behaviorA,behaviorB



## 降维
from sklearn.decomposition import PCA
from sklearn import preprocessing

def pca(data,n1=19,n2=326,n_comp=10):#第5~第326的列（也就是高度缺失的那些）做主成分、选10个主成分
    X = data.ix[:,n1:n2]
    X_scaled = preprocessing.scale(X)
    pca = PCA(n_comp)
    newData = pca.fit_transform(X_scaled)
#    newData.shape
    newData = pd.DataFrame(newData)
#    print(pca.explained_variance_ratio_)  
    print(sum(pca.explained_variance_ratio_[:n_comp]))#累积贡献率达70%
    data_PCA = pd.concat([data.ix[:,:n1],data.ix[:,n2:],newData],axis=1)
#    data_PCA.shape 
    return data_PCA

#behaviorA_pca = pca(behaviorA)


#SVD降维
from sklearn.decomposition import TruncatedSVD
def svd(data,n1=5,n2=326,n_comp=10):
    X = data.ix[:,n1:n2]
    svd = TruncatedSVD(n_components=n_comp, n_iter=7, random_state=42)
    newData = svd.fit_transform(X)
#    newData.shape
    newData = pd.DataFrame(newData)
#    print(pca.explained_variance_ratio_)  
    print(svd.explained_variance_ratio_.sum())#累积贡献率达70%
    print(svd.singular_values_)
    data_SVD = pd.concat([data.ix[:,:n1],data.ix[:,n2:],newData],axis=1)
#    data_PCA.shape 
    return data_SVD