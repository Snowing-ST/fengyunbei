#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:09:48 2018

@author: situ
"""

import os
import pandas as pd
import numpy as np
from FeatureEngineer import FeatureEngineer
pathA = "./data/train_scene_A/train_behavior_A.csv"
pathB = "./data/train_scene_B/train_behavior_B.csv"
os.chdir("/Users/situ/Documents")

def behavior_prepocess(pathA,pathB):
    
    behaviorA = pd.read_csv(pathA)
    behaviorB = pd.read_csv(pathB)

    print("begin to compute the missing rate......")
    #只有var1——var10的变量缺失值少于50%
    missing = np.array(behaviorA.apply(lambda x: sum(x.isnull())/len(behaviorA)))

#查看缺失90%以上的变量
#behaviorA.columns[missing>0.99]

#var3 var4 var6 var10存在少量缺失
#behaviorA.ix[:,:11].apply(lambda x: sum(x.isnull())/len(behaviorA))

#对于缺失太多的变量，缺失80%直接删
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
#var3	var4	var5	var6 很多类别的定性变量，看看是否有某几个类别占比多
#var2 var7	var8	var9为0-1 看0-1的比例，比例相差过大就删
#var10 除6.8%的缺失值外，其他都是1，删除

    print("begin to deal with var1-var10......")
    behavior = [behaviorA,behaviorB]
    for b in behavior:
        b["q_var1"] = 1
        b["q_var1"][b["var1"]>29]=2
        b["q_var1"][b["var1"]>39]=3
    
#    behaviorA["q_var1"].value_counts()
    
#    def topNpop(series,n=10):
#        v = np.array(series.value_counts())
#        return sum(v[np.argsort(-v)][:n])/sum(v)
#    
#    #var4 var6 类别多且不集中，直接删掉
#    for b in behavior:
#        var3_6 = b.ix[:,["var3","var4","var5","var6"]]
#        print(var3_6.apply(lambda x:topNpop(x,10)))
    
    
    #var7	var8	var9中，0或1都占了88%以上。。还是不删了，删了就没变量了。。。
#    for b in behavior:
#        var2_9 = b.ix[:,["var2","var7","var8","var9"]]
#        print(var2_9.apply(lambda x:topNpop(x,1)))
        
    for b in behavior:
        b.drop(["var1","var4","var6"],axis=1,inplace=True)
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
    
    return behaviorA,behaviorB


behaviorA,behaviorB = behavior_prepocess(pathA,pathB) 


#调用类里的函数
os.chdir("/Users/situ/Desktop/风云杯")
from FeatureEngineer import FeatureEngineer
fe = FeatureEngineer(behaviorA)
fe.null_rate()
fe.vis_null()

behaviorA["q_n_null_category"].value_counts(sort = False)

var = "n_null"
fe.cutData(var,bins=5)["q_"+var].value_counts()

fe.discretize(var,bins=10)
behaviorA["q_"+var+'_category'].value_counts()

