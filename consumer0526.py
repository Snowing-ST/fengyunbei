# -*- coding: utf-8 -*-
"""
Created on Wed May 23 18:44:10 2018

@author: Bokkin Wang
"""

import pandas as pd
import os
from datetime import datetime
import numpy as np
#import matplotlib.pyplot as plt

#os.chdir("C:/Users/steven.wang/Desktop/风云杯/data")
#consumer = pd.read_csv("train_scene_A/train_consumer_A.csv")

def discrete(dis_data): 
    index1=dis_data['V_1'].value_counts().index[:10]
    index8=dis_data['V_8'].value_counts().index[:5]
    index14=dis_data['V_14'].value_counts().index[:10]
    dis_data["V_8"].fillna('none',inplace=True) 
    dis_data["V_14"].fillna('none',inplace=True)
    dis_data['V_1']=[i if i in index1 else 'else' for i in dis_data['V_1']]
    dis_data['V_8']=[i if i in index8 else 'else' for i in dis_data['V_8']]
    dis_data['V_14']=[i if i in index14 else 'else' for i in dis_data['V_14']]
    
    group1=dis_data.groupby("ccx_id")['V_2'].count().reset_index()
    group1.rename(columns={'V_2':'consumer_count'}, inplace = True)
    data=pd.get_dummies(dis_data,columns=['V_1','V_8','V_14'])
    group2=data.groupby("ccx_id").sum().reset_index()
    group=pd.merge(group1,group2)
    return group

def continuous(consumer):
#############     异常值处理
#    consumer['V_4'][consumer['V_4']>200]=200
#    consumer['V_5'][consumer['V_5']>10000]=10000
#    consumer['V_5'][consumer['V_5']<-15]=-15
#    consumer['V_6'][consumer['V_6']>10000]=10000
#    consumer['V_9'][consumer['V_9']>500]=500
#    consumer['V_10'][consumer['V_10']>1500]=1500
#    consumer['V_10'][consumer['V_10']<-1000]=-1000
#    consumer['V_12'][consumer['V_12']>500]=500
#    consumer['V_13'][consumer['V_13']<-0]=0
#    consumer['V_13'][consumer['V_13']>1000]=1000

    consumer['V_12_none']=consumer['V_12']
    consumer['V_13_none']=consumer['V_13']
    consumer["V_12_none"].fillna('none',inplace=True) 
    consumer["V_13_none"].fillna('none',inplace=True)   
    consumer['V_12_none']=list(map(lambda x: 1 if  x=='none' else 0, consumer["V_12_none"]))
    consumer['V_13_none']=list(map(lambda x: 1 if  x=='none' else 0, consumer["V_13_none"]))
    group1=consumer.groupby("ccx_id")["V_4","V_5","V_6","V_9","V_10","V_12","V_13","V_12_none","V_13_none"].mean()
    group1=group1.reset_index()
    #group2=consumer.groupby("ccx_id")["V_12_none","V_13_none"].agg(['min','max'])
    #group2=group2.reset_index()
    #group=pd.merge(group1,group2,on='ccx_id')
    return group1

    
def date(date_data):
#    date_var['nowtime']=dt.datetime(2018,5,22)               #加入当前时间
##发生交易的结束时间V7_hour变量
    date_data["V7_date"]=[item.split()[0] for item in date_data['V_7']]
    date_data["V7_hour"]=[int(item.split()[1].split(':')[0]) for item in date_data['V_7']]
    date_data['V7_period']='night'
    date_data['V7_period'][(date_data['V7_hour']>=6) & (date_data['V7_hour']<12)]='morning'
    date_data['V7_period'][(date_data['V7_hour']>=12) & (date_data['V7_hour']<=18)]='afternoon'
#    date_data['V7_hour'][(date_data['V7_hour']>=18) | (date_data['V7_hour']<6)]='night'
#    date_data["V7_hour"]=[str(item) for item in date_data['V7_hour']]
    
##发生交易的起始时间V11_hour变量    
    date_data['V_11'][date_data['V_11']=='0000-00-00 00:00:00'] = '2010-01-01 25:00:00'
    date_data["V11_date"]=[item.split()[0] for item in date_data['V_11']]
    date_data["V11_hour"]=[int(item.split()[1].split(':')[0]) for item in date_data['V_11']]
    date_data['V11_period']='night'
    date_data['V11_period'][date_data['V11_hour']>24]='none'
    date_data['V11_period'][(date_data['V11_hour']>=6) & (date_data['V11_hour']<12)]='moring'
    date_data['V11_period'][(date_data['V11_hour']>=12) & (date_data['V11_hour']<=18)]='afternoon'
#    date_data['V11_period'][((date_data['V11_hour']>=18) | (date_data['V11_hour']<6))]='night'
#    date_data["V11_hour"]=[str(item) for item in date_data['V11_hour']]


##交易时长    
    date_data['V_11'][date_data['V_11']=='2010-01-01 25:00:00'] = '2010-01-01 00:00:00'
    date_data['V_7']=pd.to_datetime(date_data['V_7'])
    date_data['V_11']=pd.to_datetime(date_data['V_11'])
    ## 时间间隔
    date_data['time_interval']=date_data['V_7']-date_data['V_11']
    date_data['days_interval']=[i.days for i in date_data['time_interval']]
    date_data['seconds_interval']=[i.total_seconds() for i in date_data['time_interval']]
    ## 异常值处理
    date_data['seconds_interval'][(date_data['days_interval']>100) | (date_data['days_interval']<0)]=np.nan

    data1=pd.get_dummies(date_data[['ccx_id','V7_period','V11_period']],columns=['V7_period','V11_period'])
    group1=data1.groupby("ccx_id").mean().reset_index()
    group21 = date_data.groupby("ccx_id")['seconds_interval'].mean().reset_index()
    group22 = date_data.groupby("ccx_id")['seconds_interval'].median().reset_index()
    group2=pd.merge(group21,group22,on='ccx_id')
    group=pd.merge(group1,group2)

    return group


##############################主函数##########################################
def consu_pre(consumer):
    dis_data=consumer[['ccx_id','V_1','V_2','V_8','V_14']]
    date_data = consumer[["ccx_id","V_7","V_11"]]
    group1=discrete(dis_data)
    group2=continuous(consumer)
    group3=date(date_data)
    ###
    data=pd.merge(group1,group2,on='ccx_id')
    data=pd.merge(data,group3)
    return data
