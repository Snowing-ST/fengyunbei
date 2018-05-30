# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:03:31 2018

@author: Bokkin Wang
"""

import pandas as pd
import numpy as np

class consumer(object):

    def __init__(self,path):
        f= open(path)
        self.consumer = pd.read_csv(f)
        self.dis_data=self.consumer[['ccx_id','V_1','V_2','V_8','V_14']]
        self.date_data = self.consumer[["ccx_id","V_7","V_11"]]
        
    def discrete(self): 
        index1=self.dis_data['V_1'].value_counts().index[:10]
        index8=self.dis_data['V_8'].value_counts().index[:5]
        index14=self.dis_data['V_14'].value_counts().index[:10]
        self.dis_data["V_8"].fillna('none',inplace=True) 
        self.dis_data["V_14"].fillna('none',inplace=True)
        self.dis_data['V_1']=[i if i in index1 else 'else' for i in self.dis_data['V_1']]
        self.dis_data['V_8']=[i if i in index8 else 'else' for i in self.dis_data['V_8']]
        self.dis_data['V_14']=[i if i in index14 else 'else' for i in self.dis_data['V_14']]
        
        group1=self.dis_data.groupby("ccx_id")['V_2'].count().reset_index()
        group1.rename(columns={'V_2':'consumer_count'}, inplace = True)
        data=pd.get_dummies(self.dis_data,columns=['V_1','V_8','V_14'])
        group2=data.groupby("ccx_id").sum().reset_index()
        group=pd.merge(group1,group2)
        return group
    
    def continuous(self):
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
        
        self.consumer['V_12_none']=self.consumer['V_12']
        self.consumer['V_13_none']=self.consumer['V_13']
        self.consumer["V_12_none"].fillna('none',inplace=True) 
        self.consumer["V_13_none"].fillna('none',inplace=True)   
        self.consumer['V_12_none']=list(map(lambda x: 1 if  x=='none' else 0, self.consumer["V_12_none"]))
        self.consumer['V_13_none']=list(map(lambda x: 1 if  x=='none' else 0, self.consumer["V_13_none"]))
        group=self.consumer.groupby("ccx_id")["V_4","V_5","V_6","V_9","V_10","V_12","V_13","V_12_none","V_13_none"].mean()
        group=group.reset_index()
        return group
        
            
    def date(self):
        #    date_var['nowtime']=dt.datetime(2018,5,22)               #加入当前时间
        ##发生交易的结束时间V7_hour变量
        self.date_data["V7_date"]=[item.split()[0] for item in self.date_data['V_7']]
        self.date_data["V7_hour"]=[int(item.split()[1].split(':')[0]) for item in self.date_data['V_7']]
        self.date_data['V7_period']='night'
        self.date_data['V7_period'][(self.date_data['V7_hour']>=6) & (self.date_data['V7_hour']<12)]='morning'
        self.date_data['V7_period'][(self.date_data['V7_hour']>=12) & (self.date_data['V7_hour']<=18)]='afternoon'
        #    date_data['V7_hour'][(date_data['V7_hour']>=18) | (date_data['V7_hour']<6)]='night'
        #    date_data["V7_hour"]=[str(item) for item in date_data['V7_hour']]
        
        ##发生交易的起始时间V11_hour变量    
        self.date_data['V_11'][self.date_data['V_11']=='0000-00-00 00:00:00'] = '2010-01-01 25:00:00'
        self.date_data["V11_date"]=[item.split()[0] for item in self.date_data['V_11']]
        self.date_data["V11_hour"]=[int(item.split()[1].split(':')[0]) for item in self.date_data['V_11']]
        self.date_data['V11_period']='night'
        self.date_data['V11_period'][self.date_data['V11_hour']>24]='none'
        self.date_data['V11_period'][(self.date_data['V11_hour']>=6) & (self.date_data['V11_hour']<12)]='moring'
        self.date_data['V11_period'][(self.date_data['V11_hour']>=12) & (self.date_data['V11_hour']<=18)]='afternoon'
        #    date_data['V11_period'][((date_data['V11_hour']>=18) | (date_data['V11_hour']<6))]='night'
        #    date_data["V11_hour"]=[str(item) for item in date_data['V11_hour']]
        
        
        ##交易时长    
        self.date_data['V_11'][self.date_data['V_11']=='2010-01-01 25:00:00'] = '2010-01-01 00:00:00'
        self.date_data['V_7']=pd.to_datetime(self.date_data['V_7'])
        self.date_data['V_11']=pd.to_datetime(self.date_data['V_11'])
        ## 时间间隔
        self.date_data['time_interval']=self.date_data['V_7']-self.date_data['V_11']
        self.date_data['days_interval']=[i.days for i in self.date_data['time_interval']]
        self.date_data['seconds_interval']=[i.total_seconds() for i in self.date_data['time_interval']]
        ## 异常值处理
        self.date_data['seconds_interval'][(self.date_data['days_interval']>100) | (self.date_data['days_interval']<0)]=np.nan
        
        data1=pd.get_dummies(self.date_data[['ccx_id','V7_period','V11_period']],columns=['V7_period','V11_period'])
        group1=data1.groupby("ccx_id").mean().reset_index()
        group21 = self.date_data.groupby("ccx_id")['seconds_interval'].mean().reset_index()
        group22 = self.date_data.groupby("ccx_id")['seconds_interval'].median().reset_index()
        group2=pd.merge(group21,group22)
        group=pd.merge(group1,group2)
        
        return group
    
    def consumer_pre(self):
        group1=self.discrete()
        group2=self.continuous()
        group3=self.date()
        ###
        data=pd.merge(group1,group2)
        data=pd.merge(data,group3)
        return data
    
    @property
    def show_me(self):
        print('需要处理的数据的前5行为：')
        print(self.consumer.head())
        print('需要处理的数据的行和列为：')
        print(self.consumer.shape)


if __name__ == '__main__':
   
    h=consumer("D:/bigdatahw/风云杯/data/选手建模数据/train_scene_A/train_consumer_A.csv")
    h.show_me
    mydata=h.consumer_pre()

    print(mydata.shape)
    mydata.head()





