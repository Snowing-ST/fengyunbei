# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:52:22 2018

@author: steven.wang
"""

import pandas as pd
import os

def ccx_pre(data):
    index2=data['var_02'].value_counts().index[:8]
    index3=data['var_03'].value_counts().index[:5]
    index4=data['var_04'].value_counts().index[:5]
    index5=data['var_05'].value_counts().index[:4]
    data['var_02']=[i if i in index2 else 'else' for i in data['var_02']]
    data['var_03']=[i if i in index3 else 'else' for i in data['var_03']]
    data['var_04']=[i if i in index4 else 'else' for i in data['var_04']]
    data['var_05']=[i if i in index5 else 'else' for i in data['var_05']]
    data2=pd.get_dummies(data,columns=['var_01','var_02','var_03','var_04','var_05'])
    group=data2.groupby("ccx_id")
    group1=group.sum().reset_index()
    group2=group['var_06'].count().reset_index()
    group2.rename(columns={'var_06':'ccx_count'}, inplace = True)
    group_all=pd.merge(group1,group2)
    return group_all


