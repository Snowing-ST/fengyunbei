#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:27:44 2018

@author: situ
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


class FeatureEngineer:
    def __init__(self,dt):#bins with same count
#        self._Y = Y
        self._data = dt
        
    def cutData(self,var,bins):
        """连续变量离散化1:等数切分数据"""
        q_var = "q_"+var
        
        plot_data = self._data.loc[:,[var]].copy()
        if (len(plot_data[var].value_counts())>20) & (var not in ['addr_state']):#when group >20, catalog data

            bin_acc = bins
            while((q_var in plot_data.columns.tolist())==False):
                try:
                    plot_data[q_var] = pd.qcut(plot_data[var],bin_acc)
                except:
                    #print("can't cut into %s groups" %bin_acc)
                    if bin_acc > 1:
                        bin_acc = bin_acc -1
                        continue
                    else: break


            if(bin_acc==1):
                print("can't cut return uncutted data")
                bins = 1
                plot_data[q_var] = plot_data[var].copy()
            else:
                print("we have cut into %s groups" %bin_acc)
#                print(plot_data[q_var].value_counts())


        else:
            #print("catalog number is lower than 20, we do not re-organize data")
            plot_data[q_var] = plot_data[var].copy()

        return plot_data
    
    def discretize(self,var,bins):
        """连续变量离散化2:根据切分的数据区间标记1-10"""
        ###append NaN as new catalog
        group_data = self.cutData(var,bins)
        q_var = 'q_'+var
        q_var_category = q_var+'_category'
        group_data[q_var_category] = group_data[q_var].astype("category")
        #catalog as 1 2 3 4 5
        group_num = 1+group_data[q_var_category].describe()['unique']
        group_data[q_var_category].cat.categories = [str(i) for i in range(1,int(group_num))]
        new_cat = str(1+int(group_data[q_var_category].describe()['unique']))
        #print('new_cat is',new_cat)
        
        group_data[q_var_category] = group_data[q_var_category].cat.add_categories([new_cat])
        group_data[q_var_category][group_data[q_var_category].isnull()]= new_cat
        self._data[q_var_category] = group_data[q_var_category].copy()
        self._data[q_var_category] = self._data[q_var_category].astype("int")
#        self._data.drop(var,axis=1,inplace = True)
    
    def rank(self,num_var_list):
        """排序特征"""
        for var in num_var_list:
            self._data['r'+var] = self._data[var].rank(method='max')
    
    def n_discretize(self,discre_var_list):
        """连续变量离散化3:统计每个样本中，离散后的连续型变量中取值1-10的个数"""
        for i in range(1,self.bins+2):
            self._data['n'+str(i)] = (self._data[discre_var_list]==i).sum(axis=1)

    def null_rate(self):
        """每个样本的缺失个数"""
        self._data["n_null"]=self._data.apply(lambda x: sum(x.isnull()),axis = 1)
        self.discretize("n_null",5)
        
    def vis_null(self):
        """可视化每个样本的缺失个数"""
        if "n_null" in self._data.columns.tolist()==False:
            self.null_rate()
        t = self._data.n_null.values
        t.sort()
        x = range(len(t))
        plt.scatter(x,t,c='k')
        plt.show()
    
        