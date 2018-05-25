# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:33:38 2018

@author: Hannah.N
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import sys
import math
from MSG_read import extract_data,plot_compare,plot_scatter_fit, plot_stats,plot_single_map
from math import sin, cos, sqrt, atan2, radians
from scipy import stats
from pandas.plotting import table
#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score
pd.options.mode.chained_assignment = None  
pd.set_option('display.max_columns', 500)
#date = "20180520"

dates = ["20180515","20180516" ,"20180517","20180518","20180519","20180520",'20180521','20180522','20180523',"20180524"]

df_fires = pd.read_csv("./txt_data/"+dates[0]+"_fires")
df_pixels = pd.read_csv("./txt_data/"+dates[0]+"_pixels")
for date in dates[1:]:
    df_fires_day=pd.read_csv("./txt_data/"+date+"_fires")
    df_pixels_day = pd.read_csv("./txt_data/"+date+"_pixels")
    df_fires =  df_fires.append(df_fires_day,ignore_index=True) # = pd.concat([df_fires,df_fires_day],axis=0, join='outer')
    df_pixels = pd.concat([df_pixels,df_pixels_day],axis=0, join='outer')
    
df_fires = df_fires.sort_values(by=['cluster_no'],ascending=True)
m8_frp = df_fires.groupby(['SAT']).get_group(8)['summed_FRP']
m8_vza = df_fires.groupby(['SAT']).get_group(8)['mean_vza']
m11_frp = df_fires.groupby(['SAT']).get_group(11)['summed_FRP']
m11_vza = df_fires.groupby(['SAT']).get_group(11)['mean_vza']


date_time = df_fires.groupby(['SAT']).get_group(8)['DATE_TIME']
clusters = df_fires.groupby(['SAT']).get_group(8)['cluster_no']
pixels_m8 = df_pixels[df_pixels['SAT'] == 8 ].reset_index(drop=True)
pixels_m11 = df_pixels[df_pixels['SAT'] == 11 ].reset_index(drop=True)

fires_m8 = df_fires[df_fires['SAT'] == 8 ].reset_index(drop=True)
#print fires_m8
fires_m11 = df_fires[df_fires['SAT'] == 11 ].reset_index(drop=True)

#print fires_m8.duplicated('cluster_no')
#print pd.concat(g for _, g in fires_m11.groupby("cluster_no") if len(g) > 1)

fire_diff = abs(m8_frp.values[:] - m11_frp.values[:] )
vza_diff = abs(m8_vza.values[:] - m11_vza.values[:] )

#print len(vza_diff),len(fire_diff),len(clusters)

fire_data = pd.DataFrame({'DATE_TIME':date_time.values[:],'M8_summed_FRP':m8_frp.values[:],'M11_summed_FRP':m11_frp.values[:],'FRP_diff':fire_diff,'VZA_diff':vza_diff,'cluster_no':clusters  })

fire_data['vza_bin'] = np.where((fire_data.VZA_diff <= 10),
              'deg_10',
              np.where(np.logical_and(fire_data.VZA_diff > 10,fire_data.VZA_diff <= 20),
              'deg10_20',
              np.where(np.logical_and(fire_data.VZA_diff > 20,fire_data.VZA_diff <= 30),
              'deg20_30',
              np.where(np.logical_and(fire_data.VZA_diff > 30,fire_data.VZA_diff <= 40),
              'deg30_40',
              'deg40_'))))
  
upper_limit = 500
plot_tag = "_bounded_2_" + str(upper_limit)
fire_data = fire_data[(fire_data.M8_summed_FRP < upper_limit) ] 
fire_data = fire_data[(fire_data.M11_summed_FRP < upper_limit) ]


vza_bins =['deg_10','deg10_20','deg20_30','deg30_40','deg40_']

for bin in vza_bins:
    cluster_list = fire_data.groupby(['vza_bin']).get_group(bin)['cluster_no']
    
    cluster_bin = df_fires[df_fires['cluster_no'].isin(cluster_list)]
    #print df_fires[df_fires['cluster_no'] == 21201516.0 ]
    latitude = fires_m8[fires_m8['cluster_no'].isin(cluster_list)]['mean_lat'].values[:]
    #print cluster_bin.sort_values(by=['mean_long'],ascending=True)['mean_long']
    #print fire_data[fire_data['cluster_no'] == 21201516.0 ]
    longitude = fires_m8[fires_m8['cluster_no'].isin(cluster_list)]['mean_long'].values[:]
    #plot_single_map(latitude,longitude,("./Plots/"+bin + "clusters" +plot_tag),bin)


def fit_and_plot_bins(fire_data, m8_frp, m11_frp,  vza_diff):  
    m8_frp = fire_data['M8_summed_FRP']
    m11_frp = fire_data['M11_summed_FRP']
    vza_diff = fire_data['VZA_diff']  
    vza_max = vza_diff.max()
    bias_all = fire_data['FRP_diff'].mean()
    SD_all = fire_data['FRP_diff'].std()
    rmsd_all = sqrt(bias_all**2 + SD_all**2)
    fire_count_all = len(vza_diff)
    slope_all, intercept_all, r_value, p_value_all, std_err_all = stats.linregress(m8_frp,m11_frp)
    r2_all = r_value**2
    z = m8_frp*slope_all + intercept_all
    textstr1 = 'Fire count=%d \n slope=%.3f \n intercept=%.3f\n R$^{2}$=%.3f\n SE=%.3f'%(fire_count_all,slope_all, intercept_all, r2_all,std_err_all)
    textstr2 = 'Bias=%.3f \n  Scatter =%.3f \n RMSD=%.3f'%(bias_all, SD_all, rmsd_all)
    plot_scatter_fit(m8_frp,m11_frp,vza_diff,z,'all',('./Plots/FRP_VZA_colour'+plot_tag), textstr1,textstr2,vza_max)
    
    
    vza_bins =['all_deg','deg_10','deg10_20','deg20_30','deg30_40','deg40_']
    stats_f = pd.DataFrame({'VZA_BIN':vza_bins,'BIAS':bias_all,'SCATTER':SD_all,'RMSD':rmsd_all,'FIRE_COUNT':fire_count_all,'SLOPE':slope_all,'INTERCEPT':intercept_all, 'R2':r2_all,'SE':std_err_all})
    
    for bin in vza_bins[1:]:
            
        m8_frp = fire_data.groupby(['vza_bin']).get_group(bin)['M8_summed_FRP']
        m11_frp = fire_data.groupby(['vza_bin']).get_group(bin)['M11_summed_FRP']
        vza_diff = fire_data.groupby(['vza_bin']).get_group(bin)['VZA_diff']  
        fire_count = len(vza_diff)
        bias = fire_data.groupby(['vza_bin']).get_group(bin)['FRP_diff'].mean()
        SD = fire_data.groupby(['vza_bin']).get_group(bin)['FRP_diff'].std()
        rmsd = sqrt(bias**2 + SD**2)
        slope, intercept, r_value, p_value, std_err = stats.linregress(m8_frp,m11_frp)
        r2 = r_value**2
        stat_title = ['BIAS','SCATTER','RMSD','FIRE_COUNT','SLOPE','INTERCEPT','R2','SE'] ; stat_value = [bias,SD,rmsd,fire_count,slope,intercept,r2,std_err]
        for i in range(0,8,1):
            n = stats_f.VZA_BIN[stats_f.VZA_BIN == bin].index            
            stats_f.at[n,stat_title[i]] = stat_value[i]
        
        z = m8_frp*slope + intercept
        textstr1 ='Fire count=%d \n slope=%.3f \n intercept=%.3f\n R$^{2}$=%.3f\n SE=%.3f'%(fire_count,slope, intercept, r2,std_err)
        textstr2 = 'Bias=%.3f \n  Scatter =%.3f \n RMSD=%.3f'%(bias, SD, rmsd)
        plot_scatter_fit(m8_frp,m11_frp,vza_diff,z,bin,('./Plots/FRP_VZA_colour'+plot_tag), textstr1,textstr2,vza_max)
   
    return stats_f

    
    
    
stats_f = fit_and_plot_bins(fire_data, m8_frp, m11_frp,vza_diff)

labels = ['Bias','Scatter','RMSD']
plot_stats(stats_f['VZA_BIN'][:],stats_f['BIAS'][:],stats_f['SCATTER'][:],stats_f['RMSD'][:],'./Plots/Fire_statistics',('VZA_dif'+plot_tag),labels)

labels = ['Slope','R$^2$','Standard Error']
plot_stats(stats_f['VZA_BIN'][:],stats_f['SLOPE'][:],stats_f['R2'][:],stats_f['SE'][:],'./Plots/linear_fit_stats',('VZA_dif'+plot_tag),labels)
stats_f.to_csv('./fire_stats_table' +plot_tag +'.csv')












