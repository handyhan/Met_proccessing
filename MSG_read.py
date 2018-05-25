from matplotlib import rcParams
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib as mpl
import matplotlib.colors as colors
import h5py
import numpy as np
import pandas as pd
import sys
import math
import os


def roundup(x):
    return int(math.ceil(x / 100.0)) * 100


# Get hdf FRP data into pandas #



def extract_data(date, time):
    #print date
    m8 = "./Data/NetCDF_LSASAF_MSG-IODC_FRP-PIXEL-ListProduct_IODC-Disk_" + date+time
    m11 = "./Data/HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_MSG-Disk_" + date +time
    
    if os.path.exists(m8) and os.path.exists(m11):
        m8_f = h5py.File(m8,'r+')
        m11_f = h5py.File(m11,'r+') 
        m8_d = {'LATITUDE':m8_f['LATITUDE'][:],'LONGITUDE':m8_f['LONGITUDE'][:],'FRP':m8_f['FRP'][:],'FRP_UNCERTAINTY':m8_f['FRP_UNCERTAINTY'][:],'PIXEL_SIZE':m8_f['PIXEL_SIZE'][:],'PIXEL_VZA':m8_f['PIXEL_VZA'][:]}
        m11_d = {'LATITUDE':m11_f['LATITUDE'][:],'LONGITUDE':m11_f['LONGITUDE'][:],'FRP':m11_f['FRP'][:],'FRP_UNCERTAINTY':m11_f['FRP_UNCERTAINTY'][:],'PIXEL_SIZE':m11_f['PIXEL_SIZE'][:],'PIXEL_VZA':m11_f['PIXEL_VZA'][:]}
        m8_df = pd.DataFrame(data=m8_d,dtype=np.float32)
        m11_df = pd.DataFrame(data=m11_d,dtype=np.float32)
        m8_df['DATE']=date ; m11_df['DATE']=date; m8_df['TIME']=time; m11_df['TIME']=time
        cols = ['DATE','TIME','LATITUDE','LONGITUDE','FRP','FRP_UNCERTAINTY','PIXEL_SIZE','PIXEL_VZA']
        m8_df = m8_df[cols]
        m11_df = m11_df[cols]    
        # account for scaling factors ect #
        m8_df['LATITUDE'] = m8_df['LATITUDE']/100 ; m11_df['LATITUDE'] = m11_df['LATITUDE']/100
        m8_df['LONGITUDE'] = m8_df['LONGITUDE']/100 ; m11_df['LONGITUDE'] = m11_df['LONGITUDE']/100
        m8_df['FRP'] = m8_df['FRP']/10 ; m11_df['FRP'] = m11_df['FRP']/10
        m8_df['FRP_UNCERTAINTY'] = m8_df['FRP_UNCERTAINTY']/100 ; m11_df['FRP_UNCERTAINTY'] = m11_df['FRP_UNCERTAINTY']/100
        m8_df['PIXEL_SIZE'] = m8_df['PIXEL_SIZE']/100 ; m11_df['PIXEL_SIZE'] = m11_df['PIXEL_SIZE']/100
        m8_df['PIXEL_VZA'] = m8_df['PIXEL_VZA']/100 ; m11_df['PIXEL_VZA'] = m11_df['PIXEL_VZA']/100
        return m8_df, m11_df
    elif not os.path.exists(m8):
        print "ERROR when trying to open file, this file does not exist:" + m8    
        m8_df, m11_df = [pd.DataFrame(),pd.DataFrame()]
        return m8_df, m11_df
    elif not os.path.exists(m11):
        print "ERROR when trying to open file, this file does not exist:" + m11    
        m8_df, m11_df = [pd.DataFrame(),pd.DataFrame()]
        return m8_df, m11_df

    


def plot_pixel_size(plot_list):
    
    data_name= ['Meteosat-8','Metiosat-11']
    max_frp=np.max([np.max(m8_df.FRP),np.max(m11_df.FRP)])
    graph_max=roundup(max_frp)
    print graph_max

    fig = plt.figure()    
    m8_latitude = m8_df['LATITUDE'].values; m11_latitude = m11_df['LATITUDE'].values
    m8_longitude = m8_df['LONGITUDE'].values; m11_longitude = m11_df['LONGITUDE'].values
    m8_p_angle = m8_df['PIXEL_VZA'].values  ; m11_p_angle = m11_df['PIXEL_VZA'].values 
    m8_p_size = m8_df['PIXEL_SIZE'].values; m11_p_size = m11_df['PIXEL_SIZE'].values

    ax = fig.add_subplot(121)
    m8_z = np.polyfit(m8_p_angle,m8_p_size,3)
    m8_f = np.poly1d(m8_z)   
    m8_x_new = np.linspace(m8_p_angle[0], m8_p_size[-1], 50)
    m8_y_new = m8_f(m8_x_new)
    ax.plot(m8_p_angle,m8_p_size,'o', m8_x_new, m8_y_new)
    #ax.scatter(m8_p_angle,m8_p_size,s = 50, alpha = 0.8, cmap='hot',marker ='.')
    ax = fig.add_subplot(122)
    #ax.scatter(m11_p_angle,m11_p_size,s = 50, alpha = 0.8, cmap='hot',marker ='.') 
    
    m11_z = np.polyfit(m11_p_angle,m11_p_size,3)
    m11_f = np.poly1d(m11_z)   
    m11_x_new = np.linspace(m11_p_angle[0], m11_p_size[-1], 50)
    m11_y_new = m11_f(m11_x_new)
    print m8_z, m11_z
    ax.plot(m11_p_angle,m11_p_size,'o', m11_x_new, m11_y_new)



    plt.savefig('pixel_frp_plot' + date_time, dpi = 500)
    plt.close()
    
#plot_pixel_size(plot_list)
    

def plot_single_map(latitude,longitude,title,bin_no):
    #formated_dt = date_time[0:4]+'-'+date_time[4:6]+'-'+date_time[6:8]+'_'+date_time[8:10]+':'+ date_time[10:12] +':00'    
    fig = plt.figure()
    #latitude = df['LATITUDE'].values
    #longitude = df['LONGITUDE'].values
    #FRP = df['FRP'].values
      
    ax = fig.add_subplot(111)
    m_m8 = Basemap(llcrnrlon=-30,llcrnrlat=-35,urcrnrlon=60,urcrnrlat=40,
                                    resolution='i',projection='tmerc',lon_0=15,lat_0=0,ax=ax)
    #m_m8 = Basemap(llcrnrlon=28,llcrnrlat=8,urcrnrlon=32,urcrnrlat=10,
    #                                resolution='i',projection='tmerc',lon_0=30,lat_0=9,ax=ax)   
    m_m8.drawcoastlines(linewidth = 0.5)
    m_m8.drawmapboundary()
    #m.fillcontinents(lake_color='aqua',zorder=0)
    m_m8.drawparallels(np.arange(-30,60.,20), labels = [True],fontsize=5 )
    m_m8.drawmeridians(np.arange(-40.,100.,20))
    #m_m8.drawcountries(linewidth=0.25)
    x, y = m_m8(longitude, latitude)
    sc = plt.scatter(x, y, c='orangered', s = 8, alpha = 0.4, cmap='hot',marker =',',linewidth=0.0),
                            #zorder=1,norm=mpl.colors.SymLogNorm(linthresh=2, vmin=0, vmax=1000))
    plt.title(bin_no,fontsize=9)
    #cb = m_m8.colorbar(sc, pad=0.001, ticks = [0, 1000] )#location='bottom')
    #cb.ax.tick_params(labelsize=8)
    #cb.ax.set_ylabel('FRP (MW)',fontsize=9)
    #cb.ax.set_yticklabels([])
    #cb.set_ticks([0,200,400,600,800,1000,1200,1400])         # need editting to be more case adaptive range
    #cb.ax.set_ylabel('FRP',fontsize=9)

    plt.savefig(title, dpi = 500)
    plt.close()


def plot_compare(plot_list,date_time):
    formated_dt = date_time[0:4]+'-'+date_time[4:6]+'-'+date_time[6:8]+'_'+date_time[8:10]+':'+ date_time[10:12] +':00'    
    m8_df,m11_df = plot_list
    fig = plt.figure()
    latitude = m8_df['LATITUDE'].values
    longitude = m8_df['LONGITUDE'].values
    FRP = m8_df['FRP'].values
      
    ax = fig.add_subplot(121)
    m_m8 = Basemap(llcrnrlon=-30,llcrnrlat=-35,urcrnrlon=60,urcrnrlat=40,
                                    resolution='i',projection='tmerc',lon_0=15,lat_0=0,ax=ax)
    #m_m8 = Basemap(llcrnrlon=28,llcrnrlat=8,urcrnrlon=32,urcrnrlat=10,
    #                                resolution='i',projection='tmerc',lon_0=30,lat_0=9,ax=ax)   
    m_m8.drawcoastlines(linewidth = 0.5)
    m_m8.drawmapboundary()
    #m.fillcontinents(lake_color='aqua',zorder=0)
    m_m8.drawparallels(np.arange(-35,60.,20), labels = [True],fontsize=5 )
    m_m8.drawmeridians(np.arange(-40.,100.,20))
    #m_m8.drawcountries(linewidth=0.25)
    x, y = m_m8(longitude, latitude)
    sc = plt.scatter(x, y, c=FRP, s = 10, alpha = 0.8, cmap='hot',marker =',',
                            zorder=1,norm=mpl.colors.SymLogNorm(linthresh=2, vmin=0, vmax=1000))
    plt.title('Meteosat-8',fontsize=9)
    cb = m_m8.colorbar(sc, pad=0.001, ticks = [0, 1000] )#location='bottom')
    cb.ax.tick_params(labelsize=8)
    cb.ax.set_yticklabels([])
    #cb.set_ticks([0,200,400,600,800])         # need editting to be more case adaptive range
    #cb.ax.set_ylabel('FRP',fontsize=9)


    latitude = m11_df['LATITUDE'].values
    longitude = m11_df['LONGITUDE'].values
    FRP = m11_df['FRP'].values
    
    ax = fig.add_subplot(122)
    m_m11 = Basemap(llcrnrlon=-30,llcrnrlat=-35,urcrnrlon=60,urcrnrlat=40,
                                    resolution='i',projection='tmerc',lon_0=15,lat_0=0,ax=ax)
    #m_m11 = Basemap(llcrnrlon=28,llcrnrlat=8,urcrnrlon=32,urcrnrlat=10,
    #                                resolution='i',projection='tmerc',lon_0=30,lat_0=9,ax=ax)    
    m_m11.drawcoastlines(linewidth = 0.5)
    m_m11.drawmapboundary()
    #m.fillcontinents(lake_color='aqua',zorder=0)
    m_m11.drawparallels(np.arange(-35,60.,20), labels = [True],fontsize=5 )
    m_m11.drawmeridians(np.arange(-40.,100.,20),labels = [True])
    #m_m11.drawcountries(linewidth=0.25)
    # m_ice.shadedrelief()
    x, y = m_m11(longitude, latitude)
    sc = plt.scatter(x, y, c=FRP, s = 10, alpha = 0.8, cmap='hot',marker =',',
                            zorder=1,norm=mpl.colors.SymLogNorm(linthresh=2, vmin=0, vmax=1000))
    plt.title('Meteosat-11',fontsize=9)
    plt.suptitle('FRP-PIXEL on ' + formated_dt ,fontsize=12 , y=0.09)    
    #fig.subplots_adjust(right=0.8)
    rcParams['figure.subplot.wspace'] = 0.2 # more width between subplots
    #cb_ax = fig.add_axes([0, 800, 0,800])
    cb = m_m11.colorbar(sc,  pad=0.001)#location='bottom')
    cb.ax.tick_params(labelsize=8)
    #cb.ax.set_yticklabels([0,10,50,100,200,500,1000])         # need editting to be more case adaptive range
    cb.ax.set_ylabel('FRP (MW)',fontsize=9)
    #fig.tight_layout()
    plt.savefig('FRP_cluster_' + date_time, dpi = 500)
    plt.close()

def plot_scatter_fit(x,y,colour,z, date, title,textstr1,textstr2,c_max): # 
    fig = plt.figure(figsize = [4,4])
    ax1 = fig.add_subplot(111)
    fire_max = round((max(x.max(),y.max())) / 500.0) * 500.0
    #print m1,c1,m2,c2
    #ax1.plot([w, z], [w, z],'k-', color = 'r')
    sc = ax1.scatter(x,y, c=colour, alpha = 0.5,vmin=0,vmax=c_max,linewidth=0.0)
    ax1.set_ylabel('M11 FRP (MW) ',fontsize=9)
    ax1.set_xlabel('M8 FRP (MW)',fontsize=9)
    cb = plt.colorbar(sc)
    cb.ax.set_ylabel('Absolute VZA difference (Degrees)',fontsize=9)
    plt.subplots_adjust(top = 0.85)
    ax1.plot(x,z,color='r')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.xticks(np.arange(0, fire_max+1, step=250))
    plt.yticks(np.arange(0, fire_max+1, step=250))
    plt.setp(ax1.get_xticklabels(), rotation=40, horizontalalignment='right')
    ax1.text(0.45,0.95,textstr1, transform=ax1.transAxes, fontsize=8,
        va='top',ha='right')# bbox=props)
    plt.suptitle(textstr2,fontsize=10)
    plt.tight_layout()
    plt.savefig(title+'_scatter_'   + date, dpi = 800)
    plt.close()
    #ax1.scatter(m11_frp,m11_vza,c='b',)

    
def plot_stats(x,y,z,w,title1, title2, labels): # 
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    label_1, label_2, label_3 = labels
    #print m1,c1,m2,c2
    #ax1.plot([w, z], [w, z],'k-', color = 'r')
    #sc = ax1.scatter(x,y, c=colour, alpha = 0.5,vmin=0,vmax=c_max)
    #ax1.set_ylabel('FRP (MW) ',fontsize=9)
    ax1.set_xlabel('Absolute VZA difference Bin',fontsize=9)
    #plt.yticks(np.arange(0, 1, step=0.1))
    #cb = plt.colorbar(sc)
    #cb.ax.set_ylabel('Absolute VZA difference (Degrees)',fontsize=9)
    #plt.subplots_adjust(top = 0.85)
    ax1.plot(x,y,color='dodgerblue', label = label_1)
    ax1.plot(x,z,color='darkorange', label = label_2)
    ax1.plot(x,w,color='forestgreen', label = label_3)
    #ax1.plot(x,w,color='palevioletred', label = 'Standard Error')
    ax1.legend()
    #ax1.text(0.3,0.95,textstr1, transform=ax1.transAxes, fontsize=8,
    #   va='top',ha='right')# bbox=props)
    plt.suptitle('Statistics on linear fit binned by their ' + title2 ,fontsize=10)
    #ax1 = fig.add_subplot(122)
    #ax1.plot(np.unique(w), np.poly1d(np.polyfit(w,z, 1))(np.unique(w)))
    #ax1.scatter(w,z, c='r')
    #ax1.set_ylabel('VZA M11',fontsize=9)
    #ax1.set_xlabel('VZA M8',fontsize=9)
    plt.savefig(title1+ title2 , dpi = 700)
    plt.close()
    #ax1.scatter(m11_frp,m11_vza,c='b',)

    'Slope','R$^2$','Standard Error'
    
    #ax1 = fire_stats.plot.scatter(x=m8_frp,y=m11_frp)




