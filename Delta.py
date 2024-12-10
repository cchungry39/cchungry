#!/usr/bin/env python
# coding: utf-8

# In[12]:


import xarray as xr
import os,glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall
from sklearn.metrics import r2_score,mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# 设定工作目录到存储数据的路径
os.chdir("D:\Program\Results")

# 读取气候模型数据
ds_pr_gcm_raw = xr.open_dataset("D:\Program\CanESM5\pr_day_CanESM5_historical_r1i1p1f1_gn_18500101-20141231.nc")
ds_pr_gcm_raw = ds_pr_gcm_raw.sel(time=ds_pr_gcm_raw.time.dt.year.isin(range(1961, 2015)))

# 读取观测数据（我把观测数据也放在了gcm同路径，为了读取方便）
ds_pr_obs_raw = xr.open_dataset("D:\Program\CN.05\CN05.1_Pre_1961_2018_daily_025x025.nc")
ds_pr_obs_19792014 = ds_pr_obs_raw.sel(time=ds_pr_obs_raw.time.dt.year.isin(range(1961, 2015)))

# 读取外部CSV文件中的经纬度数据（我把csv放在gcm同路径，为了读取方便）
lat_lon_df = pd.read_csv('D:\Program\Point\Point.csv')

# 循环处理每个经纬度点
for index, row in lat_lon_df.iterrows():
    lon, lat = row['lon'], row['lat']

    # 插值气候模型数据
    ds_pr_gcm = ds_pr_gcm_raw.interp(lon=lon, lat=lat)

    # 插值观测数据
    ds_pr_obs = ds_pr_obs_19792014.interp(lon=lon, lat=lat)
    
    # 计算月均值和偏差
    ds_pr_obs_monthlymean=ds_pr_obs.resample(time="1M").mean().pre
    ds_pr_gcm=ds_pr_gcm.pr*86400
    #这里就是针对缺失闰年的模式的处理方法，把time=1M改为time=MS
    ds_pr_gcm_monthlymean=ds_pr_gcm.resample(time="MS").mean()
    ds_pr_obs_multimonthlymean=ds_pr_obs_monthlymean.groupby(ds_pr_obs_monthlymean.time.dt.month).mean()
    ds_pr_gcm_multimonthlymean=ds_pr_gcm_monthlymean.groupby(ds_pr_gcm_monthlymean.time.dt.month).mean()
    delta_pr=ds_pr_gcm_monthlymean.groupby('time.month').mean()/ds_pr_obs_monthlymean.groupby('time.month').mean()
    delta_pr=delta_pr.values.squeeze()
    
    # 降尺度操作
    result=[] # 利用list 保存结果
    for i in range(1,13)[:]:
        tmp_pr=ds_pr_gcm.sel(time=ds_pr_gcm.time.dt.month==i) # 通过sel筛选time，选择月份
        gcm_pr_downscaled=tmp_pr*delta_pr[i-1] # delta是数组，所以0-11，i-1
        result.append(gcm_pr_downscaled)
    gcm_pr_downscaled_final=xr.merge(result)

    # 构建文件名
    model_name = "CanESM5pr"  # 举例，根据实际模型名字修改
    first_date = str(gcm_pr_downscaled_final.time.values[0])[:10]  # 获取第一个时间变量的日期并转为字符串
    file_name = f"{lon}_{lat}_{model_name}_{first_date}his.nc"  # 格式化文件名

    # 将降尺度结果保存为NetCDF文件
    gcm_pr_downscaled_final.to_netcdf(file_name)

    # 打印进度
    print(f"Processed {lon}, {lat}; Saved as {file_name}")


# In[14]:
# In[13]:


#需要手动修改，导入新的gcm数据，建议结合wps批量一键替换
df_pr_gcm_raw1=xr.open_dataset("D:\Program\CanESM5\pr_day_CanESM5_ssp126_r1i1p1f1_gn_20150101-21001231.nc")
df_pr_gcm_raw1=df_pr_gcm_raw1.sel(time=df_pr_gcm_raw1.time.dt.year.isin(range(2015,2101)))


# 读取外部CSV文件中的经纬度数据（我把csv放在gcm同路径，为了读取方便）
lat_lon_df = pd.read_csv('D:\Program\Point\Point.csv')

# 循环处理每个经纬度点
for index, row in lat_lon_df.iterrows():
    lon, lat = row['lon'], row['lat']
    
    # 插值气候模型数据
    df_pr_gcm1 = df_pr_gcm_raw1.interp(lon=lon, lat=lat) .pr*86400


    result1=[]
    for i in range(1,13)[:]:
        tmp_pr=df_pr_gcm1.sel(time=df_pr_gcm1.time.dt.month==i)
        gcm_pr_downscaled1=tmp_pr*delta_pr[i-1]
        result1.append(gcm_pr_downscaled1)
    gcm_pr_downscaled1_final=xr.merge(result1)

    #根据实际模型名字修改
    model_name = "CanESM5pr"
    first_date = str(gcm_pr_downscaled1_final.time.values[0])[:10]
    file_name = f"{lon}_{lat}_{model_name}_{first_date}_ssp126.nc"
    gcm_pr_downscaled1_final.to_netcdf(file_name)
    # 打印进度
    print(f"Processed {lon}, {lat}; Saved as {file_name}")


# In[15]:


#需要手动修改，导入新的gcm数据，建议结合wps批量一键替换
df_pr_gcm_raw1=xr.open_dataset("D:\Program\CanESM5\pr_day_CanESM5_ssp245_r1i1p1f1_gn_20150101-21001231.nc")
df_pr_gcm_raw1=df_pr_gcm_raw1.sel(time=df_pr_gcm_raw1.time.dt.year.isin(range(2015,2101)))


# 读取外部CSV文件中的经纬度数据（我把csv放在gcm同路径，为了读取方便）
lat_lon_df = pd.read_csv('D:\Program\Point\Point.csv')

# 循环处理每个经纬度点
for index, row in lat_lon_df.iterrows():
    lon, lat = row['lon'], row['lat']
    
    # 插值气候模型数据
    df_pr_gcm1 = df_pr_gcm_raw1.interp(lon=lon, lat=lat) .pr*86400


    result1=[]
    for i in range(1,13)[:]:
        tmp_pr=df_pr_gcm1.sel(time=df_pr_gcm1.time.dt.month==i)
        gcm_pr_downscaled1=tmp_pr*delta_pr[i-1]
        result1.append(gcm_pr_downscaled1)
    gcm_pr_downscaled1_final=xr.merge(result1)

    #根据实际模型名字修改
    model_name = "CanESM5pr"
    first_date = str(gcm_pr_downscaled1_final.time.values[0])[:10]
    file_name = f"{lon}_{lat}_{model_name}_{first_date}_ssp245.nc" 
    gcm_pr_downscaled1_final.to_netcdf(file_name)
    # 打印进度
    print(f"Processed {lon}, {lat}; Saved as {file_name}")


# In[16]:


#需要手动修改，导入新的gcm数据，建议结合wps批量一键替换
df_pr_gcm_raw1=xr.open_dataset("D:\Program\CanESM5\pr_day_CanESM5_ssp585_r1i1p1f1_gn_20150101-21001231.nc")
df_pr_gcm_raw1=df_pr_gcm_raw1.sel(time=df_pr_gcm_raw1.time.dt.year.isin(range(2015,2101)))


# 读取外部CSV文件中的经纬度数据（我把csv放在gcm同路径，为了读取方便）
lat_lon_df = pd.read_csv('D:\Program\Point\Point.csv')

# 循环处理每个经纬度点
for index, row in lat_lon_df.iterrows():
    lon, lat = row['lon'], row['lat']
    
    # 插值气候模型数据
    df_pr_gcm1 = df_pr_gcm_raw1.interp(lon=lon, lat=lat) .pr*86400


    result1=[]
    for i in range(1,13)[:]:
        tmp_pr=df_pr_gcm1.sel(time=df_pr_gcm1.time.dt.month==i)
        gcm_pr_downscaled1=tmp_pr*delta_pr[i-1]
        result1.append(gcm_pr_downscaled1)
    gcm_pr_downscaled1_final=xr.merge(result1)

    #根据实际模型名字修改
    model_name = "CanESM5pr"
    first_date = str(gcm_pr_downscaled1_final.time.values[0])[:10]
    file_name = f"{lon}_{lat}_{model_name}_{first_date}_ssp585.nc" 
    gcm_pr_downscaled1_final.to_netcdf(file_name)
    # 打印进度
    print(f"Processed {lon}, {lat}; Saved as {file_name}")





