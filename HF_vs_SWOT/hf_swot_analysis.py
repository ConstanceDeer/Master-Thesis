# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:25:42 2026

@author: const
"""

#%% Combined analysis of the three stations

from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature   
import xarray as xr
import os
from netCDF4 import Dataset, num2date
import cftime
import pandas as pd


# --------- load stats files -----------

FEDJ= np.load('./stats/FEDJ_junejulyaugust2025_stats.npz',allow_pickle=True) 

SLAT= np.load('./stats/SLAT_junejulyaugust2025_stats.npz',allow_pickle=True) 

KRAK1= np.load('./stats/KRAK_september2025_stats.npz',allow_pickle=True)        #september
KRAK2= np.load('./stats/KRAK_junejulyaugust2025_stats.npz',allow_pickle=True)   # other three months

# --------- load lon/lat positions for the three stations -----------

# load data
ds1 = xr.open_dataset('./hf/KRAK/KRAK_junejulyaugust2025_rgne.nc')
lon= ds1['lon'].values    # lon
lat= ds1['lat'].values    #lat

lon_hf_KRAK=lon[0]
lat_hf_KRAK=lat[0]


ds1 = xr.open_dataset('./hf/SLAT/SLAT_junejulyaugust2025_rgne.nc')
lon= ds1['lon'].values    # lon
lat= ds1['lat'].values    #lat

lon_hf=lon[0]
lat_hf=lat[0]

ds1 = xr.open_dataset('./hf/FEDJ/FEDJ_junejulyaugust2025_rgne.nc')
lon= ds1['lon'].values    # lon
lat= ds1['lat'].values    #lat

lon_hf_FEDJ=lon[0]
lat_hf_FEDJ=lat[0]


#---------- stat vs time plot -----------

plt.figure(figsize=(16/1.5,3/1.5),dpi=150)
plt.axhline(np.nanmean(FEDJ['corr_all']), color='k', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(SLAT['corr_all']), color='b', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(np.append(KRAK1['corr_all'], KRAK2['corr_all'])), color='r', linestyle='--',alpha=0.2)
plt.plot(FEDJ['time'], FEDJ['corr_all'], 'k*',label="FEDJ")
plt.plot(SLAT['time'], SLAT['corr_all'], 'b^',label="SLAT")
plt.plot(KRAK1['time'], KRAK1['corr_all'], 'ro',label="KRAK")
plt.plot(KRAK2['time'], KRAK2['corr_all'], 'ro')
#plt.plot(time_l, corr_rob, '*',label="Outlier removed")
plt.ylabel("RAW Correlation")
plt.minorticks_on()
plt.ylim([0,1])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(16/1.5,3/1.5),dpi=150)
plt.axhline(np.nanmean(FEDJ['rms_all']), color='k', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(SLAT['rms_all']), color='b', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(np.append(KRAK1['rms_all'], KRAK2['rms_all'])), color='r', linestyle='--',alpha=0.2)
plt.plot(FEDJ['time'], FEDJ['rms_all'], 'k*',label="FEDJ")
plt.plot(SLAT['time'], SLAT['rms_all'], 'b^',label="SLAT")
plt.plot(KRAK1['time'], KRAK1['rms_all'], 'ro',label="KRAK")
plt.plot(KRAK2['time'], KRAK2['rms_all'], 'ro')
#plt.plot(time_l, corr_rob, '*',label="Outlier removed")
plt.ylabel("RAW rms")
plt.legend()
plt.minorticks_on()
plt.grid()
plt.show()

plt.figure(figsize=(16/1.5,3/1.5),dpi=150)
plt.axhline(np.nanmean(FEDJ['agree_all']), color='k', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(SLAT['agree_all']), color='b', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(np.append(KRAK1['agree_all'], KRAK2['agree_all'])), color='r', linestyle='--',alpha=0.2)
plt.plot(FEDJ['time'], FEDJ['agree_all'], 'k*',label="FEDJ")
plt.plot(SLAT['time'], SLAT['agree_all'], 'b^',label="SLAT")
plt.plot(KRAK1['time'], KRAK1['agree_all'], 'ro',label="KRAK  ")
plt.plot(KRAK2['time'], KRAK2['agree_all'], 'ro')
#plt.plot(time_l, corr_rob, '*',label="Outlier removed")
plt.ylabel("RAW agreement ration")
plt.legend()
plt.ylim([0,1])
plt.minorticks_on()
plt.grid()
plt.show()

std_all_FEDJ = np.sqrt(FEDJ['rms_all']**2 - (np.nanmean(FEDJ['swot']-FEDJ['hf']))**2)
std_all_SLAT = np.sqrt(SLAT['rms_all']**2 - (np.nanmean(SLAT['swot']-SLAT['hf']))**2)
std_all_KRAK1 = np.sqrt(KRAK1['rms_all']**2 - (np.nanmean(KRAK1['swot']-KRAK1['hf']))**2)
std_all_KRAK2 = np.sqrt(KRAK2['rms_all']**2 - (np.nanmean(KRAK2['swot']-KRAK2['hf']))**2)

plt.figure(figsize=(16/1.5,3/1.5),dpi=150)
plt.axhline(np.nanmean(std_all_FEDJ), color='k', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(std_all_SLAT), color='b', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(np.append(std_all_KRAK1, std_all_KRAK2)), color='r', linestyle='--',alpha=0.2)
plt.plot(FEDJ['time'],
         std_all_FEDJ,
         'k*', label="FEDJ")

plt.plot(SLAT['time'],
         std_all_SLAT   ,
         'b^', label="SLAT")

plt.plot(KRAK1['time'],
         std_all_KRAK1      ,
         'ro', label="KRAK")

plt.plot(KRAK2['time'],
         std_all_KRAK2,
         'ro')
#plt.plot(time_l, corr_rob, '*',label="Outlier removed")
plt.ylabel("RAW standard deviation")
plt.legend()
plt.minorticks_on()
plt.grid()
plt.show()

plt.figure(figsize=(16/1.5,3/1.5),dpi=150)
plt.plot(FEDJ['time'], FEDJ['count'], 'k*',label="FEDJ")
plt.plot(SLAT['time'], SLAT['count'], 'b^',label="SLAT")
plt.plot(KRAK1['time'], KRAK1['count'], 'ro',label="KRAK")
plt.plot(KRAK2['time'], KRAK2['count'], 'ro')
#plt.plot(time_l, corr_rob, '*',label="Outlier removed")
plt.ylabel("RAW number of points")
plt.legend()
plt.minorticks_on()
plt.grid()
plt.show()

# ------ 5% outlier removed only days with over 100 points --------

# mask (only keep valid points)
n=0
mask_SLAT = SLAT['count'] > n
mask_KRAK1 = KRAK1['count'] > n
mask_KRAK2 = KRAK2['count'] > n
mask_FEDJ = FEDJ['count'] > n

plt.figure(figsize=(16,3),dpi=150)
plt.axhline(np.nanmean(FEDJ['corr_rob']), color='k', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(SLAT['corr_rob']), color='b', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(np.append(KRAK1['corr_rob'], KRAK2['corr_rob'])), color='r', linestyle='--',alpha=0.2)
plt.plot(FEDJ['time'][mask_FEDJ], FEDJ['corr_rob'][mask_FEDJ], 'k*',label="FEDJ")
plt.plot(SLAT['time'][mask_SLAT], SLAT['corr_rob'][mask_SLAT], 'b^',label="SLAT")
plt.plot(KRAK1['time'][mask_KRAK1], KRAK1['corr_rob'][mask_KRAK1], 'ro',label="KRAK")
plt.plot(KRAK2['time'][mask_KRAK2], KRAK2['corr_rob'][mask_KRAK2], 'ro')
plt.title('Correlation for days > 100 points, 5 procentil outlier removed')
plt.ylabel("Correlation [-]")
plt.ylim([0,1])
plt.minorticks_on()
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(16,3),dpi=150)
plt.axhline(np.nanmean(FEDJ['rms_rob']), color='k', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(SLAT['rms_rob']), color='b', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(np.append(KRAK1['rms_rob'], KRAK2['rms_rob'])), color='r', linestyle='--',alpha=0.2)
plt.plot(FEDJ['time'][mask_FEDJ], FEDJ['rms_rob'][mask_FEDJ], 'k*',label="FEDJ")
plt.plot(SLAT['time'][mask_SLAT], SLAT['rms_rob'][mask_SLAT], 'b^',label="SLAT")
plt.plot(KRAK1['time'][mask_KRAK1], KRAK1['rms_rob'][mask_KRAK1], 'ro',label="KRAK")
plt.plot(KRAK2['time'][mask_KRAK2], KRAK2['rms_rob'][mask_KRAK2], 'ro')
plt.title('RMS for days > 100 points, 5 procentil outlier removed')
plt.ylabel("RMS [m/s]")
plt.minorticks_on()
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(16,3),dpi=150)
plt.axhline(np.nanmean(FEDJ['agree_rob']), color='k', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(SLAT['agree_rob']), color='b', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(np.append(KRAK1['agree_rob'], KRAK2['agree_rob'])), color='r', linestyle='--',alpha=0.2)
plt.plot(FEDJ['time'][mask_FEDJ], FEDJ['agree_rob'][mask_FEDJ], 'k*',label="FEDJ")
plt.plot(SLAT['time'][mask_SLAT], SLAT['agree_rob'][mask_SLAT], 'b^',label="SLAT")
plt.plot(KRAK1['time'][mask_KRAK1], KRAK1['agree_rob'][mask_KRAK1], 'ro',label="KRAK")
plt.plot(KRAK2['time'][mask_KRAK2], KRAK2['agree_rob'][mask_KRAK2], 'ro')
plt.title('Sign agreement ratio for days > 100 points, 5 procentil outlier removed')
plt.ylabel("agree ratio [-]")
plt.minorticks_on()
plt.ylim([0,1])
plt.legend()
plt.grid()
plt.show()


std_rob_FEDJ = np.sqrt(FEDJ['rms_rob']**2 - (np.nanmean(FEDJ['swot']-FEDJ['hf']))**2)
std_rob_SLAT = np.sqrt(SLAT['rms_rob']**2 - (np.nanmean(SLAT['swot']-SLAT['hf']))**2)
std_rob_KRAK1 = np.sqrt(KRAK1['rms_rob']**2 - (np.nanmean(KRAK1['swot']-KRAK1['hf']))**2)
std_rob_KRAK2 = np.sqrt(KRAK2['rms_rob']**2 - (np.nanmean(KRAK2['swot']-KRAK2['hf']))**2)


plt.figure(figsize=(16,3),dpi=150)
plt.axhline(np.nanmean(std_rob_FEDJ), color='k', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(std_rob_SLAT), color='b', linestyle='--',alpha=0.2)
plt.axhline(np.nanmean(np.append(std_rob_KRAK1, std_rob_KRAK2)), color='r', linestyle='--',alpha=0.2)
plt.plot(FEDJ['time'][mask_FEDJ], std_rob_FEDJ[mask_FEDJ], 'k*',label="FEDJ")
plt.plot(SLAT['time'][mask_SLAT], std_rob_SLAT[mask_SLAT], 'b^',label="SLAT")
plt.plot(KRAK1['time'][mask_KRAK1], std_rob_KRAK1[mask_KRAK1], 'ro',label="KRAK")
plt.plot(KRAK2['time'][mask_KRAK2], std_rob_KRAK2[mask_KRAK2], 'ro')
plt.title('Standard deviation for days > 100 points, 5 procentil outlier removed')
plt.ylabel("Standard deviation [m/s]")
plt.minorticks_on()
plt.legend()
plt.grid()
plt.show()


# ----------- statistics pr Grid cell ----------------

# Combine KRAK1 + KRAK2
KRAK = {
    'swot': np.concatenate([KRAK1['swot'], KRAK2['swot']], axis=0),
    'hf':   np.concatenate([KRAK1['hf'],   KRAK2['hf']], axis=0)
}

regions = {
    'FEDJ': FEDJ,
    'SLAT': SLAT,
    'KRAK': KRAK
}

stats = {}

for name, data in regions.items():

    swot_stack = np.array(data['swot'])
    hf_stack   = np.array(data['hf'])
    diff_stack = swot_stack - hf_stack

    nt, ny, nx = diff_stack.shape

    mean_map = np.full((ny, nx), np.nan)
    rms_map  = np.full((ny, nx), np.nan)
    std_map  = np.full((ny, nx), np.nan)
    corr_map = np.full((ny, nx), np.nan)

    for j in range(ny):
        for i in range(nx):

            u = swot_stack[:, j, i]
            v = hf_stack[:, j, i]
            d = diff_stack[:, j, i]

            mask = np.isfinite(u) & np.isfinite(v)

            if np.sum(mask) < 3:  # not enough valid points for statistics
                continue

            uu = u[mask]
            vv = v[mask]
            dd = d[mask]

            mean_map[j, i] = np.mean(dd)
            rms_map[j, i]  = np.sqrt(np.mean(dd**2))
            std_map[j, i]  = np.std(dd)

            if np.std(uu) > 0 and np.std(vv) > 0:
                corr_map[j, i] = np.corrcoef(uu, vv)[0, 1]

    stats[name] = {
        'mean': mean_map,
        'rms': rms_map,
        'std': std_map,
        'corr': corr_map
    }
# ---------- save as individual variables for easier plotting ----------
mean_map_FEDJ = stats['FEDJ']['mean']
rms_map_FEDJ  = stats['FEDJ']['rms']
corr_map_FEDJ  = stats['FEDJ']['corr']
std_map_FEDJ  = stats['FEDJ']['std']

mean_map_SLAT = stats['SLAT']['mean']
rms_map_SLAT  = stats['SLAT']['rms']
corr_map_SLAT  = stats['SLAT']['corr'] 
std_map_SLAT  = stats['SLAT']['std']     

mean_map_KRAK = stats['KRAK']['mean']
rms_map_KRAK  = stats['KRAK']['rms']
corr_map_KRAK  = stats['KRAK']['corr']
std_map_KRAK  = stats['KRAK']['std']    

# ------------ Grid cell statistis plots --------------------

# loacation
lon_min, lon_max = 1.5, 6
lat_min, lat_max = 58.5, 63.5

# Projection 
proj = ccrs.Orthographic(
    central_longitude=(lon_min + lon_max) / 2,
    central_latitude=(lat_min + lat_max) / 2
)


# mean map
fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': proj},dpi=200)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.LAND)
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = True
gl.right_labels = False
gl.bottom_labels = False
gl.left_labels = True

ax.scatter(
        lon_hf_FEDJ.ravel(),
        lat_hf_FEDJ.ravel(),
        c=abs(mean_map_FEDJ.ravel()),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=0,
        vmax=0.25,
        s=10
        )

ax.scatter(
        lon_hf.ravel(),
        lat_hf.ravel(),
        c=abs(mean_map_SLAT.ravel()),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=0,
        vmax=0.25,
        s=10
        )

im=ax.scatter(
        lon_hf_KRAK.ravel(),
        lat_hf_KRAK.ravel(),
        c=abs(mean_map_KRAK.ravel()),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=0,
        vmax=0.25,
        s=10
        )
plt.colorbar(im, ax=ax, shrink=0.3, label='Absolute mean [m/s]', orientation='horizontal',pad=0)
plt.show()

# RMS map
lon_min, lon_max = 1.5, 6
lat_min, lat_max = 58.5, 63.5

proj = ccrs.Orthographic(
    central_longitude=(lon_min + lon_max) / 2,
    central_latitude=(lat_min + lat_max) / 2
)

fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': proj},dpi=200)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.LAND)
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = True
gl.right_labels = False
gl.bottom_labels = False
gl.left_labels = True

im=ax.scatter(
        lon_hf_FEDJ.ravel(),
        lat_hf_FEDJ.ravel(),
        c=rms_map_FEDJ.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=-0,
        vmax=0.5,
        s=10
        )

ax.scatter(
        lon_hf.ravel(),
        lat_hf.ravel(),
        c=rms_map_SLAT.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=-0,
        vmax=0.5,
        s=10
        )

ax.scatter(
        lon_hf_KRAK.ravel(),
        lat_hf_KRAK.ravel(),
        c=rms_map_KRAK.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=-0,
        vmax=0.5,
        s=10
        )
plt.colorbar(im, ax=ax, shrink=0.3, label='RMS [m/s]', orientation='horizontal',pad=0)
plt.show()


fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': proj},dpi=200)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.LAND)
ax.coastlines()
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = True
gl.right_labels = False
gl.bottom_labels = False
gl.left_labels = True

im=ax.scatter(
        lon_hf_FEDJ.ravel(),
        lat_hf_FEDJ.ravel(),
        c=corr_map_FEDJ.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        s=10
        )

ax.scatter(
        lon_hf.ravel(),
        lat_hf.ravel(),
        c=corr_map_SLAT.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        s=10
        )

ax.scatter(
        lon_hf_KRAK.ravel(),
        lat_hf_KRAK.ravel(),
        c=corr_map_KRAK.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        s=10
        )
plt.colorbar(im, ax=ax, shrink=0.3, label='correlation [-]', orientation='horizontal',pad=0)
plt.show()


fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': proj},dpi=200)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.LAND)
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = True
gl.right_labels = False
gl.bottom_labels = False
gl.left_labels = True

im=ax.scatter(
        lon_hf_FEDJ.ravel(),
        lat_hf_FEDJ.ravel(),
        c=std_map_FEDJ.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=-0,
        vmax=0.2,
        s=10
        )

ax.scatter(
        lon_hf.ravel(),
        lat_hf.ravel(),
        c=std_map_SLAT.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=-0,
        vmax=0.2,
        s=10
        )

ax.scatter(
        lon_hf_KRAK.ravel(),
        lat_hf_KRAK.ravel(),
        c=std_map_KRAK.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=-0,
        vmax=0.2,
        s=10
        )
plt.colorbar(im, ax=ax, shrink=0.3, label='std [m/s]', orientation='horizontal',pad=0)
plt.show()

#%% Code for individual analysis 

from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature   
import xarray as xr
import os
from netCDF4 import Dataset, num2date
import cftime
import pandas as pd



# --- Load data from wished station ---- 

# --- FEDJ ---
#HF= np.load('./stats/FEDJ_junejulyaugust2025_stats.npz',allow_pickle=True) 

#ds1 = xr.open_dataset('./hf/FEDJ/FEDJ_junejulyaugust2025_rgne.nc')
#lon= ds1['lon'].values    # lon
#lat= ds1['lat'].values    #lat

#lon_hf=lon[0]
#lat_hf=lat[0]

# --- SLAT ---
#HF= np.load('./stats/SLAT_junejulyaugust2025_stats.npz',allow_pickle=True) 

#ds1 = xr.open_dataset('./hf/SLAT/SLAT_junejulyaugust2025_rgne.nc')
#lon= ds1['lon'].values    # lon
#lat= ds1['lat'].values    #lat

#lon_hf=lon[0]
#lat_hf=lat[0]

# --- KRAK ---
KRAK1= np.load('./stats/KRAK_september2025_stats.npz',allow_pickle=True)        #september
KRAK2= np.load('./stats/KRAK_junejulyaugust2025_stats.npz',allow_pickle=True)   # other three months
HF = {
    'swot': np.concatenate([KRAK1['swot'], KRAK2['swot']], axis=0),
    'hf':   np.concatenate([KRAK1['hf'],   KRAK2['hf']], axis=0),
    'time': np.concatenate([KRAK1['time'], KRAK2['time']]),
    'corr_all': np.concatenate([KRAK1['corr_all'], KRAK2['corr_all']]),
    'rms_all': np.concatenate([KRAK1['rms_all'], KRAK2['rms_all']]),
    'agree_all': np.concatenate([KRAK1['agree_all'], KRAK2['agree_all']]),
    'corr_rob': np.concatenate([KRAK1['corr_rob'], KRAK2['corr_rob']]),
    'rms_rob': np.concatenate([KRAK1['rms_rob'], KRAK2['rms_rob']]),
    'agree_rob': np.concatenate([KRAK1['agree_rob'], KRAK2['agree_rob']]),
    'count': np.concatenate([KRAK1['count'], KRAK2['count']]),      
}

  
ds1 = xr.open_dataset('./hf/KRAK/KRAK_junejulyaugust2025_rgne.nc')
lon= ds1['lon'].values    # lon
lat= ds1['lat'].values    #lat

lon_hf=lon[0]
lat_hf=lat[0]

#% ------- global  stats - all data (including outliers)  -------
diff_list = [HF['swot'][i] - HF['hf'][i] for i in range(len(HF['time']))]   
diff_all = np.concatenate([d.ravel() for d in diff_list])
u_all = np.concatenate(HF['swot'])
HF_all = np.concatenate(HF['hf'])
time_l = np.array(HF['time'])

mask = ~np.isnan(u_all) & ~np.isnan(HF_all)

# values
u = u_all[mask]
hf = HF_all[mask]

# RMS
rms = np.sqrt(np.mean((u - hf)**2))

# correlation
corr = np.corrcoef(u, hf)[0, 1]

# sign agreement (ratio)
same_sign = np.sign(u) == np.sign(hf)
sign_agreement = np.mean(same_sign)

# counts
count_total = len(u)
count_same = np.sum(same_sign)

# mean diff (optional)
mean_diff = np.nanmean(u - hf)
print('--- Global stats (all data) ---')
print("mean diff:", mean_diff)
print("correlation:", corr)
print("Count:", count_same, "/", count_total)
print("Sign Agreement:", sign_agreement)
print("RMS:", rms)

# --------  global - outliers     -----------

mask = ~np.isnan(u_all) & ~np.isnan(HF_all) 

# valid data
u = u_all[mask]
hf = HF_all[mask]

# residual
diff = u - hf

# --- OUTLIER REMOVAL (5–95%) ---
p_low, p_high = np.nanpercentile(diff, [5, 95])
mask_rob = (diff >= p_low) & (diff <= p_high)

u_r = u[mask_rob]
hf_r = hf[mask_rob]

# --- GLOBAL STATS (cleaned) ---

# RMS
rms = np.sqrt(np.mean((u_r - hf_r)**2))

# correlation
corr = np.corrcoef(u_r, hf_r)[0, 1]

# sign agreement
same_sign = np.sign(u_r) == np.sign(hf_r)
sign_agreement = np.mean(same_sign)

# counts
count_total = len(u_r)
count_same = np.sum(same_sign)

# mean diff (bias)
mean_diff = np.mean(u_r - hf_r)

print("--- 5% outlier removed ---")
print("mean diff:", mean_diff)
print("correlation:", corr)
print("Count:", count_same, "/", count_total)
print("Sign Agreement:", sign_agreement)
print("RMS:", rms)


# ------ Histrogram -------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

data = diff_all[~np.isnan(diff_all)]

# Fit Gaussian
mu, std = norm.fit(data)

# Define x range properly
xmin, xmax = np.min(data), np.max(data)
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, mu, std)

plt.figure(figsize=(10,5))

# Histogram
plt.hist(data, bins=50, density=True, alpha=0.7,
         label='Residuals (SWOT - HF radar)')

# Gaussian fit
plt.plot(x, p, 'k', linewidth=2,
         label=f'Gaussian fit\nμ={mu:.3f}, σ={std:.3f}')

plt.xlabel('Difference (SWOT range velocity - HF radar daily mean) (m/s)')
plt.ylabel('Density')
plt.title('Histogram of Differences between SWOT and HF radar')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# ----- GMM fit and histrogram ------ 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

data = diff_all[~np.isnan(diff_all)].reshape(-1, 1)

# Fit 3-component GMM
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(data)

# Create x-axis for plotting
x = np.linspace(data.min(), data.max(), 500).reshape(-1, 1)

# Full model density
logprob = gmm.score_samples(x)
pdf = np.exp(logprob)

# Individual Gaussians
weights = gmm.weights_
means = gmm.means_.flatten()
covs = gmm.covariances_.flatten()
stds = np.sqrt(covs)

plt.figure(figsize=(8,4),dpi=250)

# Histogram
plt.hist(data, bins=100, density=True, alpha=0.6,
         label='Residuals')

# Total GMM fit
plt.plot(x, pdf, 'k', lw=2, label='GMM (3 components)')

# Individual components
for i in range(3):
    component = weights[i] * (1 / (np.sqrt(2*np.pi)*stds[i])) * \
                np.exp(-0.5 * ((x.flatten() - means[i]) / stds[i])**2)

    plt.plot(x, component, linestyle='--',
             label=f'Comp {i+1}: μ={means[i]:.3f}, σ={stds[i]:.3f}')

plt.xlabel('Residuals [m/s]')
plt.ylabel('Count (# of residuals)')
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend(loc='upper right')
plt.show()
