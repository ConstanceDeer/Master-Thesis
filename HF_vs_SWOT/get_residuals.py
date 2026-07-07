# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:24:10 2026

@author: const
"""
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from scipy.spatial import cKDTree
from datetime import timedelta
import cftime

# ----------------------------
# Mean filter
# ----------------------------
def mean_filter(image, ksize=3):
    '''
    BoxMean filter:
    
    Input: 2D image, kernel size 
    
    Output: Smooth image
    
    '''
    pad = ksize // 2
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+ksize, j:j+ksize]
            output[i, j] = np.nanmean(window)
    return output


# ----------------------------
# User input
# ----------------------------

# HF file
ds1 = xr.open_dataset('./hf/KRAK/KRAK_september2025_rgne.nc')


# HF radar location (Outcomment right one )
# SLAT
#lat0 = np.deg2rad(59.9086667)
#lon0 = np.deg2rad(5.0669167)

# KRAK
lat0 = np.deg2rad(62.0329833)
lon0 = np.deg2rad(4.9878833)

#FEDJ
#lat0 = np.deg2rad(60.7759500)
#lon0 = np.deg2rad(4.6946500)

# swot file

load_file = nc.Dataset('./swot/JJAS_nativ_filtered.nc', "r")

# save filename
save_file="./stats/filtered_k1/krak2.npz"

# ------   HF load -----

hf_vel = ds1['velocity']
hf_lon = ds1['lon'].values[0]
hf_lat = ds1['lat'].values[0]

hf_time = pd.to_datetime(ds1['time'].values)

daily_mean = hf_vel.groupby("time.date").mean("time")
valid = hf_vel.groupby("time.date").count("time") == 24
daily_mean = daily_mean.where(valid)

hf_dates = pd.to_datetime(hf_time).normalize().unique()


# ------------- SWOT load ---------------


swot = {k: load_file.variables[k][:] for k in load_file.variables.keys()}

swot_time = swot['time']
swot_lon = swot['lon']
swot_lat = swot['lat']
swot_u = swot['ugos']
swot_v = swot['vgos']

# convert time
swot_dates = np.full(swot_time.shape, np.datetime64('NaT'), dtype='datetime64[ns]')
valid = np.isfinite(swot_time)

converted = cftime.num2date(
    swot_time[valid],
    units='seconds since 2000-01-01 00:00:00',
    calendar='standard'
)

swot_dates[valid] = np.array([d.isoformat() for d in converted], dtype='datetime64[ns]')


# -------- INITALIZE --------------------

corr_all, rms_all, agree_all = [], [], []
corr_rob, rms_rob, agree_rob, count_rob = [], [], [], []
time_list = []
diff_list, swot_all, hf_all = [],[],[]


# ----------------------------
# LOOP
# ----------------------------
for t in hf_dates:

    tmin = np.datetime64(t)
    tmax = np.datetime64(t + pd.Timedelta(days=1))

    swot_mask = (swot_dates >= tmin) & (swot_dates < tmax)
    swot_idx = np.where(swot_mask)[0]

    if len(swot_idx) == 0:
        continue

    # ----------------------------
    # SWOT slice
 
    lon_s = swot_lon[:, swot_idx]
    lat_s = swot_lat[:, swot_idx]

    u = swot_u[:, swot_idx]
    v = swot_v[:, swot_idx]

    u = mean_filter(u, ksize=1)
    v = mean_filter(v, ksize=1)

    # ----------------------------
    # bearing
   

    latr = np.deg2rad(lat_s)
    lonr = np.deg2rad(lon_s)

    dlon = lonr - lon0

    theta = np.arctan2(
        np.sin(dlon) * np.cos(latr),
        np.cos(lat0) * np.sin(latr) -
        np.sin(lat0) * np.cos(latr) * np.cos(dlon)
    )

    swot_bear = (np.rad2deg(theta) + 360) % 360
    theta = np.deg2rad(swot_bear)

    swot_radial = u * np.sin(theta) + v * np.cos(theta)
    swot_radial[np.abs(swot_radial) > 10] = np.nan

    # ----------------------------
    # interpolation
    
    swot_points = np.column_stack((lon_s.ravel(), lat_s.ravel()))
    hf_points = np.column_stack((hf_lon.ravel(), hf_lat.ravel()))
    swot_vals = swot_radial.ravel()

    R = 6371.0
    tree = cKDTree(np.deg2rad(swot_points))
    dist, idx = tree.query(np.deg2rad(hf_points), k=4)

    dist_km = dist * R
    mask = dist_km <= 5

    neighbors = swot_vals[idx]
    neighbors = np.where(mask, neighbors, np.nan)

    u_interp = np.nanmean(neighbors, axis=1).reshape(hf_lon.shape)

    # ----------------------------
    # Match data
  
    hf_day = pd.to_datetime(t).date()

    if hf_day not in daily_mean.date.values:
        continue

    hf_field = daily_mean.sel(date=hf_day).values

    u1 = u_interp
    u2 = hf_field

    mask_valid = np.isfinite(u1) & np.isfinite(u2)

    if np.sum(mask_valid) < 10:
        continue

    # ------------------
    # Residual
    
    res = u1 - u2

    corr = np.corrcoef(u1[mask_valid], u2[mask_valid])[0, 1]
    rms = np.sqrt(np.mean(res[mask_valid]**2))
    agree = np.mean(np.sign(u1[mask_valid]) == np.sign(u2[mask_valid]))

    # ------------------------
    # Remove 5% outliers 
    
    res_valid = res[mask_valid]

    p_low, p_high = np.nanpercentile(res_valid, [5, 95])

    mask_outlier_free = np.zeros_like(mask_valid, dtype=bool)
    mask_outlier_free[np.where(mask_valid)] = (
        (res_valid >= p_low) & (res_valid <= p_high)
    )

    if np.sum(mask_outlier_free) < 10:
        continue
    
    # ------------------------
    # save statistics

    corr_r = np.corrcoef(u1[mask_outlier_free], u2[mask_outlier_free])[0, 1]
    rms_r = np.sqrt(np.mean((u1[mask_outlier_free] - u2[mask_outlier_free])**2))
    agree_r = np.mean(np.sign(u1[mask_outlier_free]) == np.sign(u2[mask_outlier_free]))

    corr_rob.append(corr_r)
    rms_rob.append(rms_r)
    agree_rob.append(agree_r)
    count_rob.append(np.sum(mask_outlier_free))
    
    corr_all.append(corr)
    rms_all.append(rms)
    agree_all.append(agree)

    diff_list.append(res)
    swot_all.append(u1)
    hf_all.append(u2)
    time_list.append(t)

    print(t, corr_r)

print("DONE")

# ----------------------------
# SAVE
# ----------------------------

np.savez(
    save_file,
    time=np.array(time_list),
    corr_all=corr_all,
    rms_all=rms_all,
    agree_all=agree_all, 
    corr_rob=corr_rob,
    rms_rob=rms_rob,
    agree_rob=agree_rob,
    count=count_rob,
    hf=hf_all,
    swot=swot_all,
    
)
  