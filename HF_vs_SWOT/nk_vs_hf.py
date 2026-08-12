# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:13:19 2026

@author: const
"""


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

FEDJ= np.load('./nk/fedj.npz',allow_pickle=True) 

SLAT= np.load('./nk/slat.npz',allow_pickle=True) 


ds1 = xr.open_dataset('./hf/SLAT/SLAT_junejulyaugust2025_rgne.nc')
lon= ds1['lon'].values    # lon
lat= ds1['lat'].values    #lat

lon_hf_slat=lon[0]
lat_hf_slat=lat[0]

ds1 = xr.open_dataset('./hf/FEDJ/FEDJ_junejulyaugust2025_rgne.nc')
lon= ds1['lon'].values    # lon
lat= ds1['lat'].values    #lat

lon_hf_FEDJ=lon[0]
lat_hf_FEDJ=lat[0]



# ----------- statistics pr Grid cell ----------------


regions = {
    'FEDJ': FEDJ,
    'SLAT': SLAT,
}

stats = {}

for name, data in regions.items():

    swot_stack = np.array(data['nk_geo'])
    hf_stack   = np.array(data['hf_day'])
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
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj},dpi=200)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.LAND,color='black')
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = True
gl.right_labels = False
gl.bottom_labels = False
gl.left_labels = True

im=ax.scatter(
        lon_hf_FEDJ.ravel(),
        lat_hf_FEDJ.ravel(),
        c=abs(mean_map_FEDJ.ravel()),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=0,
        vmax=0.25,
        s=2
        )

ax.scatter(
        lon_hf_slat.ravel(),
        lat_hf_slat.ravel(),
        c=abs(mean_map_SLAT.ravel()),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=0,
        vmax=0.25,
        s=2
        )

plt.colorbar(im, ax=ax, shrink=0.29, label='Absolute mean [m/s]', orientation='horizontal',pad=0)
plt.show()

# RMS map
lon_min, lon_max = 1.5, 6
lat_min, lat_max = 58.5, 63.5

proj = ccrs.Orthographic(
    central_longitude=(lon_min + lon_max) / 2,
    central_latitude=(lat_min + lat_max) / 2
)

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj},dpi=200)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.LAND,color='black')
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
        s=2
        )

ax.scatter(
        lon_hf_slat.ravel(),
        lat_hf_slat.ravel(),
        c=rms_map_SLAT.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=-0,
        vmax=0.5,
        s=2
        )

plt.colorbar(im, ax=ax, shrink=0.29, label='RMS [m/s]', orientation='horizontal',pad=0)
plt.show()


fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj},dpi=200)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.LAND,color='black')
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
        s=2
        )

ax.scatter(
        lon_hf_slat.ravel(),
        lat_hf_slat.ravel(),
        c=corr_map_SLAT.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        s=2
        )

plt.colorbar(im, ax=ax, shrink=0.29, label='correlation [-]', orientation='horizontal',pad=0)
plt.show()


fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj},dpi=200)
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.LAND,color='black')
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
        s=2
        )

ax.scatter(
        lon_hf_slat.ravel(),
        lat_hf_slat.ravel(),
        c=std_map_SLAT.ravel(),
        transform=ccrs.PlateCarree(),
        cmap='Reds',
        vmin=-0,
        vmax=0.2,
        s=3
        )

plt.colorbar(im, ax=ax, shrink=0.29, label='std [m/s]', orientation='horizontal',pad=0)
plt.show()


#%%

from sklearn.mixture import GaussianMixture
geo_velocity = np.array(FEDJ['diff_geo'])

hf_mean=np.array(FEDJ['hf_day']) # daily mean hf data

nk_g=np.array(FEDJ['nk_geo']) # geostropgic velocity from nk on hf grid 
t = np.array(FEDJ['time'])


u1=nk_g
u2=hf_mean
res=u1-u2

mask_valid = np.isfinite(u1) & np.isfinite(u2)
corr = np.corrcoef(u1[mask_valid], u2[mask_valid])[0, 1]
rms = np.sqrt(np.mean(res[mask_valid]**2))
agree = np.mean(np.sign(u1[mask_valid]) == np.sign(u2[mask_valid]))

mask_valid = np.isfinite(u1) & np.isfinite(u2)

corr = np.corrcoef(u1[mask_valid], u2[mask_valid])[0, 1]
rms = np.sqrt(np.mean(res[mask_valid]**2))
agree = np.mean(np.sign(u1[mask_valid]) == np.sign(u2[mask_valid]))

mean_bias = np.mean(res[mask_valid])
std_bias = np.std(res[mask_valid])      # population standard deviation
# or
std_bias = np.std(res[mask_valid], ddof=1)  # sample standard deviation

data = res[~np.isnan(res)].reshape(-1, 1)

gmm = GaussianMixture(n_components=1, random_state=0)
gmm.fit(data)


x = np.linspace(data.min(), data.max(), 500).reshape(-1, 1)

logprob = gmm.score_samples(x)
pdf = np.exp(logprob)
weights = gmm.weights_
means = gmm.means_.flatten()
covs = gmm.covariances_.flatten()
stds = np.sqrt(covs)

# plot 
plt.figure(figsize=(5,5),dpi=150)

counts, bins, _ = plt.hist(data, bins=100, alpha=0.6,
                           label='Residuals')

bin_width = bins[1] - bins[0]

plt.plot(x, pdf * len(data) * bin_width, 'k', lw=2,
         label='Gaussian fit')


plt.xlabel('Residuals [m/s]')
plt.ylabel('Count')
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend()
plt.show()


print("---- Geostrophic ----")
print(f"Correlation : {corr:.3f}")
print(f"RMSD        : {rms:.3f}")
print(f"Mean bias   : {mean_bias:.3f}")
print(f"Std. bias   : {std_bias:.3f}")
print(f"Sign agree  : {agree:.3%}")   # e.g. 84.2%
