# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:04:06 2026

@author: const
"""

import xarray as xr
import numpy as np
from scipy.interpolate import RectBivariateSpline
import glob
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# --- 1. Load reference grid (GOCO06s) ---
ref_ds = xr.open_dataset("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/old/raw_grid/GOCO06s_geoid.nc")
lon_ref = ref_ds["lon"].values[0, :]  # 1D
lat_ref = ref_ds["lat"].values[:, 0]  # 1D
# Make sure lat is decreasing for RectBivariateSpline (from min→max)
if lat_ref[0] > lat_ref[-1]:
    lat_ref = lat_ref[::-1]

# --- 2. Folder with your geoid files ---
files = glob.glob("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/NKG.nc")  # or "*.nc"

for f in files:
    print("Processing:", f)
    
    ds = xr.open_dataset(f)
    geoid = ds["geoid"].values
    lat = ds["latitude"].values[:]  # 1D
    lon = ds["longitude"].values[:]  # 1D
    
    # Make sure lat is increasing for interpolation
    flip_lat = False
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        geoid = geoid[::-1, :]
        flip_lat = True
    
    # Remove NaNs for interpolation
    mask = ~np.isnan(geoid)
    if not np.all(mask):
        # Simple approach: fill NaNs with nearest neighbor for spline
        from scipy.ndimage import generic_filter
        filled = geoid.copy()
        filled[np.isnan(filled)] = 0
        def nanmean_filter(x):
            x = x.reshape((3,3))
            return np.nanmean(x) if not np.isnan(x[4]) else np.nanmean(x)
        filled = generic_filter(filled, nanmean_filter, size=3)
        geoid = filled
    
    # --- 3. Spline interpolation ---
    spline = RectBivariateSpline(lat, lon, geoid, kx=3, ky=3)
    geoid_new = spline(lat_ref, lon_ref)
    
    lon_min, lon_max = -0.9, 8
    lat_min, lat_max = 58, 65

    proj = ccrs.Orthographic(
        central_longitude=(lon_min+lon_max)/2,
        central_latitude=(lat_min+lat_max)/2
    )
    
    fig = plt.figure(figsize=(20,15),dpi=300)
    ax = fig.add_subplot(2,1,1, projection=proj)

    ax.coastlines(resolution='10m')
    gl1 = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,alpha=0.5,linestyle="--")
    gl1.top_labels = False
    gl1.right_labels = False
    ax.set_extent([-1, 10, 57, 65], crs=ccrs.PlateCarree())

    sc = ax.pcolormesh(
        lon,
        lat,
        geoid.squeeze(),
        vmin=36,
        vmax=52,
        cmap='RdBu',      # your colormap
        shading='auto',
        transform=ccrs.PlateCarree()
    )

    plt.colorbar(sc,shrink=1,label='ssha [m]')
    plt.title('Geoid: EGG old res')

    plt.show()


    fig = plt.figure(figsize=(20,15),dpi=300)
    ax = fig.add_subplot(2,1,1, projection=proj)

    ax.coastlines(resolution='10m')
    gl1 = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,alpha=0.5,linestyle="--")
    gl1.top_labels = False
    gl1.right_labels = False
    ax.set_extent([-1, 10, 57, 65], crs=ccrs.PlateCarree())

    sc = ax.pcolormesh(
        lon_ref,
        lat_ref,
        geoid_new.squeeze(),
        vmin=36,
        vmax=52,
        cmap='RdBu',      # your colormap
        shading='auto',
        transform=ccrs.PlateCarree()
    )

    plt.colorbar(sc,shrink=1,label='ssha [m]')
    plt.title('Geoid: EGG new res')

    plt.show()
    

    # --- 4. Save regridded to NetCDF ---
    
lat2d_ref, lon2d_ref = np.meshgrid(lon_ref, lat_ref)
lat=lon2d_ref
lon=lat2d_ref

import xarray as xr
import numpy as np

ds = xr.Dataset(
    {
        "geoid": (["y", "x"], geoid_new),
        "lat": (["y", "x"], lat),
        "lon": (["y", "x"], lon)
    }
)

ds.geoid.attrs["units"] = "m"
ds.geoid.attrs["long_name"] = "Geoid height"

ds.lat.attrs["units"] = "degrees_north"
ds.lon.attrs["units"] = "degrees_east"


    
out_file = "C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/final/EKG2015.nc"
ds.to_netcdf(out_file, encoding={"geoid":{"zlib":True,"complevel":4}})
print("Saved:", out_file)
    
    
#%% Load test

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

ds = xr.open_dataset("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/XGM2016_grid.nc")

geoid = ds["geoid"].values
lat = ds["lat"].values
lon = ds["lon"].values


lon_min, lon_max = -0.9, 8
lat_min, lat_max = 58, 65

proj = ccrs.Orthographic(
    central_longitude=(lon_min+lon_max)/2,
    central_latitude=(lat_min+lat_max)/2
)


fig = plt.figure(figsize=(20,15), dpi=300)
ax = fig.add_subplot(2,1,1, projection=proj)

ax.coastlines(resolution="10m")

gl = ax.gridlines(draw_labels=True,
                  x_inline=False,
                  y_inline=False,
                  linestyle="--",
                  alpha=0.5)

gl.top_labels = False
gl.right_labels = False

ax.set_extent([-1, 10, 57, 65], crs=ccrs.PlateCarree())

sc = ax.pcolormesh(
    lon,
    lat,
    geoid,
    cmap="RdBu",
    vmin=36,
    vmax=52,
    shading="auto",
    transform=ccrs.PlateCarree()
)

plt.colorbar(sc, shrink=0.8, label="Geoid height [m]")
plt.title("Geoid: Load test")

plt.show()