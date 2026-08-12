# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:04:47 2026

@author: const
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
import scipy
import os
import numpy.ma as ma
import pyinterp
import pyinterp.backends.xarray
import xarray as xr
from scipy.interpolate import Rbf
import numpy as np


# initals
extrapolate = 1
plot = 0

# Date interval

dates = pd.date_range(
    "2025-06-01",
    "2025-08-31"
)

# Load nk grid

date0 = dates[0]

day_str = date0.strftime("%Y%m%d")
year = date0.strftime("%Y")
month = date0.strftime("%m")

url = (
    f"https://thredds.met.no/thredds/dodsC/"
    f"romshindcast/norkyst_v3/zdepth/"
    f"{year}/{month}/norkyst800-{day_str}.nc"
)

ds = nc.Dataset(url)


lat_full = ds.variables["lat"][:]
lon_full = ds.variables["lon"][:]


# Extract region once

mask = (
    (lat_full >= 55.5) & (lat_full <= 69.5) &
    (lon_full >= 1) & (lon_full <= 14.5)
)

ys, xs = np.where(mask)

y0, y1 = ys.min(), ys.max()
x0, x1 = xs.min(), xs.max()


lat_nk = lat_full[y0:y1, x0:x1]
lon_nk = lon_full[y0:y1, x0:x1]


# Create regular lon/lat grid

lon_new = np.arange(
    1,
    14.5,
    0.03
)

lat_new = np.arange(
    55.5,
    69.5,
    0.015
)


lon_grid, lat_grid = np.meshgrid(
    lon_new,
    lat_new
)


# Prepare interpolation 

points = (
    lon_nk.ravel(),
    lat_nk.ravel()
)


# ready dac load 
current_month = None
dac = None


# initalise

adt_all=ma.zeros([0,934,450])
time_all=([])

# Loop over days

for date in dates:
    

    print(date.strftime("%Y-%m-%d"))
    
    month = date.strftime("%m")

    if month != current_month:

        if dac is not None:
            dac.close()

        if month == "06":
            dac = xr.open_dataset("./data/dac/dac_daily_mean_June2025.nc")
        elif month == "07":
            dac = xr.open_dataset("./data/dac/dac_daily_mean_July2025.nc")
        elif month == "08":
            dac = xr.open_dataset("./data/dac/dac_daily_mean_August2025.nc")

        current_month = month

    day_str = date.strftime("%Y%m%d")
    year = date.strftime("%Y")
    month = date.strftime("%m")


    url = (
        f"https://thredds.met.no/thredds/dodsC/"
        f"romshindcast/norkyst_v3/zdepth/"
        f"{year}/{month}/norkyst800-{day_str}.nc"
    )


    ds = nc.Dataset(url)


 
    # daily mean sea level

    zeta = (
        ds.variables["zeta"][:, y0:y1, x0:x1]
        .mean(axis=0)
    )
    
    
    # get mask from masked array
    
    land_mask = np.ma.getmaskarray(zeta)
    
    
    # convert masked values to NaN
    
    zeta = zeta.filled(np.nan)
    
    
    # Remove missing values
    
    values = zeta.ravel()
    
    valid = np.isfinite(values)
    
    
    # Regrid to regular lon/lat
    
    adt_reg = griddata(
        (
            lon_nk.ravel()[valid],
            lat_nk.ravel()[valid]
        ),
        values[valid],
        (
            lon_grid,
            lat_grid
        ),
        method="cubic"
    )
    
    # daily mean sea level
    
    
    # create ocean mask
    # 1 = ocean, 0 = land
    
    ocean_mask = valid.astype(float)
    
    
    mask_reg = griddata(
        (
            lon_nk.ravel(),
            lat_nk.ravel()
        ),
        ocean_mask.ravel(),
        (
            lon_grid,
            lat_grid
        ),
        method="nearest"
    )
    
    
    # remove interpolated values over land
    
    adt_reg[mask_reg < 0.5] = np.nan
    
    #fig = plt.figure(figsize=(20,15))
    #ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    #ax.coastlines(resolution='10m')
   
    #gl1 = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,alpha=0.5,linestyle="--")
    #gl1.top_labels = False
    #gl1.right_labels = False
    
    #ax.set_extent([1, 14, 56, 69], crs=ccrs.PlateCarree())
  
    #sc=ax.pcolormesh(lon_grid, lat_grid, adt_reg, vmin=-1, vmax=1,cmap='turbo')
    #plt.colorbar(sc,shrink=1,label='ADT [m]')
    #plt.suptitle(date.strftime("%Y-%m-%d"),fontsize=20)
    
    #plt.show()
    #fig.savefig(os.path.join('./data/Duacs/figures/',dir_list[i][0:-3]+'.png'),dpi=100)
    #plt.close() 


    # DAC correction add
    
    dac_day = dac["dac"].sel(
        time=date.strftime("%Y-%m-%d")
    )
    
    lon_dac,lat_dac=np.meshgrid(dac["longitude"].values,dac["latitude"].values)
    
    # interpolate DAC regular grid -> NorKyst curvilinear grid
    DAC_points = np.column_stack(
        (
            lon_dac.ravel(),
            lat_dac.ravel()
        )
    )
    
    NK_points = np.column_stack(
        (
            lon_grid.ravel(),
            lat_grid.ravel()
        )
    )
    #
    dac_corr = griddata(
        DAC_points,
        dac_day.values.ravel(),
        NK_points,
        method="cubic"
    ).reshape(lon_grid.shape)
    
    
    adt_corrected = adt_reg - dac_corr  #+ 0.2-0.049
   
    adt_corrected[np.abs(adt_corrected) > 1] = np.nan
  
    # fill out land points 
    if extrapolate == 1:
        #get only the valid values
        mask = np.isnan(adt_reg)
        
        latg1 = lat_grid[~mask]
        long1 = lon_grid[~mask]
        newadt = adt_corrected[~mask]
        
        adt = scipy.interpolate.griddata((latg1, long1), newadt.ravel(),
                                  (lat_grid, lon_grid),
                                     method='nearest')
    
  
    
    if plot == 1:
        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax2.coastlines(resolution='10m')
        gl1 = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,alpha=0.5,linestyle="--")
        gl1.top_labels = False
        gl1.right_labels = False
        gl2 = ax2.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,alpha=0.5,linestyle="--")
        gl2.top_labels = False
        gl2.right_labels = False
        ax.set_extent([1, 14, 56, 69], crs=ccrs.PlateCarree())
        ax2.set_extent([1, 14, 56, 69], crs=ccrs.PlateCarree())
        
        sc=ax.pcolormesh(lon_grid, lat_grid, adt_corrected, vmin=-1, vmax=1,cmap='RdBu_r')
        sc2=ax2.pcolormesh(lon_grid,lat_grid,adt,vmin=-1,vmax=1,cmap='RdBu_r')
        plt.colorbar(sc2,shrink=1,label='ADT [m]')
        plt.suptitle(date.strftime("%Y-%m-%d"),fontsize=20)
        
        plt.show()
        #fig.savefig(os.path.join('./data/Duacs/figures/',dir_list[i][0:-3]+'.png'),dpi=100)
        plt.close() 
    
    adt_all = np.vstack([adt_all,np.reshape(adt,[1,934,450])])
    time_all = np.append(time_all,date)
    #plt.matshow(adt)

#%% save

try: ncfile.close()  # just to be safe, make sure dataset is not already open.
except: pass
ncfile = nc.Dataset('./data/nk_BC/NK_fullNO.nc',mode='w',format='NETCDF4_CLASSIC') 

time_dim = ncfile.createDimension('time',np.size(time_all))
lon_dim = ncfile.createDimension('lon',450)
lat_dim = ncfile.createDimension('lat',934)

lon = ncfile.createVariable('lon', np.float32, ('lon'))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
lat = ncfile.createVariable('lat', np.float32, ('lat'))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'days since 1950-01-01 00:00:00'
time.long_name = 'time'
adt = ncfile.createVariable('adt', np.float32, ('time','lat','lon'))
adt.units = 'm'
adt.long_name = 'sea level anomaly'

lon[:] = lon_new
lat[:] = lat_new
ref_date = np.datetime64('1950-01-01T00:00:00')

time_num = (
    (np.array(time_all, dtype='datetime64[ns]') - ref_date)
    / np.timedelta64(1, 'D')
).astype(np.float64)

time[:] = time_num
adt[:,:,:] = adt_all

ncfile.close()