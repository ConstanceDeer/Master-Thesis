# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:17:23 2026

@author: const
"""
import xarray as xr
from netCDF4 import Dataset, num2date
import numpy as np
from datetime import datetime, timedelta


# -------------------------------------------------
# User settings

#location

lat_min, lat_max = 57, 64.5
lon_min, lon_max = -1, 10

# Months

months = [9]  # 6 for June, 7 for July

# output file have 
output_file='./hf/KRAK/KRAK_september2025_rgne.nc'

# Choose radar

radar="KRAK"  # "SLAT", "FEDJ", "KRAK"

# file size diff radars ( out comment the right one)

Q,P=38,72 #KRAK
#Q,P=40,72 #Others

# -------------------------------------------------
# Loop

# Loop over days in month 2025
start_date = datetime(2025, min(months), 1)
end_date = datetime(2025, max(months), 25)


# Initialize lists to store daily arrays
rdva_list = []
drva_list = []
bear_list = []
rnge_list = []
lat_list = []
lon_list = []
time_list = []
err_t_list=[]
err_s_list=[]


# Loop over each day, load data, and append to lists

for n in range((end_date - start_date).days + 1):
    day = start_date + timedelta(days=n)
    day_str = day.strftime("%Y%m%d")
    
    # Construct the THREDDS file URL for that day
    url = f"https://thredds.met.no/thredds/dodsC/remotesensinghfradar/{radar}/2025/{day_str[4:6]}/{day_str[6:8]}/RDLm_{radar}_2025_{day_str[4:6]}_{day_str[6:8]}.nc"
    
    print(f"Processing {day_str} ...")
   
    try:
        nc = Dataset(url)
    except Exception as e:
        print(f"Could not open {url}: {e}")
        continue
    
    # Load lat/lon grid once
    lat_grid = nc.variables["LATITUDE"][:]
    lon_grid = nc.variables["LONGITUDE"][:]

    rdva = nc.variables["RDVA"][:]  # shape (time, ny, nx)
    drva = nc.variables["DRVA"][:]
    print(rdva.shape)

    bear = nc.variables["BEAR"][:]  # shape (time, ny, nx)
    rnge = nc.variables["RNGE"][:]  # shape (time, ny, nx)

    time_var = nc.variables["TIME"][:]

    err_space = nc.variables["ESPC"][:]  # Rdaial std space shape (time, ny, nx)
    err_time = nc.variables["ETMP"][:]  # radial std over time shape (time, ny, nx)
    
    nt = time_var.shape[0]  # number of timesteps for the day

    # Repeat lat/lon along time axis
    lat_3d_day = np.repeat(lat_grid[np.newaxis, :, :], nt, axis=0)
    lon_3d_day = np.repeat(lon_grid[np.newaxis, :, :], nt, axis=0)

    # Append to lists
    rdva_list.append(rdva)
    drva_list.append(drva)
    lat_list.append(lat_3d_day)
    lon_list.append(lon_3d_day)
    time_list.append(time_var)
    err_t_list.append(err_time)
    err_s_list.append(err_space)

# Concatenate along time axis
rdva_3d = np.concatenate(rdva_list, axis=0)  # shape (total_time, ny, nx)
drva_3d = np.concatenate(drva_list, axis=0)
time_3d = np.concatenate(time_list, axis=0)
lat_3d = np.concatenate(lat_list, axis=0)  # shape (total_time, ny, nx)
lon_3d = np.concatenate(lon_list, axis=0)
err_s_3d = np.concatenate(err_s_list, axis=0)
err_t_3d = np.concatenate(err_t_list, axis=0)


bear_2d = np.repeat(bear[:, np.newaxis], repeats=Q, axis=1)  # shape (bear, range)
rnge_2d = np.repeat(rnge[np.newaxis,:], repeats=P, axis=0)  # shape (bear, range)

print("rdva_3d shape:", rdva_3d.shape)
print("drva_3d shape:", drva_3d.shape)
print("bear_3d shape:", bear_2d.shape)
print("rnge_3d shape:", rnge_2d.shape)
print("err_t_3d shape:", err_t_3d.shape)

# -------------------------------------------------
# Save

from netCDF4 import Dataset, date2num
ncfile = Dataset(output_file, mode='w', format='NETCDF4_CLASSIC')

time_dim = ncfile.createDimension('time', len(time_3d))  # unlimited time dimension
lon_dim = ncfile.createDimension('lon', P)
lat_dim = ncfile.createDimension('lat', Q) # for KRAK

time_var = ncfile.createVariable('time', np.float64, ('time',), fill_value=np.nan)
time_var.units = 'days since 1950-01-01 00:00:00.0Z'
time_var.long_name = 'time'
time_var[:] = time_3d

lon_var = ncfile.createVariable('lon', np.float32, ('time','lon','lat'))
lon_var.units = 'degrees_east'
lon_var.long_name = 'longitude'
lon_var[:,:,:] = lon_3d

lat_var = ncfile.createVariable('lat', np.float32, ('time','lon','lat'))
lat_var.units = 'degrees_north'
lat_var.long_name = 'latitude'
lat_var[:,:,:] = lat_3d

range_var = ncfile.createVariable('range', np.float32, ('lon','lat'))
range_var.units = 'km'
range_var.long_name = 'range away from instrument'
range_var[:,:] = rnge_2d

bear_var = ncfile.createVariable('bear', np.float32, ('lon','lat'))
bear_var.units = 'degrees_true'
bear_var.long_name = 'bearing away from instrument'
bear_var[:,:] = bear_2d

u_var = ncfile.createVariable('velocity', np.float32, ('time','lon','lat'))
u_var.units = 'm/s'
u_var.long_name = 'radial velocity away from instrument'
u_var[:,:,:] = rdva_3d      

vg_var = ncfile.createVariable('drva', np.float32, ('time','lon','lat'))
vg_var.units = '0-360 degrees'
vg_var.long_name = 'direction of radial vector away from instrument'
vg_var[:,:,:] = drva_3d

errt_var = ncfile.createVariable('err_t', np.float32, ('time','lon','lat'))
errt_var.units = 'm/s'
errt_var.long_name = 'std of the velocities from time cell'
errt_var[:,:,:] = err_t_3d

errs_var = ncfile.createVariable('err_s', np.float32, ('time','lon','lat'))
errs_var.units = 'm/s'
errs_var.long_name = 'std of the velocities from space cell'
errs_var[:,:,:] = err_s_3d

ncfile.close()
print("Saved HF-radar monthly data to:", output_file)
