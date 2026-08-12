# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:50:00 2026

@author: const
"""


def geo_curr_swot(lat,lon,ADT,res):
    # geo curr
    
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    dlat_y, dlat_x = np.gradient(lat_rad)
    dlon_y, dlon_x = np.gradient(lon_rad)
    
    heading = np.arctan2(dlon_y, dlat_y)
    
    heading_mean = np.nanmean(np.rad2deg(heading))
    
    orbit_angle = heading   # already same shape as ssha

    # calc gradients in ADT in along and cross track
    
    grad_along, grad_cross = np.gradient(ADT, res, res)
 
   
    # calc meridonal and zonal gradient  
    
    grad_along_x = grad_along*np.sin(orbit_angle)
    
    grad_along_y = grad_along*np.cos(orbit_angle)
    

    grad_cross_x = grad_cross*np.cos(orbit_angle)
    
    grad_cross_y = -grad_cross*np.sin(orbit_angle)
        
    
    
    grad_x = grad_cross_x + grad_along_x
    
    grad_y = grad_cross_y + grad_along_y
    
    
    # Clac Surface Geostrophic current
    g = 9.82
    Omega = 7.292e-5 # s^-1
    f = 2*Omega * np.sin(np.deg2rad(lat))
     
    u = -g/f * (grad_y) 
    v = g/f * (grad_x) 
    
    
    speed = np.sqrt(u**2 + v**2)
    
    return u,v,speed
    
    
#%% SWOT v NK - Loop 
import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, date, timedelta
from netCDF4 import  Dataset,num2date
import cartopy.feature as cfeature
import cftime
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import xarray as xr

# ----------------
# USER INPUT
# --------------

# Area of interest
Global=True # true fullNO, false WC

# Day avg
avg=True

# depth layer 
depth_index = 0             # surface

# SWOT file
load_file = nc.Dataset('./data/process_swot/geoids/july_geoid_XGM2019e_nativ_MDT_filtered.nc', "r")
load_file = nc.Dataset('./data/process_swot/geoids/july_nativ_geoid_nativ_MDT_filtered.nc', "r")

# save file name
save_file="./geoid_comparison/nativgeoid_nativmdt_filtered_NKDA.npz"
save_file="./Github/norkyst_vs_swot/NKDA/nativgeoid_nativmdt_filtered_NKDA.npz"
# -----------------------------
# Load SWOT data
# -----------------------------
swot = {k: load_file.variables[k][:] for k in load_file.variables.keys()}

swot_time = swot['time']
swot_lon = swot['lon']
swot_lat = swot['lat']
swot_ADT=swot['ssha']
swot_u=swot['ugos']
swot_v=swot['vgos']

# convert time
swot_date = np.full(swot_time.shape, np.datetime64('NaT'), dtype='datetime64[ns]')
valid = np.isfinite(swot_time)

converted = cftime.num2date(
    swot_time[valid],
    units='seconds since 2000-01-01 00:00:00',
    calendar='standard'
)

#convert no nan
swot_date_text = np.array([d.isoformat() for d in converted], dtype='datetime64[ns]')

#convert with nan
swot_date[valid]=np.array([d.isoformat() for d in converted], dtype='datetime64[ns]')

#Unique dates for loop 
swot_dates = pd.to_datetime(swot_date_text).normalize().unique()


#% find date and time for each swot pass in file.

# loop over each day where swot data present 

#adt's
swot=[] # swot adt
nk=[]   # norkyst adt
nk_int=[] # norkyst adt in swot grid 

swot_ugos=[]
swot_vgos=[]

nk_ugos=[]
nk_vgos=[]

nk_ugos_int=[]
nk_vgos_int=[]

#coord's
lon=[]    #swot lon
lat=[]    #swot lat

#other
res=[]
TIME=[]

start_date=swot_dates[0]

# ready dac load 
current_month = None
dac = None


for t in swot_dates:
    
    month = t.strftime("%m")
   
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
    
    #Loop over each hour in day
    for i in range(0,24):
        
        #print(i)
        # ---- SWOT --- 
        T=np.datetime64(t+pd.Timedelta(hours=i))
        tmin = np.datetime64(T - pd.Timedelta(minutes=30))
        tmax = np.datetime64(T + pd.Timedelta(minutes=30))
        
        swot_mask = (swot_date >= tmin) & (swot_date < tmax)
        swot_idx = np.where(swot_mask)[0]
    
        if len(swot_idx) == 0:
            continue
        
        # ----------------------------
        # SWOT slice
        # ----------------------------
        lon_s = swot_lon[:, swot_idx]
        lat_s = swot_lat[:, swot_idx]
        adt_s=swot_ADT[:, swot_idx]
        
      
        
        ugos_s,vgos_s,_=geo_curr_swot(lat_s,lon_s,adt_s,2000)
        
        # append
        
        lon.append(lon_s)
        lat.append(lat_s)
        swot.append(adt_s)
        swot_ugos.append(ugos_s)
        swot_vgos.append(vgos_s)
        
    
        # ----- NorKyst800 --- 
        day = pd.Timestamp(T)
        day_str = day.strftime("%Y%m%d")
        
        # Construct the THREDDS file URL for that day
        year=day.strftime("%Y")
        month=day.strftime("%m")
        d=day.strftime("%d")
        hr=day.strftime("%H")
        hr= int(hr)
        url = f"https://thredds.met.no/thredds/dodsC/romshindcast/norkyst_v3/zdepth/{year}/{month}/norkyst800-{day_str}.nc"
        print(f"Processing {day_str} ...")
        try:
            nc = Dataset(url)
        except Exception as e:
            print(f"Could not open {url}: {e}")
            continue
        
        if t==start_date:
            if Global==True:
                lon_min, lon_max = 1, 14
                lat_min, lat_max = 56, 69
                
            else:
                lat_min, lat_max = 58, 64
                lon_min, lon_max = 1.5, 6

                
            lat_nk = nc.variables["lat_rho"][:]
            lon_nk = nc.variables["lon_rho"][:]
            time = nc.variables["ocean_time"][:]
            
            #converted = cftime.num2date(
            #    time,
            #    units='seconds since 1970-01-01 00:00:00',
            #    calendar='standard'
            #)
            #print(converted)
            # Find grid indices for the bounding box
            
            mask = (lat_nk >= lat_min) & (lat_nk <= lat_max) & (lon_nk >= lon_min) & (lon_nk <= lon_max)
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
    
            lat_sub = lat_nk[y0:y1, x0:x1]
            lon_sub = lon_nk[y0:y1, x0:x1]
            
            lon_nk=lon_sub
            lat_nk=lat_sub
            
            # for interp
            NK_points = np.column_stack((lon_nk.ravel(), lat_nk.ravel()))
        
        
        if avg==True:
            zeta = nc.variables["zeta"][:, y0:y1, x0:x1].mean(axis=0)
        else:
            zeta = nc.variables["zeta"][hr, y0:y1, x0:x1]
        
        if Global==False:   
            nk.append(zeta)
        
        
        # interpolat on swot 
        
        swot_points = np.column_stack((lon_s.ravel(), lat_s.ravel()))
        
        NK_vals_zeta = zeta.ravel()
        
        zeta_nk_swot=griddata(NK_points,NK_vals_zeta,swot_points,method='linear').reshape(lon_s.shape)
        
        # ==========================
        # DAC -> SWOT -> Correct NK
        # ==========================

        dac_day = dac["dac"].sel(
            time=t.strftime("%Y-%m-%d")
        )
        
        lon_dac,lat_dac=np.meshgrid(dac["longitude"].values,dac["latitude"].values)
        
        # interpolate DAC regular grid -> NorKyst curvilinear grid
        DAC_points = np.column_stack(
            (
                lon_dac.ravel(),
                lat_dac.ravel()
            )
        )
        
        
        #
        dac_corr = griddata(
            DAC_points,
            dac_day.values.ravel(),
            swot_points,
            method="cubic"
        ).reshape(lon_s.shape)
        
        
        zeta_nk_swot=zeta_nk_swot-dac_corr
        
        # CALC GEO CURR
        
        
        u_nk_swot,v_nk_swot,_=geo_curr_swot(lat_s,lon_s,zeta_nk_swot,2000)
        
        # append
        nk_int.append(zeta_nk_swot)
        nk_ugos_int.append(u_nk_swot)
        nk_vgos_int.append(v_nk_swot)
                
        # residual
        r=adt_s-zeta_nk_swot

        res.append(r)
        TIME.append(T)
      


#%% Save - loop results
save_file="./Github/norkyst_vs_swot/NKDA/nativgeoid_nativmdt_filtered_NKDA.npz"

time_obj = np.empty(len(TIME), dtype=object)
time_obj[:] = TIME

swot_obj = np.empty(len(swot), dtype=object)
swot_obj[:] = swot

#nk_obj = np.empty(len(nk), dtype=object)
#nk_obj[:] = nk

nk_int_obj = np.empty(len(nk_int), dtype=object)
nk_int_obj[:] = nk_int

res_obj = np.empty(len(res), dtype=object)
res_obj[:] = res

lon_obj = np.empty(len(lon), dtype=object)
lon_obj[:] = lon

lat_obj = np.empty(len(lat), dtype=object)
lat_obj[:] = lat

swot_ugos_obj = np.empty(len(swot_ugos), dtype=object)
swot_ugos_obj[:] = swot_ugos

swot_vgos_obj = np.empty(len(swot_vgos), dtype=object)
swot_vgos_obj[:] = swot_vgos

nk_ugos_obj = np.empty(len(nk_ugos_int), dtype=object)
nk_ugos_obj[:] = nk_ugos_int

nk_vgos_obj = np.empty(len(nk_vgos_int), dtype=object)
nk_vgos_obj[:] = nk_vgos_int


np.savez(
    save_file,
    swot=swot_obj,
    #nk=nk_obj,
    nk_int=nk_int_obj,
    res=res_obj,
    lon=lon_obj,
    lat=lat_obj,
    lon_nk=lon_nk,
    lat_nk=lat_nk,
    swot_vgos=swot_vgos_obj,
    swot_ugos=swot_ugos_obj,
    nk_ugos_int=nk_ugos_obj,
    nk_vgos_int=nk_vgos_obj,
    time=time_obj
)
print(f"Saved {save_file}")

