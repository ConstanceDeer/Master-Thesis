# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:43:52 2026

@author: const
"""

        
#%%
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
    

# depth layer 
depth_index = 0             # surface

# SWOT file
load_file = nc.Dataset('./data/process_swot/geoids/july_nativ_geoid_nativ_MDT_filtered.nc', "r")


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


# -----------------------------
# Output containers
# -----------------------------
swot_adt_list = []
nk_adt_list = []
nk_int_list = []

swot_ugos_list = []
swot_vgos_list = []

nk_ugos_list = []
nk_vgos_list = []

nk_ugos_int_list = []
nk_vgos_int_list = []

lon_list = []
lat_list = []

date_list = []

# -----------------------------
# Choose date interval
# -----------------------------
start_date = pd.Timestamp("2025-07-09")
end_date   = pd.Timestamp("2025-07-16")

selected_dates = swot_dates[
    (swot_dates >= start_date) &
    (swot_dates <= end_date)
]

# ready dac load 
current_month = None
dac = None


for day in selected_dates:

    print("Processing:", day)

    
    month = day.strftime("%m")
   
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

    # ==========================
    # Load NorKyst daily mean
    # ==========================

    day = pd.Timestamp(day)

    day_str = day.strftime("%Y%m%d")
    year = day.strftime("%Y")
    month = day.strftime("%m")


    url = (
        f"https://thredds.met.no/thredds/dodsC/"
        f"romshindcast/norkyst_v3/zdepth/"
        f"{year}/{month}/"
        f"norkyst800-{day_str}.nc"
    )
    #url =f"https://thredds.met.no/thredds/dodsC/sea_norshelf_files/{year}/{month}/norshelf_qck_an_{day_str}T00Z.nc"


    try:
        nc = Dataset(url)

    except Exception as e:
        print("Could not open:", url)
        continue



    # Load NorKyst grid once
    if 'NK_points' not in locals():

        lat_nk = nc.variables["lat_rho"][:]
        lon_nk = nc.variables["lon_rho"][:]


        mask = (
            (lat_nk >= 56) &
            (lat_nk <= 69) &
            (lon_nk >= 1) &
            (lon_nk <= 14)
        )


        ys,xs = np.where(mask)

        y0,y1 = ys.min(), ys.max()
        x0,x1 = xs.min(), xs.max()


        lat_nk = lat_nk[y0:y1,x0:x1]
        lon_nk = lon_nk[y0:y1,x0:x1]


        NK_points = np.column_stack(
            (
                lon_nk.ravel(),
                lat_nk.ravel()
            )
        )



    # daily mean NorKyst SSH
    #zeta_day = (
    #    nc.variables["zeta"]
    #    [:,y0:y1,x0:x1]
    #    .mean(axis=0)
    #)
    
    zeta_day = (
        nc.variables["zeta"]
        [0,y0:y1,x0:x1]
    )



    # ==========================
    # Find SWOT observations that day
    # ==========================

    swot_idx = np.where(
        swot_date.astype('datetime64[D]') == 
        np.datetime64(day)
    )[0]


    if len(swot_idx)==0:
        continue



    # one or more SWOT passes that day
    lon_s = swot_lon[:,swot_idx]
    lat_s = swot_lat[:,swot_idx]
    adt_s = swot_ADT[:,swot_idx]



    # ==========================
    # Interpolate NorKyst -> SWOT
    # ==========================

    swot_points = np.column_stack(
        (
            lon_s.ravel(),
            lat_s.ravel()
        )
    )


    zeta_nk_swot = griddata(
        NK_points,
        zeta_day.ravel(),
        swot_points,
        method="linear"
    ).reshape(lon_s.shape)
    
    # ==========================
    # DAC -> SWOT -> Correct NK
    # ==========================

    
    dac_day = dac["dac"].sel(
        time=day.strftime("%Y-%m-%d")
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
    # ==========================
    # calc geo curr's
    # ==========================

    # currents
    swot_u, swot_v,_ = geo_curr_swot(
        lat_s,
        lon_s,
        adt_s,
        2000
    )


    nk_u, nk_v,_ = geo_curr_swot(
        lat_s,
        lon_s,
        zeta_nk_swot,
        2000
    )


    # save
    lon_list.append(lon_s)
    lat_list.append(lat_s)

    swot_adt_list.append(adt_s)
    nk_int_list.append(zeta_nk_swot)

    swot_ugos_list.append(swot_u)
    swot_vgos_list.append(swot_v)

    nk_ugos_int_list.append(nk_u)
    nk_vgos_int_list.append(nk_v)

    date_list.append(day)
    
#%%
for i in range(len(date_list)):

    speed = np.sqrt(
        swot_ugos_list[i]**2 +
        swot_vgos_list[i]**2
    )


    lon = np.ma.filled(lon_list[i], np.nan)
    lat = np.ma.filled(lat_list[i], np.nan)
    speed= np.ma.filled(speed, np.nan)


    speed_plot = speed


    plt.figure(figsize=(7,6))


    im = plt.pcolormesh(
        lon,
        lat,
        speed_plot,
        shading="auto",
        cmap="viridis"
    )


    plt.colorbar(
        im,
        label="Current speed (m/s)"
    )

    plt.title(
        f"SWOT geostrophic speed\n{date_list[i]}"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.axis("equal")

    plt.show()
    
    
#%% Make ready for plot 
print(type(swot_ugos_list))
print(len(swot_ugos_list))

print(swot_ugos_list[0].shape)
print(swot_vgos_list[0].shape)

print(np.nanmin(swot_ugos_list[0]))
print(np.nanmax(swot_ugos_list[0]))

print(np.nanmin(swot_vgos_list[0]))
print(np.nanmax(swot_vgos_list[0]))

# Coordinates
lon_all = np.concatenate([
    np.ma.filled(x, np.nan).ravel()
    for x in lon_list
])

lat_all = np.concatenate([
    np.ma.filled(x, np.nan).ravel()
    for x in lat_list
])


# Speed magnitude for each pass
speed_all_s = np.concatenate([
    np.ma.filled(
        np.sqrt(u**2 + v**2),
        np.nan
    ).ravel()
    for u, v in zip(swot_ugos_list, swot_vgos_list)
])

# Speed magnitude for each pass
speed_all_n = np.concatenate([
    np.ma.filled(
        np.sqrt(u**2 + v**2),
        np.nan
    ).ravel()
    for u, v in zip(nk_ugos_int_list, nk_vgos_int_list)
])


# Remove invalid points
valid = (
    np.isfinite(lon_all) &
    np.isfinite(lat_all) &
    np.isfinite(speed_all_s)
)

lon_all_s = lon_all[valid]
lat_all_s = lat_all[valid]
speed_all_s = speed_all_s[valid]


# Remove invalid points
valid = (
    np.isfinite(lon_all) &
    np.isfinite(lat_all) &
    np.isfinite(speed_all_n)
)

lon_all_n = lon_all[valid]
lat_all_n = lat_all[valid]
speed_all_n = speed_all_n[valid]

#%% Plot

import cartopy.crs as ccrs
import matplotlib.pyplot as plt


lon_min, lon_max = 1, 14
lat_min, lat_max = 56, 69


def plot_scatter_map(lon, lat, field, title,
                     cmap="viridis", vmin=None, vmax=None):

    

    fig = plt.figure(figsize=(8,7), dpi=150)

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    ax.set_extent(
        [lon_min,lon_max,lat_min,lat_max],
        crs=ccrs.PlateCarree()
    )

    ax.coastlines(resolution="10m")
    ax.add_feature(
        cfeature.LAND,
        facecolor="black",
        zorder=1
    )

    ax.coastlines(resolution="10m")


    gl = ax.gridlines(
        draw_labels=True,
        linestyle="--",
        alpha=0.5
    )

    gl.top_labels = False
    gl.right_labels = False


    ax.set_extent(
        [lon_min, lon_max, lat_min, lat_max],
        crs=ccrs.PlateCarree()
    )


    sc = ax.scatter(
        lon,
        lat,
        c=field,
        s=0.1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )


    plt.colorbar(
        sc,
        shrink=1,
        pad=0.01,
        aspect=40,
        label="Geostrophic speed [m/s]"
    )
    ax.set_aspect(1/np.cos(62*np.pi/180))


    plt.title(title)

    plt.show()
    
plot_scatter_map(
    lon_all_s,
    lat_all_s,
    speed_all_s,
    "",
    cmap="Reds",
    vmin=0,
    vmax=1
)

plot_scatter_map(
    lon_all_n,
    lat_all_n,
    speed_all_n,
    "",
    cmap="Reds",
    vmin=0,
    vmax=1
)
