# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:52:00 2026

@author: const


Code used for making SWOT files used in HF vs SWOT comparison 
"""
#%% Inital SWOT files with all nativ reference planes.

# L3 only 
import os
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import scipy
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from datetime import datetime, date, timedelta
import xarray as xr
from matplotlib.colors import ListedColormap
import cartopy.feature as cfeature
from netCDF4 import num2date



# ==========================
# INITIAL ARRAYS
# ==========================
ssha_all = ma.zeros([69, 0])
lat_all  = ma.zeros([69, 0])
lon_all  = ma.zeros([69, 0])
time_all = ma.zeros([69, 0])
xac_all  = ma.zeros([69, 0])
ugos_all  = ma.zeros([69, 0])
vgos_all  = ma.zeros([69, 0])
longos_all  = ma.zeros([69, 0])
latgos_all  = ma.zeros([69, 0])

# ==========================
# PATHS 
# ==========================

# where to load SWOT files
base_path = "D:/swot/"

nc_files = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".nc"):
            nc_files.append(os.path.join(root, file))

L3_list = nc_files


# ==========================
# OBSERVATION TIMES, not used but needs to be there 
# ==========================
obslist = os.listdir('./observations/test_norkyst_init_5/')
obstime = np.zeros(len(obslist), dtype=datetime)
for i in range(len(obslist)):
    obstime[i] = datetime.strptime(obslist[i][-20:-6], 'y%Ym%md%dh%H')

# ==========================
# SETTINGS
# ==========================
plot = 0
plotobs = 0
#xovercor = 'L3'   # FORCE L3 MODE
unedit = 0

# ==========================
# Land mask
# ==========================
ds = xr.open_dataset("data/LandMask_gebco.nc")
mask = ds["mask"]
lat = ds["lat"]
lon = ds["lon"]


# start plot if needed

if plot==1:
    
    #%
    lon_min, lon_max = 1, 14
    lat_min, lat_max = 56, 69
    
    proj = ccrs.Orthographic(
        central_longitude=(lon_min+lon_max)/2,
        central_latitude=(lat_min+lat_max)/2
    )
    
    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=300, subplot_kw={'projection': proj})
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max])
    
    ax1.coastlines(resolution='50m')
    ax1.add_feature(cfeature.LAND)
    ax1.gridlines(draw_labels=True, dms=True)
    
    # Manual landmask plot (pcolormesh)
    cmap = ListedColormap(["lightsteelblue", "dimgrey"])  # ocean, land
    pcm = ax1.pcolormesh(
        lon, lat, mask,
        cmap=cmap,
        shading='auto',
        alpha=0.7,
        transform=proj
    )
    

# ==========================
# MAIN LOOP 
# ==========================

for i in range(len(L3_list)): #len(L3_list)):

    print(L3_list[i]) # print file in process
 

    # --------------------------
    # LOAD 
    
    # Open the file
    load_file = nc.Dataset( L3_list[i], "r")
    
    # Read all variables into a dictionary as NumPy arrays
    L3_swot = {k: load_file.variables[k][:] for k in load_file.variables.keys()}
    
    # --------------------------
    # DATE FROM FILENAME
 
    date2 = datetime.strptime(L3_list[i][43:54], '%Y%m%dT%H')

    # --------------------------
    # READ VARIABLES 
    lat = L3_swot['latitude']
    lon = L3_swot['longitude']
    ssha = L3_swot['ssha_filtered']
    qual = L3_swot['quality_flag']
    time = L3_swot['time']
    xac = L3_swot['cross_track_distance']
    cal = L3_swot['calibration']
    mss = L3_swot['mss']
    mdt = L3_swot['mdt']
    dac = L3_swot['dac']
    internaltide = L3_swot['internal_tide']
    oceantides = L3_swot['ocean_tide']
   
    #calc ADT
    dac=dac+internaltide+oceantides
    ssha = ssha
    ssha = ssha+mdt#+dac
    #ssha = ssha+internaltide+oceantides
    

    xac = np.tile(xac[np.newaxis, :], (lat.shape[0], 1))
    
   
    # --------------------------
    # LONGITUDE -180 to 180
    # --------------------------

    lon = np.where(lon>180,lon-360,lon)
    
    # --------------------------
    # Lask needed longitude 
    # --------------------------
    mask_lon = (lon <= 1) | (lon >= 14)  # True where we want to remove   
    time = ma.masked_array(time,mask=mask_lon[:,0])
    
    ssha=ma.masked_array(ssha,mask=mask_lon)
    lon=ma.masked_array(lon,mask=mask_lon)
    lat=ma.masked_array(lat,mask=mask_lon)
    xac=ma.masked_array(xac,mask=mask_lon)
    qual= ma.masked_array(qual, mask=mask_lon)
    dac= ma.masked_array(dac, mask=mask_lon)

    # --------------------------
    # Location filter

    mask_loc = np.where((lon >1) & (lon < 14) &
                        (lat > 56) & (lat < 69), 0, 1)

    
    mask = mask_loc
    ssha=ma.masked_array(ssha,mask=mask)
    lon=ma.masked_array(lon,mask=mask)
    lat=ma.masked_array(lat,mask=mask)
    xac=ma.masked_array(xac,mask=mask)
    dac=ma.masked_array(dac,mask=mask)
    ssha_plot2=ssha
    
    print(np.nanmean(dac))
    
   
    # --------------------------
    # QUALITY FILTERING
   
    
    if unedit==0:
        # Mask for bad quality
        mask_qual = qual > 1
        
        # Mask for SSHA extremes
        mask_ampl = np.abs(ssha) > 1000
        
        # Optional: region mask
        mask_loc = (lon < 1) | (lon > 14) | (lat < 56) | (lat > 69)
        
        # Combine masks
        mask = mask_qual | mask_ampl | mask_loc
        
        
        ssha=ma.masked_array(ssha,mask=mask)
        lon=ma.masked_array(lon,mask=mask)
        lat=ma.masked_array(lat,mask=mask)
        xac=ma.masked_array(xac,mask=mask)

    
    ssha_extra=ssha
   
    
    # ----- Remove rows completly masked out 
    
    ssha=ma.masked_array(ssha,mask=np.isnan(ssha))
    ssha_plot4 = ssha
    mask_loc = np.where((lon >1) & (lon < 14) &
                        (lat > 56) & (lat < 69), 0, 1)
    mask = mask|mask_loc
    ssha=ma.masked_array(ssha,mask=mask)

    # Plot things 
    lon_plot=lon
    lat_plot=lat
        
    
    maskbool=ma.getmask(ssha)
    ssha = ssha[~np.all(maskbool == True, axis=1)]
    lon = lon[~np.all(maskbool == True, axis=1)]
    lat = lat[~np.all(maskbool == True, axis=1)]
    time = time[~np.all(maskbool == True, axis=1)]
    xac = xac[~np.all(maskbool == True, axis=1)]
    
    
    if np.all(maskbool):
        continue
    

    # select only valid (unmasked) values
    lon_valid = lon.compressed()
    lat_valid = lat.compressed()
    ssha_valid = ssha.compressed()
    
    # --------------------------
    # calc geostrofic currents 
    # --------------------------
    
    # get lon and lat in rad
    
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    dlat_y, dlat_x = np.gradient(lat_rad)
    dlon_y, dlon_x = np.gradient(lon_rad)
    
    heading = np.arctan2(dlon_y, dlat_y)
    
    heading_mean = np.nanmean(np.rad2deg(heading))
    
    orbit_angle = heading   # already same shape as ssha
    
    # calc gradients in ADT in along and cross track
    
    grad_along, grad_cross = np.gradient(ssha)
    
    grad_cross /= 2.0 # Convert unit to m / km
    
    grad_along /= 2.0 # Convert unit to m / km
    
   
    # calc u and v, descending v ascending is a feature needed for L2 data
    # have not bothered removeing totally, 
    
    
    if (heading_mean > 90) or (heading_mean < -90):# descending
    
        correction_angle = orbit_angle
        
        
    
        # Along track
        grad_along_x = grad_along*np.sin(orbit_angle)
    
        grad_along_y = grad_along*np.cos(orbit_angle)
    
     
    
        # Cross track
    
        grad_cross_x = grad_cross*np.cos(orbit_angle)
    
        grad_cross_y = -grad_cross*np.sin(orbit_angle)
       
    
    else: # ascending 
    
        # Along track
    
        grad_along_x = grad_along*np.sin(orbit_angle)
    
        grad_along_y = grad_along*np.cos(orbit_angle)
    
     
    
        # Cross track
    
        grad_cross_x = grad_cross*np.cos(orbit_angle)
    
        grad_cross_y = -grad_cross*np.sin(orbit_angle)
        
    
    # calc lon/lat gradients
    
    grad_x = grad_cross_x + grad_along_x
    
    grad_y = grad_cross_y + grad_along_y
    
    
    # ClacSurface Geostrophic current
    g = 9.82
    Omega = 7.292e-5 # s^-1
    f = 2*Omega * np.sin(np.deg2rad(lat))
    
     
    u = -g/f * (grad_y/1000) # convert gradient to m/m
    v = g/f * (grad_x/1000) # convert gradient to m/m
    speed = np.sqrt(u**2 + v**2)
    
    # for plot
    u_valid = u.compressed()
    
    
        
    if plot==1:
        sc = ax1.scatter(
            lon_valid, lat_valid, c=u_valid,
            cmap='RdBu_r',
            vmin=-0.25,
            vmax=0.25,
            s=0.5,
            transform=ccrs.PlateCarree()
        )
        
        
    elif plot==2:
        

            lon_min, lon_max = 1, 14
            lat_min, lat_max = 56, 69
            
            proj = ccrs.Orthographic(
                central_longitude=(lon_min+lon_max)/2,
                central_latitude=(lat_min+lat_max)/2
            )
            
            plt.ion()
            fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=300, subplot_kw={'projection': proj})
            ax2.set_extent([lon_min, lon_max, lat_min, lat_max])
            
            ax2.coastlines(resolution='50m')
            ax2.add_feature(cfeature.LAND)
            ax2.gridlines(draw_labels=True, dms=True)
            
            sc2 = ax2.scatter(
                lon, lat, c=v,
                cmap='RdBu_r',
                vmin=-0.25,
                vmax=0.25,
                s=0.5,
                transform=ccrs.PlateCarree()
            )
            
            plt.colorbar(sc2,ax=ax2, shrink=0.7, label='SSHA [m]')
            plt.title(f'{date2}', fontsize=14)
            plt.show()
    
    # save for after loop 
    
    ssha_all=ma.hstack([ssha_all,ssha.T])
    lon_all=ma.hstack([lon_all,lon.T])
    lat_all=ma.hstack([lat_all,lat.T])
    time_all=ma.append(time_all,time)
    xac_all=ma.hstack([xac_all,xac.T/1000])
    ugos_all=ma.hstack([ugos_all,u.T])
    vgos_all=ma.hstack([vgos_all,v.T])
    
 

if plot==1:
    # After the loop, add colorbar and title
    plt.colorbar(sc,ax=ax1, shrink=0.7, label='SSHA [m]')
    plt.title('All L3 passes overlapping', fontsize=14)
    plt.show()
  
# fill all empty values with  NaN 
  
lon_all=lon_all.filled(fill_value=np.nan)
lat_all=lat_all.filled(fill_value=np.nan)
ssha_all=ssha_all.filled(fill_value=np.nan)
xac_all=xac_all.filled(fill_value=np.nan)
ugos_all=ugos_all.filled(fill_value=np.nan)
vgos_all=vgos_all.filled(fill_value=np.nan)

# make it so not maksed array anymore 
mask=np.where(np.isnan(lon_all),1,0)

lon_all=ma.masked_array(lon_all,mask=mask)
lat_all=ma.masked_array(lat_all,mask=mask)
ssha_all=ma.masked_array(ssha_all,mask=mask)
xac_all=ma.masked_array(xac_all,mask=mask)
ugos_all=ma.masked_array(ugos_all,mask=mask)
vgos_all=ma.masked_array(vgos_all,mask=mask)

#% Save data in file 
ncfile = nc.Dataset('./swot/JJAS_nativ_filtered.nc',mode='w',format='NETCDF4_CLASSIC') 


nc_dim = ncfile.createDimension('nC',69)
time_dim = ncfile.createDimension('time',np.size(lon_all,1))

lon = ncfile.createVariable('lon', np.float32, ('nC','time'))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
lat = ncfile.createVariable('lat', np.float32, ('nC','time'))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'seconds since 2000-01-01 00:00:00.0'
time.long_name = 'time in UTC'
ssha = ncfile.createVariable('ssha', np.float32, ('nC','time'))
ssha.units = 'm'
ssha.long_name = 'sea surface height anomaly'
xac = ncfile.createVariable('x_ac', np.float32, ('nC','time'))
xac.units = 'km'
xac.long_name = 'Across track distance from nadir'

ugos = ncfile.createVariable('ugos', np.float32, ('nC','time'))
ugos.units = 'm/s'
ugos.long_name = 'Dervied Geostrophic currect from SWOTs ssh'

vgos = ncfile.createVariable('vgos', np.float32, ('nC','time'))
vgos.units = 'm/s'
vgos.long_name = 'Dervied Geostrophic currect from SWOTs ssh'

lon[:,:] = lon_all
lat[:,:] = lat_all
time[:] = time_all.filled(np.nan)
time[:] = np.ma.filled(time_all, np.nan)
ssha[:,:] = ssha_all
ugos[:,:] = ugos_all
vgos[:,:] = vgos_all
xac[:,:] = xac_all

ncfile.close()


#%% SWOT data with different MDT

# L3 only 
import os
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import scipy
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from datetime import datetime, date, timedelta
import xarray as xr
from matplotlib.colors import ListedColormap
import cartopy.feature as cfeature
from netCDF4 import num2date
from scipy.interpolate import griddata
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# ==========================
# INITIAL ARRAYS
# ==========================
ssha_all = ma.zeros([69, 0])
lat_all  = ma.zeros([69, 0])
lon_all  = ma.zeros([69, 0])
time_all = ma.zeros([69, 0])
xac_all  = ma.zeros([69, 0])
ugos_all  = ma.zeros([69, 0])
vgos_all  = ma.zeros([69, 0])
longos_all  = ma.zeros([69, 0])
latgos_all  = ma.zeros([69, 0])

# ==========================
# PATHS 
# ==========================

# where to load SWOT files
base_path = "D:/swot/augu/"

nc_files = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".nc"):
            nc_files.append(os.path.join(root, file))

L3_list = nc_files


# ==========================
# OBSERVATION TIMES, not used but needs to be there 
# ==========================
obslist = os.listdir('./observations/test_norkyst_init_5/')
obstime = np.zeros(len(obslist), dtype=datetime)
for i in range(len(obslist)):
    obstime[i] = datetime.strptime(obslist[i][-20:-6], 'y%Ym%md%dh%H')

# ==========================
# SETTINGS
# ==========================
plot = 0
plotobs = 0
#xovercor = 'L3'   # FORCE L3 MODE
unedit = 0

# ==========================
# Land mask
# ==========================
ds = xr.open_dataset("data/LandMask_gebco.nc")
mask = ds["mask"]
lat = ds["lat"]
lon = ds["lon"]

# ==========================
# MDT
# ==========================

# Filename of the saved subset
import xarray as xr
import numpy as np
from scipy.interpolate import griddata


file_nk_mdt="C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/mdt/norkyst_mdt_BIcorr_2025.nc"
file_swot_mdt='C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/mdt/mdt_hybrid_cnes_cls22_cmems2020_global.nc'
file_dtu_mdt = "C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/mdt/DTUUH22MDT.xyz"


dnk = xr.open_dataset(file_nk_mdt)

lon_mdt = dnk["lon"].values
lat_mdt = dnk["lat"].values
mdt_norkyst = dnk["mdt"].values


## Load xyz file
#dtu_data = np.loadtxt(file_dtu_mdt)

#lon_mdt = dtu_data[:, 0]
#lat_mdt = dtu_data[:, 1]
#mdt_dtu = dtu_data[:, 2]




if plot==1:
    
    #%
    lon_min, lon_max = 1, 14
    lat_min, lat_max = 56, 69
    
    proj = ccrs.Orthographic(
        central_longitude=(lon_min+lon_max)/2,
        central_latitude=(lat_min+lat_max)/2
    )
    
    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=300, subplot_kw={'projection': proj})
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max])
    
    ax1.coastlines(resolution='50m')
    ax1.add_feature(cfeature.LAND)
    ax1.gridlines(draw_labels=True, dms=True)
    
    # Manual landmask plot (pcolormesh)
    cmap = ListedColormap(["lightsteelblue", "dimgrey"])  # ocean, land
    pcm = ax1.pcolormesh(
        lon, lat, mask,
        cmap=cmap,
        shading='auto',
        alpha=0.7,
        transform=proj
    )
    

# ==========================
# MAIN LOOP 
# ==========================

for i in range(len(L3_list)): #len(L3_list)):

    print(L3_list[i]) # print file in process
 

    # --------------------------
    # LOAD 
    
    # Open the file
    load_file = nc.Dataset( L3_list[i], "r")
    
    # Read all variables into a dictionary as NumPy arrays
    L3_swot = {k: load_file.variables[k][:] for k in load_file.variables.keys()}
    
    # --------------------------
    # DATE FROM FILENAME
 
    date2 = datetime.strptime(L3_list[i][43:54], '%Y%m%dT%H')

    # --------------------------
    # READ VARIABLES 
    lat = L3_swot['latitude']
    lon = L3_swot['longitude']
    ssha = L3_swot['ssha_filtered']
    qual = L3_swot['quality_flag']
    time = L3_swot['time']
    xac = L3_swot['cross_track_distance']
    cal = L3_swot['calibration']
    mss = L3_swot['mss']
    mdt = L3_swot['mdt']
    dac = L3_swot['dac']
    internaltide = L3_swot['internal_tide']
    oceantides = L3_swot['ocean_tide']

    ssha=ssha#+dac+internaltide
    xac = np.tile(xac[np.newaxis, :], (lat.shape[0], 1))
    
   
    # --------------------------
    # LONGITUDE -180 to 180
    # --------------------------

    lon = np.where(lon>180,lon-360,lon)
    
    
    # --------------------------
    # Lask needed longitude 
    # --------------------------
    mask_lon = (lon <= 1) | (lon >= 14)  # True where we want to remove   
    time = ma.masked_array(time,mask=mask_lon[:,0])
    
    ssha=ma.masked_array(ssha,mask=mask_lon)
    lon=ma.masked_array(lon,mask=mask_lon)
    lat=ma.masked_array(lat,mask=mask_lon)
    xac=ma.masked_array(xac,mask=mask_lon)
    qual= ma.masked_array(qual, mask=mask_lon)

    # --------------------------
    # Location filter

    mask_loc = np.where((lon >1) & (lon < 14) &
                        (lat > 56) & (lat < 69), 0, 1)

    
    mask = mask_loc
    ssha=ma.masked_array(ssha,mask=mask)
    lon=ma.masked_array(lon,mask=mask)
    lat=ma.masked_array(lat,mask=mask)
    xac=ma.masked_array(xac,mask=mask)
    ssha_plot2=ssha
    
    # --- Interpolate MDT ---
    
    #points = np.column_stack((lon_mdt.values.ravel(), lat_mdt.values.ravel()))
    points = np.column_stack((lon_mdt.ravel(), lat_mdt.ravel()))
    #values = mdt_norkyst.values.ravel()
    values = mdt_norkyst.ravel()
    
    points_swot = np.column_stack((lon.ravel(), lat.ravel()))
    
    mask = ~np.isnan(values)
    
    # 1. Linear interpolation (no extrapolation)
    mdt_linear = griddata(
        points[mask],
        values[mask],
        points_swot,
        method='cubic'
    )
    
    # 2. Distance check using nearest neighbour
    tree = cKDTree(points[mask])
    dist, _ = tree.query(points_swot, k=1)
    
    max_dist = 2000.0  # 2 km (must match coordinate units)
    
    # 3. Enforce cutoff: only keep values within 2 km AND valid linear result
    valid = (dist <= max_dist) & (~np.isnan(mdt_linear))
    
    mdt_interp = np.full(points_swot.shape[0], np.nan)
    mdt_interp[valid] = mdt_linear[valid]
    mdt_interp = mdt_interp.reshape(lon.shape)
    
    
    #calc ADT
    ssha = ssha
    ssha = ssha+mdt_interp #+dac
    #ssha = ssha+internaltide+oceantides
    
    # CHACK PLOT FOR MDT INTERPOLATION
    
    #mask = np.ma.getmaskarray(lon) | np.ma.getmaskarray(lat)

    #%
    #lon_min, lon_max = 1, 14
    #lat_min, lat_max = 56, 69
    
    #proj = ccrs.Orthographic(
    #    central_longitude=(lon_min+lon_max)/2,
    #    central_latitude=(lat_min+lat_max)/2
    #)
    
    #plt.ion()
    #fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=300, subplot_kw={'projection': proj})
    #ax1.set_extent([lon_min, lon_max, lat_min, lat_max])
    
    #ax1.coastlines(resolution='50m')
    #ax1.add_feature(cfeature.LAND)
    #ax1.gridlines(draw_labels=True, dms=True)
    
    
    #pcm = ax1.pcolormesh(
    #    lon.data,
    #    lat.data,
    #    np.ma.masked_where(mask, mdt_interp),
    #    cmap="RdBu_r",
    #    shading="auto",
    #    transform=ccrs.PlateCarree()
    #)
    
    #plt.show()
    
   
    # --------------------------
    # QUALITY FILTERING
   
    
    if unedit==0:
        # Mask for bad quality
        mask_qual = qual > 1
        
        # Mask for SSHA extremes
        mask_ampl = np.abs(ssha) > 1000
        
        # Optional: region mask
        mask_loc = (lon < 1) | (lon > 14) | (lat < 56) | (lat > 69)
        
        # Combine masks
        mask = mask_qual | mask_ampl | mask_loc
        
        
        ssha=ma.masked_array(ssha,mask=mask)
        lon=ma.masked_array(lon,mask=mask)
        lat=ma.masked_array(lat,mask=mask)
        xac=ma.masked_array(xac,mask=mask)

    
    ssha_extra=ssha
   
    
    # ----- Remove rows completly masked out 
    
    ssha=ma.masked_array(ssha,mask=np.isnan(ssha))
    ssha_plot4 = ssha
    mask_loc = np.where((lon >1) & (lon < 14) &
                        (lat > 56) & (lat < 69), 0, 1)
    mask = mask|mask_loc
    ssha=ma.masked_array(ssha,mask=mask)

    # Plot things 
    lon_plot=lon
    lat_plot=lat
        
    
    maskbool=ma.getmask(ssha)
    ssha = ssha[~np.all(maskbool == True, axis=1)]
    lon = lon[~np.all(maskbool == True, axis=1)]
    lat = lat[~np.all(maskbool == True, axis=1)]
    time = time[~np.all(maskbool == True, axis=1)]
    xac = xac[~np.all(maskbool == True, axis=1)]
    
    
    if np.all(maskbool):
        continue
    

    # select only valid (unmasked) values
    lon_valid = lon.compressed()
    lat_valid = lat.compressed()
    ssha_valid = ssha.compressed()
    
    # --------------------------
    # calc geostrofic currents 
    # --------------------------
    
    # get lon and lat in rad
    

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    dlat_y, dladt_x = np.gradient(lat_rad)
    dlon_y, dlon_x = np.gradient(lon_rad)
    
    heading = np.arctan2(dlon_y, dlat_y)
    
    heading_mean = np.nanmean(np.rad2deg(heading))
    
    orbit_angle = heading   # already same shape as ssha

    # calc gradients in ADT in along and cross track
    
    grad_along, grad_cross = np.gradient(ssha)
    
    grad_cross /= 2.0 # Convert unit to m / km
    
    grad_along /= 2.0 # Convert unit to m / km
    
   
    # calc u and v, dependend on if pass ascending or descending. 
    
    if (heading_mean > 90) or (heading_mean < -90):# descending
    
        correction_angle = orbit_angle
        
        
    
        # Along track
        grad_along_x = grad_along*np.sin(orbit_angle)
    
        grad_along_y = grad_along*np.cos(orbit_angle)
    
     
    
        # Cross track
    
        grad_cross_x = grad_cross*np.cos(orbit_angle)
    
        grad_cross_y = -grad_cross*np.sin(orbit_angle)
       
    
    else: # ascending 
    
        # Along track
    
        grad_along_x = grad_along*np.sin(orbit_angle)
    
        grad_along_y = grad_along*np.cos(orbit_angle)
    
     
    
        # Cross track
    
        grad_cross_x = grad_cross*np.cos(orbit_angle)
    
        grad_cross_y = -grad_cross*np.sin(orbit_angle)
        
    
    # calc lon/lat gradients
    
    grad_x = grad_cross_x + grad_along_x
    
    grad_y = grad_cross_y + grad_along_y
    
    
    # ClacSurface Geostrophic current
    g = 9.82
    Omega = 7.292e-5 # s^-1
    f = 2*Omega * np.sin(np.deg2rad(lat))
    
     
    u = -g/f * (grad_y/1000) # convert gradient to m/m
    v = g/f * (grad_x/1000) # convert gradient to m/m
    speed = np.sqrt(u**2 + v**2)
    
    # for plot
    u_valid = u.compressed()
    
    
        
    if plot==1:
        sc = ax1.scatter(
            lon_valid, lat_valid, c=u_valid,
            cmap='RdBu_r',
            vmin=-0.25,
            vmax=0.25,
            s=0.5,
            transform=ccrs.PlateCarree()
        )
        
        
    elif plot==2:
        

            lon_min, lon_max = 1, 14
            lat_min, lat_max = 56, 69
            
            proj = ccrs.Orthographic(
                central_longitude=(lon_min+lon_max)/2,
                central_latitude=(lat_min+lat_max)/2
            )
            
            plt.ion()
            fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=300, subplot_kw={'projection': proj})
            ax2.set_extent([lon_min, lon_max, lat_min, lat_max])
            
            ax2.coastlines(resolution='50m')
            ax2.add_feature(cfeature.LAND)
            ax2.gridlines(draw_labels=True, dms=True)
            
            
            sc2 = ax2.scatter(
                lon, lat, c=np.ma.filled(speed, np.nan),
                cmap='Reds',
                vmin=0,
                vmax=1,
                s=0.5,
                transform=ccrs.PlateCarree()
            )
            
            plt.colorbar(sc2,ax=ax2, shrink=0.7, label='SSHA [m]')
            plt.title(f'{date2}', fontsize=14)
            plt.show()
    
    # save for after loop 
    
    ssha_all=ma.hstack([ssha_all,ssha.T])
    lon_all=ma.hstack([lon_all,lon.T])
    lat_all=ma.hstack([lat_all,lat.T])
    time_all=ma.append(time_all,time)
    xac_all=ma.hstack([xac_all,xac.T/1000])
    ugos_all=ma.hstack([ugos_all,u.T])
    vgos_all=ma.hstack([vgos_all,v.T])
    
 

if plot==1:
    # After the loop, add colorbar and title
    plt.colorbar(sc,ax=ax1, shrink=0.7, label='SSHA [m]')
    plt.title('All L3 passes overlapping', fontsize=14)
    plt.show()
  
# fill all empty values with  NaN 
  
lon_all=lon_all.filled(fill_value=np.nan)
lat_all=lat_all.filled(fill_value=np.nan)
ssha_all=ssha_all.filled(fill_value=np.nan)
xac_all=xac_all.filled(fill_value=np.nan)
ugos_all=ugos_all.filled(fill_value=np.nan)
vgos_all=vgos_all.filled(fill_value=np.nan)

# make it so not maksed array anymore 
mask=np.where(np.isnan(lon_all),1,0)

lon_all=ma.masked_array(lon_all,mask=mask)
lat_all=ma.masked_array(lat_all,mask=mask)
ssha_all=ma.masked_array(ssha_all,mask=mask)
xac_all=ma.masked_array(xac_all,mask=mask)
ugos_all=ma.masked_array(ugos_all,mask=mask)
vgos_all=ma.masked_array(vgos_all,mask=mask)

#%% diff geoid 

# L3 only 
import os
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import scipy
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from datetime import datetime, date, timedelta
import xarray as xr
from matplotlib.colors import ListedColormap
import cartopy.feature as cfeature
from netCDF4 import num2date
import pyinterp.backends.xarray


# ==========================
# INITIAL ARRAYS
# ==========================
ssha_all = ma.zeros([69, 0])
lat_all  = ma.zeros([69, 0])
lon_all  = ma.zeros([69, 0])
time_all = ma.zeros([69, 0])
xac_all  = ma.zeros([69, 0])
ugos_all  = ma.zeros([69, 0])
vgos_all  = ma.zeros([69, 0])
longos_all  = ma.zeros([69, 0])
latgos_all  = ma.zeros([69, 0])

# ==========================
# PATHS 
# ==========================

# where to load SWOT files
base_path = "D:/swot/july/"

nc_files = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".nc"):
            nc_files.append(os.path.join(root, file))

L3_list = nc_files


# ==========================
# OBSERVATION TIMES, not used but needs to be there 
# ==========================
obslist = os.listdir('./observations/test_norkyst_init_5/')
obstime = np.zeros(len(obslist), dtype=datetime)
for i in range(len(obslist)):
    obstime[i] = datetime.strptime(obslist[i][-20:-6], 'y%Ym%md%dh%H')

# ==========================
# SETTINGS
# ==========================
plot = 2
plotobs = 0
#xovercor = 'L3'   # FORCE L3 MODE
unedit = 0

# ==========================
# Land mask
# ==========================
ds = xr.open_dataset("data/LandMask_gebco.nc")
mask = ds["mask"]
lat = ds["lat"]
lon = ds["lon"]


# ---- Load geoid ---- #

ds = xr.open_dataset(
    "C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/final/EGG2015.nc",
    chunks=None
)

ds = xr.open_dataset(
    "C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/final/XGM2019e.nc",
    chunks=None
)


#ds = xr.open_dataset(
#    "C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/final/NKG2015.nc",
#    chunks=None
#)



da = ds['geoid'].rename({
    "latitude": "lat",
    "longitude": "lon"
}).load()

# load subset
da = da.sel(lat=slice(56, 69), lon=slice(1, 14))

# make ready for spline
grid = pyinterp.backends.xarray.Grid2D(da, geodetic=True)


# start plot if needed

if plot==1:
    
    #%
    lon_min, lon_max = 1, 14
    lat_min, lat_max = 56, 69
    
    proj = ccrs.Orthographic(
        central_longitude=(lon_min+lon_max)/2,
        central_latitude=(lat_min+lat_max)/2
    )
    
    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=300, subplot_kw={'projection': proj})
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max])
    
    ax1.coastlines(resolution='50m')
    ax1.add_feature(cfeature.LAND)
    ax1.gridlines(draw_labels=True, dms=True)
    
    # Manual landmask plot (pcolormesh)
    cmap = ListedColormap(["lightsteelblue", "dimgrey"])  # ocean, land
    pcm = ax1.pcolormesh(
        lon, lat, mask,
        cmap=cmap,
        shading='auto',
        alpha=0.7,
        transform=proj
    )
    

# ==========================
# MAIN LOOP 
# ==========================

for i in range(len(L3_list)): #len(L3_list)):

    print(L3_list[i]) # print file in process
 
    # --------------------------
    # LOAD L3 FILE
    # --------------------------
    # Open the file
    load_file = nc.Dataset( L3_list[i], "r")
    
    
    # Read all variables into a dictionary as NumPy arrays
    L3_swot = {k: load_file.variables[k][:] for k in load_file.variables.keys()}

    # --------------------------
    # DATE FROM FILENAME
    # --------------------------
    date2 = datetime.strptime(L3_list[i][43:54], '%Y%m%dT%H')

    # --------------------------
    # READ VARIABLES (L3)
    # --------------------------
    lat = L3_swot['latitude']
    lon = L3_swot['longitude']
    ssha = L3_swot['ssha_filtered']
    #surface = L3_swot['surface_classification_flag']
    qual = L3_swot['quality_flag']
    #coast = L3_swot['distance_to_coast']
    #iceconc = L3_swot['ice_conc']
    time = L3_swot['time']
    xac = L3_swot['cross_track_distance']
    cal = L3_swot['calibration']
    
    mss= L3_swot['mss']
    #geoid = L2_swot['geoid']
    mdt = L3_swot['mdt']
    #mss_dtu = L3_swot['mean_sea_surface_dtu']
    internaltide = L3_swot['internal_tide']
    
    ssha = ssha + mss #internaltide

    
    xac = np.tile(xac[np.newaxis, :], (lat.shape[0], 1))
    
   
    # --------------------------
    # LONGITUDE FIX
    # --------------------------

    lon = np.where(lon>180,lon-360,lon)
    
    # --------------------------
    # BASIC MASKS
    # --------------------------
    mask_lon = (lon <= 1) | (lon >= 14)  # True where we want to remove
  
    
    time = ma.masked_array(time,mask=mask_lon[:,0])
    
    
    #%
    ssha=ma.masked_array(ssha,mask=mask_lon)
    mss=ma.masked_array(mss,mask=mask_lon)
    lon=ma.masked_array(lon,mask=mask_lon)
    lat=ma.masked_array(lat,mask=mask_lon)
    xac=ma.masked_array(xac,mask=mask_lon)
    qual      = ma.masked_array(qual, mask=mask_lon)
#%
    # --------------------------
    # L3 CROSSOVER CORRECTION
    # --------------------------
    #%
    
    cal   = ma.masked_array(cal, mask=mask_lon)
    
    # LOCATION VALG
    mask_loc = np.where((lon >1) & (lon < 14) &
                        (lat > 56) & (lat < 69), 0, 1)
    
    
    mask = mask_loc
  
    #%  
    ssha=ma.masked_array(ssha,mask=mask)
    mss_2025=ma.masked_array(mss,mask=mask)
    lon=ma.masked_array(lon,mask=mask)
    lat=ma.masked_array(lat,mask=mask)
    xac=ma.masked_array(xac,mask=mask)
    ssha_plot2=ssha
    
    # ds_geo = xr.open_dataset("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/EGM2008_grid.nc")
    # geoid = griddata(
    #     (ds_geo['lon'].values.flatten(), ds_geo['lat'].values.flatten()),
    #     ds_geo['geoid'].values.flatten(),
    #     (lon, lat),
    #     method='linear'
    # )  
    

    # ds_mss = xr.open_dataset("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/mss_cnes_cls_2022.nc")
    # lat_mss = ds_mss['latitude'].values
    # lon_mss = ds_mss['longitude'].values
    # mss_vals = ds_mss['mssh'].values.astype(np.float32)  # convert to float32 for speed


    # lon2dm, lat2dm = np.meshgrid(lon_mss, lat_mss)
    # # Create interpolator
    # mss_2022 = griddata(
    #     (lon2dm.flatten(), lat2dm.flatten()),
    #     mss_vals.flatten(),
    #     (lon, lat),
    #     method='linear'
    # )  
    # #%%
    # mss_diff=mss_2025-mss_2022
    
    # print(np.nanmean(mss_diff))
    
    # Interpolate
    values = pyinterp.icubic(
        grid,
        lon.ravel(),
        lat.ravel(),
        bounds_error=False,
        num_threads=0   
        )
    
    geoid = values.reshape(lon.shape)
    
    
    ssha=ssha-geoid  #ADT
   
    # --------------------------
    # QUALITY FILTERING
   
    
    if unedit==0:
        # Mask for bad quality
        mask_qual = qual > 1
        
        # Mask for SSHA extremes
        mask_ampl = np.abs(ssha) > 1000
        
        # Optional: region mask
        mask_loc = (lon < 1) | (lon > 14) | (lat < 56) | (lat > 69)
        
        # Combine masks
        mask = mask_qual | mask_ampl | mask_loc
        
        
        ssha=ma.masked_array(ssha,mask=mask)
        lon=ma.masked_array(lon,mask=mask)
        lat=ma.masked_array(lat,mask=mask)
        xac=ma.masked_array(xac,mask=mask)

    
    ssha_extra=ssha
   
    
    # ----- Remove rows completly masked out 
    
    ssha=ma.masked_array(ssha,mask=np.isnan(ssha))
    ssha_plot4 = ssha
    mask_loc = np.where((lon >1) & (lon < 14) &
                        (lat > 56) & (lat < 69), 0, 1)
    mask = mask|mask_loc
    ssha=ma.masked_array(ssha,mask=mask)

    # Plot things 
    lon_plot=lon
    lat_plot=lat
        
    
    maskbool=ma.getmask(ssha)
    ssha = ssha[~np.all(maskbool == True, axis=1)]
    lon = lon[~np.all(maskbool == True, axis=1)]
    lat = lat[~np.all(maskbool == True, axis=1)]
    time = time[~np.all(maskbool == True, axis=1)]
    xac = xac[~np.all(maskbool == True, axis=1)]
    
    
    if np.all(maskbool):
        continue
    

    # select only valid (unmasked) values
    lon_valid = lon.compressed()
    lat_valid = lat.compressed()
    ssha_valid = ssha.compressed()
    
    # --------------------------
    # calc geostrofic currents 
    # --------------------------
    
    # get lon and lat in rad
    

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    dlat_y, dlat_x = np.gradient(lat_rad)
    dlon_y, dlon_x = np.gradient(lon_rad)
    
    heading = np.arctan2(dlon_y, dlat_y)
    
    heading_mean = np.nanmean(np.rad2deg(heading))
    
    orbit_angle = heading   # already same shape as ssha

    # calc gradients in ADT in along and cross track
    
    grad_along, grad_cross = np.gradient(ssha)
    
    grad_cross /= 2.0 # Convert unit to m / km
    
    grad_along /= 2.0 # Convert unit to m / km
    
   
    # calc u and v, dependend on if pass ascending or descending. 
    
    if (heading_mean > 90) or (heading_mean < -90):# descending
    
        correction_angle = orbit_angle
        
        
    
        # Along track
        grad_along_x = grad_along*np.sin(orbit_angle)
    
        grad_along_y = grad_along*np.cos(orbit_angle)
    
     
    
        # Cross track
    
        grad_cross_x = grad_cross*np.cos(orbit_angle)
    
        grad_cross_y = -grad_cross*np.sin(orbit_angle)
       
    
    else: # ascending 
    
        # Along track
    
        grad_along_x = grad_along*np.sin(orbit_angle)
    
        grad_along_y = grad_along*np.cos(orbit_angle)
    
     
    
        # Cross track
    
        grad_cross_x = grad_cross*np.cos(orbit_angle)
    
        grad_cross_y = -grad_cross*np.sin(orbit_angle)
        
    
    # calc lon/lat gradients
    
    grad_x = grad_cross_x + grad_along_x
    
    grad_y = grad_cross_y + grad_along_y
    
    
    # ClacSurface Geostrophic current
    g = 9.82
    Omega = 7.292e-5 # s^-1
    f = 2*Omega * np.sin(np.deg2rad(lat))
    
     
    u = -g/f * (grad_y/1000) # convert gradient to m/m
    v = g/f * (grad_x/1000) # convert gradient to m/m
    speed = np.sqrt(u**2 + v**2)
    
    # for plot
    u_valid = u.compressed()
    
    
        
    if plot==1:
        sc = ax1.scatter(
            lon_valid, lat_valid, c=u_valid,
            cmap='RdBu_r',
            vmin=-0.25,
            vmax=0.25,
            s=0.5,
            transform=ccrs.PlateCarree()
        )
        
        
    elif plot==2:
        

            lon_min, lon_max = 1, 14
            lat_min, lat_max = 56, 69
            
            proj = ccrs.Orthographic(
                central_longitude=(lon_min+lon_max)/2,
                central_latitude=(lat_min+lat_max)/2
            )
            
            plt.ion()
            fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=300, subplot_kw={'projection': proj})
            ax2.set_extent([lon_min, lon_max, lat_min, lat_max])
            
            ax2.coastlines(resolution='50m')
            ax2.add_feature(cfeature.LAND)
            ax2.gridlines(draw_labels=True, dms=True)
            
            
            sc2 = ax2.scatter(
                lon, lat, c=np.ma.filled(speed, np.nan),
                cmap='RdBu_r',
                vmin=-1,
                vmax=1,
                s=0.5,
                transform=ccrs.PlateCarree()
            )
            
            plt.colorbar(sc2,ax=ax2, shrink=0.7, label='SSHA [m]')
            plt.title(f'{date2}', fontsize=14)
            plt.show()
    
    # save for after loop 
    
    ssha_all=ma.hstack([ssha_all,ssha.T])
    lon_all=ma.hstack([lon_all,lon.T])
    lat_all=ma.hstack([lat_all,lat.T])
    time_all=ma.append(time_all,time)
    xac_all=ma.hstack([xac_all,xac.T/1000])
    ugos_all=ma.hstack([ugos_all,u.T])
    vgos_all=ma.hstack([vgos_all,v.T])
    
 

if plot==1:
    # After the loop, add colorbar and title
    plt.colorbar(sc,ax=ax1, shrink=0.7, label='SSHA [m]')
    plt.title('All L3 passes overlapping', fontsize=14)
    plt.show()
  
# fill all empty values with  NaN 
  
lon_all=lon_all.filled(fill_value=np.nan)
lat_all=lat_all.filled(fill_value=np.nan)
ssha_all=ssha_all.filled(fill_value=np.nan)
xac_all=xac_all.filled(fill_value=np.nan)
ugos_all=ugos_all.filled(fill_value=np.nan)
vgos_all=vgos_all.filled(fill_value=np.nan)

# make it so not maksed array anymore 
mask=np.where(np.isnan(lon_all),1,0)

lon_all=ma.masked_array(lon_all,mask=mask)
lat_all=ma.masked_array(lat_all,mask=mask)
ssha_all=ma.masked_array(ssha_all,mask=mask)
xac_all=ma.masked_array(xac_all,mask=mask)
ugos_all=ma.masked_array(ugos_all,mask=mask)
vgos_all=ma.masked_array(vgos_all,mask=mask)

#%% Save data in file 
ncfile = nc.Dataset('./data/swot/august_geoid_nativ_NK_MDT2025_filtered_.nc',mode='w',format='NETCDF4_CLASSIC') 


nc_dim = ncfile.createDimension('nC',69)
time_dim = ncfile.createDimension('time',np.size(lon_all,1))

lon = ncfile.createVariable('lon', np.float32, ('nC','time'))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
lat = ncfile.createVariable('lat', np.float32, ('nC','time'))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'seconds since 2000-01-01 00:00:00.0'
time.long_name = 'time in UTC'
ssha = ncfile.createVariable('ssha', np.float32, ('nC','time'))
ssha.units = 'm'
ssha.long_name = 'sea surface height anomaly'
xac = ncfile.createVariable('x_ac', np.float32, ('nC','time'))
xac.units = 'km'
xac.long_name = 'Across track distance from nadir'

ugos = ncfile.createVariable('ugos', np.float32, ('nC','time'))
ugos.units = 'm/s'
ugos.long_name = 'Dervied Geostrophic currect from SWOTs ssh'

vgos = ncfile.createVariable('vgos', np.float32, ('nC','time'))
vgos.units = 'm/s'
vgos.long_name = 'Dervied Geostrophic currect from SWOTs ssh'

lon[:,:] = lon_all
lat[:,:] = lat_all
time[:] = time_all.filled(np.nan)
time[:] = np.ma.filled(time_all, np.nan)
ssha[:,:] = ssha_all
ugos[:,:] = ugos_all
vgos[:,:] = vgos_all
xac[:,:] = xac_all

ncfile.close()
