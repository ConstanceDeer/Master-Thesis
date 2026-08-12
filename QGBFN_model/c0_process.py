# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:05:46 2026

@author: const
"""

import netCDF4 as nc
import numpy as np
load_file = nc.Dataset('../auxx/aux_first_baroclinic_speed.nc', "r")
datakeys = load_file.variables.keys()
out = {}
for k in datakeys:
    out[k] = load_file.variables[k][:]
    
c1=out['c1'].filled(np.nan)
longitude=out['lon']
latitude=out['lat']
longitude = np.where(longitude>180,longitude-360,longitude)
longitude=np.delete(longitude,0)
c1=np.delete(c1,0,axis=0)
longitude = np.roll(longitude,int(720/2),axis=0)
c1 = np.roll(c1,int(720/2),axis=0)

lon_min, lon_max = 2, 14
lat_min, lat_max = 57, 69
lon_idx = np.where((longitude >= lon_min) & (longitude <= lon_max))[0]
lat_idx = np.where((latitude >= lat_min) & (latitude <= lat_max))[0]


longitude = longitude[lon_idx]
latitude  = latitude[lat_idx]
c1 = c1[np.ix_(lon_idx, lat_idx)]  # slice 2D array

#long, latg = np.meshgrid(latitude,longitude)
#get only the valid values
#latg1 = latg[~np.isnan(c1)]
#long1 = long[~np.isnan(c1)]
#newc1 = c1[~np.isnan(c1)]

#c1 = scipy.interpolate.griddata((latg1, long1), newc1.ravel(),(latg, long),method='nearest')

#%%


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature

# Suppose you already have:
# c1_sub: 2D array of your data
# longitude_sub: 1D array of longitudes
# latitude_sub: 1D array of latitudes

# Create a meshgrid for pcolormesh
lon2d, lat2d = np.meshgrid(longitude, latitude, indexing='ij')

# Create figure
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj})
ax.set_extent([longitude.min(), longitude.max(),
               latitude.min(), latitude.max()],
              crs=proj)

# Plot your data
pcm = ax.pcolormesh(lon2d, lat2d, c1, cmap='viridis', shading='auto', transform=proj)

# Optional: add coastlines
ax.coastlines(resolution='10m')

# Add colorbar
plt.colorbar(pcm, ax=ax, label='c1 value')

plt.show()

mean_c1 = np.nanmean(c1)
print(mean_c1)

#%% L_R

# Earth's rotation rate
Omega = 7.2921159e-5  # rad/s

# Coriolis parameter (depends only on latitude)
f = 2 * Omega * np.sin(np.deg2rad(latitude))

# Expand to the same shape as c1
f2d = np.tile(f, (len(longitude), 1))

# Rossby deformation radius (m)
Ld = c1 / np.abs(f2d)

# Convert to km
Ld_km = Ld / 1000.0

print(f"Mean Rossby deformation radius = {np.nanmean(Ld_km):.1f} km")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

lon2d, lat2d = np.meshgrid(longitude, latitude, indexing='ij')

proj = ccrs.PlateCarree()

fig, ax = plt.subplots(figsize=(10,8),
                       subplot_kw={'projection': proj})

ax.set_extent([2, 13.9, 57.5, 68.5], ccrs.PlateCarree())



gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
gl.top_labels = False
gl.right_labels = False



pcm = ax.pcolormesh(
    lon2d,
    lat2d,
    Ld_km,
    cmap='turbo',
    shading='auto',
    transform=proj
)

ax.coastlines(resolution='10m')
ax.add_feature(cfeature.LAND, facecolor="black")
ax.coastlines("10m")

ax.set_aspect(1/np.cos(np.deg2rad(62)))
fig.colorbar(
    pcm,
    ax=ax,
    orientation="vertical",
    pad=0.01,
    aspect=40,
    label="First baroclinic Rossby deformation radius [km]"
)

plt.show()

#%%

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Meshgrid
lon2d, lat2d = np.meshgrid(longitude, latitude, indexing="ij")

proj = ccrs.PlateCarree()

fig, axs = plt.subplots(
    1, 2,
    figsize=(9, 6),
    subplot_kw={"projection": proj},
    constrained_layout=True, dpi=300
)

# ---------------- c1 ----------------
ax = axs[0]

ax.set_extent([2, 13.9, 57.5, 68.5], crs=proj)

gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
gl.top_labels = False
gl.right_labels = False

pcm1 = ax.pcolormesh(
    lon2d,
    lat2d,
    c1,
    cmap="viridis",
    shading="auto",
    transform=proj
)

ax.add_feature(cfeature.LAND, facecolor="black")
ax.coastlines("10m")
ax.set_aspect(1 / np.cos(np.deg2rad(62)))

ax.text(
    0.02,0.99,"(a) First baroclinic wave speed",
    transform=ax.transAxes,
    fontsize=8,
    color="white",
    fontweight="bold",
    va="top",
    bbox=dict(
        facecolor="black",
        edgecolor="none"
    )
)

cb1 = fig.colorbar(
    pcm1,
    ax=ax,
    orientation="vertical",
    pad=0.01,
    aspect=40
)
cb1.set_label(r"$c_0$ [m s$^{-1}$]")


# ---------------- Rossby radius ----------------
ax = axs[1]

ax.set_extent([2, 13.9, 57.5, 68.5], crs=proj)

gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
gl.top_labels = False
gl.right_labels = False

pcm2 = ax.pcolormesh(
    lon2d,
    lat2d,
    Ld_km,
    cmap="turbo",
    shading="auto",
    transform=proj,
    vmin=1,
    vmax=10,
)

ax.add_feature(cfeature.LAND, facecolor="black")
ax.coastlines("10m")
ax.set_aspect(1 / np.cos(np.deg2rad(62)))

ax.text(
    0.02,0.99,"(b) Rossby deformation radius",
    transform=ax.transAxes,
    fontsize=8,
    color="white",
    fontweight="bold",
    va="top",
    bbox=dict(
        facecolor="black",
        edgecolor="none"
    )
)

cb2 = fig.colorbar(
    pcm2,
    ax=ax,
    ticks=np.arange(0, 11, 1),   # 1-km tick spacing
    pad=0.01,
    aspect=40,
)

cb2.set_label(r"$L_R$ [km]")


plt.show()

print(f"Mean c1 = {np.nanmean(c1):.2f} m/s")
print(f"Mean L_R = {np.nanmean(Ld_km):.2f} km")


#%% c1_cut2.nc
try: ncfile.close()  # just to be safe, make sure dataset is not already open.
except: pass
ncfile = nc.Dataset('./data/c1.nc',mode='w',format='NETCDF4_CLASSIC') 

lon_dim = ncfile.createDimension('lon',np.size(longitude))
lat_dim = ncfile.createDimension('lat',np.size(latitude))

lon = ncfile.createVariable('lon', np.float32, ('lon'))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
lat = ncfile.createVariable('lat', np.float32, ('lat'))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
c = ncfile.createVariable('c1', np.float32, ('lon','lat'))
c.long_name = 'first baroclinic phase speed'

lon[:] = longitude
lat[:] = latitude
c[:,:] = c1

ncfile.close()