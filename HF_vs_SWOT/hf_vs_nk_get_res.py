# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:12:01 2026

@author: const
"""

#%% packagaes
from ast import Global

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
#%% Geostrofic functions 


#% Geostrofic currents
def geostrophic_currents_stero(lat, lon, zeta, equator_mask=1.0):
    """
    Compute geostrophic currents from SSHA (sea surface height anomaly).

    Parameters
    ----------
    lat : 2D array [lat, lon] in degrees
    lon : 2D array [lat, lon] in degrees
    zeta : 2D array [lat, lon] in meters
    equator_mask : float, optional
        Mask latitude range around equator where |lat| < equator_mask

    Returns
    -------
    u_g, v_g : 2D arrays [lat, lon] in m/s
        Zonal and meridional geostrophic velocities
    """
    import numpy as np
    from pyproj import Proj
    g = 9.81
    Omega = 7.2921e-5
   

    # Convert to radians
    lat_rad = np.deg2rad(lat)

    # --- 3. Coriolis parameter ---
    f = 2 * Omega * np.sin(lat_rad)

    # --- 4. Compute grid spacing (meters) ---

    # projection from the NetCDF metadata
    proj = Proj(
        "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 "
        "+x_0=3192800 +y_0=1784000 "
        "+a=6378137 +b=6356752.3142 +units=m +no_defs"
    )

    # convert lon/lat -> x/y
    x, y = proj(lon, lat)

    # grid spacing
    dx = np.gradient(x, axis=1)
    dy = np.gradient(y, axis=0)

    dzdx = np.gradient(zeta, axis=1) / dx
    dzdy = np.gradient(zeta, axis=0) / dy

    ug = -(g/f) * dzdy
    vg =  (g/f) * dzdx

    # rotat to north, east directions 
    gamma = np.deg2rad(lon - 70.0) # Polar stereographic convergence angle 70 deg lon set to true north in te

    u_east = (
        ug * np.cos(gamma)
        + vg * np.sin(gamma)
    )

    v_north = (
        -ug * np.sin(gamma)
        + vg * np.cos(gamma)
    )
    
    mag_g=np.sqrt(u_east**2+v_north**2)

    return u_east, v_north, mag_g

def mean_filter(image, ksize=3):
    pad = ksize // 2
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+ksize, j:j+ksize]
            output[i, j] = np.nanmean(window)
    return output


#%% Work for one timestep - NorKyst nativ velocity field and hf radars nativ velocity

date=datetime(2025, 7, 5, 0)
date_string= date.strftime("%Y%m%d")
date_only = pd.Timestamp(date).normalize()
hour= date.strftime("%H")
hour = int(hour)

# ------- LOAD NORKYST ----------- #

filename = f"./norkyst800/july/norkyst_{date_string}.nc"

# Open the dataset
ds = xr.open_dataset(filename)

# Access variables
zeta0 = ds["zeta"].isel(time=hour)
zeta_mean=ds['zeta'].groupby("time.date").mean("time").squeeze()
u = ds["u_eastward"].isel(time=hour)
v = ds["v_northward"].isel(time=hour)
lat = ds["lat"]
lon = ds["lon"]


#zeta_mean=mean_filter(zeta_mean,ksize=1)
ug, vg, _=geostrophic_currents_stero(lat, lon, zeta_mean, equator_mask=1.0)

# ----- LOAD HF ------ #

# ----------------------------
# HF radar
# ----------------------------
ds1 = xr.open_dataset('./hf/SLAT/SLAT_july2025_rgne.nc')

# SLAT
lat0 = np.deg2rad(59.9086667)
lon0 = np.deg2rad(5.0669167)

# KRAK
#lat0 = np.deg2rad(62.0329833)
#lon0 = np.deg2rad(4.9878833)

#FEDJ
#lat0 = np.deg2rad(60.7759500)
#lon0 = np.deg2rad(4.6946500)

hf_time = pd.to_datetime(ds1['time'].values)
idx_hf=np.where(hf_time==date)

hf_vel = ds1['velocity'][idx_hf]
hf_lon = ds1['lon'].values[idx_hf]
hf_lat = ds1['lat'].values[idx_hf]
err_t=ds1['err_t'].values    # hf time std
err_s=ds1['err_s'].values    # hf sapce std

err=np.sqrt(ds1['err_t']**2+ds1['err_s']**2)

# -- if daily mean needed --- #
daily_mean = ds1['velocity'].groupby("time.date").mean("time")
valid = ds1['velocity'].groupby("time.date").count("time") == 24
hf_vel_mean = daily_mean.where(valid)
hf_dates = pd.to_datetime(hf_time).normalize().unique()
idx_hf_2 = np.where(hf_dates == date_only)[0]
hf_vel_mean=hf_vel_mean[idx_hf_2,:].squeeze()
print(hf_dates[idx_hf_2])


# -- calc range velocities of norkyst --- #

latr = np.deg2rad(lat)
lonr = np.deg2rad(lon)

dlon = lonr - lon0

theta = np.arctan2(
        np.sin(dlon) * np.cos(latr),
        np.cos(lat0) * np.sin(latr) -
        np.sin(lat0) * np.cos(latr) * np.cos(dlon)
)

norkyst_bear = (np.rad2deg(theta) + 360) % 360
theta = np.deg2rad(norkyst_bear)

NK_radial_g = ug * np.sin(theta) + vg * np.cos(theta)
NK_radial_g=NK_radial_g.values
NK_radial_g[np.abs(NK_radial_g) > 10] = np.nan
NK_radial_g=mean_filter(NK_radial_g, ksize=5)


NK_radial = u * np.sin(theta) + v * np.cos(theta)
NK_radial=NK_radial.values
NK_radial[np.abs(NK_radial) > 10] = np.nan
NK_radial=mean_filter(NK_radial,ksize=1)

# ---- Regrid Norkyst to HF-radar coordinats --- #
from scipy.interpolate import griddata
NK_points = np.column_stack((lon.values.ravel(), lat.values.ravel()))
hf_points = np.column_stack((hf_lon.ravel(), hf_lat.ravel()))
NK_vals_g = NK_radial_g.ravel()
NK_vals = NK_radial.ravel()
NK_radial_hf=griddata(NK_points,NK_vals,hf_points,method='linear').reshape(hf_lon.shape)
NK_radial_hf_g=griddata(NK_points,NK_vals_g,hf_points,method='linear').reshape(hf_lon.shape)

diff=NK_radial_hf-hf_vel.values
diff_g=NK_radial_hf_g-hf_vel_mean.values



#%%
# ---- Plot the results to check ---- #
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

lon_min, lon_max = 1.5, 6
lat_min, lat_max = 58, 62

proj = ccrs.Orthographic(
    central_longitude=(lon_min + lon_max) / 2,
    central_latitude=(lat_min + lat_max) / 2
)

fig, ax = plt.subplots(figsize=(9, 12), subplot_kw={'projection': proj})

ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.gridlines(draw_labels=True, dms=True)
c=ax.pcolormesh(
    lon[::1, ::1],
    lat[::1, ::1],
    NK_radial[::1, ::1],transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=-0.5,vmax=0.5
)
ax.scatter(hf_points[:,0],hf_points[:,1],c=hf_vel.values.flatten(),transform=ccrs.PlateCarree(),s=10,cmap='RdBu_r',vmin=-0.5,vmax=0.5)
ax.set_title('Raidal velocities of NorKyst800 and HF-radar SLAT')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='black')
cbar = plt.colorbar(c,ax=ax, shrink=0.5, location='bottom', pad=0.05)
cbar.set_label("Radial velocity [m/s]")

plt.show()

fig, ax = plt.subplots(figsize=(9, 12), subplot_kw={'projection': proj})

ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.gridlines(draw_labels=True, dms=True)
c=ax.pcolormesh(
    lon[::1, ::1],
    lat[::1, ::1],
    NK_radial_g[::1, ::1],transform=ccrs.PlateCarree(),cmap='RdBu_r',vmin=-0.5,vmax=0.5
)
ax.scatter(hf_points[:,0],hf_points[:,1],c=hf_vel_mean.values.flatten(),transform=ccrs.PlateCarree(),s=10,cmap='RdBu_r',vmin=-0.5,vmax=0.5)
ax.set_title('Geostrofic Raidal velocities of NorKyst800 and HF-radar SLAT')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='black')
cbar = plt.colorbar(c,ax=ax, shrink=0.5, location='bottom', pad=0.05)
cbar.set_label("Radial velocity [m/s]")

plt.show()


fig, ax = plt.subplots(figsize=(9, 12), subplot_kw={'projection': proj})

ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='black')
ax.gridlines(draw_labels=True, dms=True)
c=ax.scatter(hf_points[:,0],hf_points[:,1],c=NK_radial_hf.flatten()-hf_vel.values.flatten(),transform=ccrs.PlateCarree(),s=10,cmap='RdBu_r',vmin=-0.5,vmax=0.5)
ax.set_title('Raidal velocities: NorKyst800 - HF-radar SLAT')
cbar = plt.colorbar(c,ax=ax, shrink=0.5, location='bottom', pad=0.05)
cbar.set_label("Residuals [m/s]")

plt.show()

fig, ax = plt.subplots(figsize=(9, 12), subplot_kw={'projection': proj})

ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='black')
ax.gridlines(draw_labels=True, dms=True)
c=ax.scatter(hf_points[:,0],hf_points[:,1],c=NK_radial_hf_g.flatten()-hf_vel_mean.values.flatten(),transform=ccrs.PlateCarree(),s=10,cmap='RdBu_r',vmin=-0.5,vmax=0.5)
ax.set_title('Geostrofic Raidal velocities: NorKyst800 - HF-radar SLAT')
cbar = plt.colorbar(c,ax=ax, shrink=0.5, location='bottom', pad=0.05)
cbar.set_label("Residuals [m/s]")

plt.show()
#%%
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

print(f'err: {np.nanmean(err)}')

# ---- Calc stats ------ #

u1=NK_radial_hf_g[0,:,:]
u2=hf_vel_mean.values
res=u1-u2

mask_valid = np.isfinite(u1) & np.isfinite(u2)
corr = np.corrcoef(u1[mask_valid], u2[mask_valid])[0, 1]
rms = np.sqrt(np.mean(res[mask_valid]**2))
agree = np.mean(np.sign(u1[mask_valid]) == np.sign(u2[mask_valid]))


print('---- Geostrophic ----')
print(corr)
print(rms)
print(agree)

# gaussian dist analysisn 

data = res[~np.isnan(res)].reshape(-1, 1)

gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data)


x = np.linspace(data.min(), data.max(), 500).reshape(-1, 1)

logprob = gmm.score_samples(x)
pdf = np.exp(logprob)
weights = gmm.weights_
means = gmm.means_.flatten()
covs = gmm.covariances_.flatten()
stds = np.sqrt(covs)

# plot 
plt.figure(figsize=(10,5))
plt.hist(data, bins=100, density=True, alpha=0.6,
         label='Residuals')

plt.plot(x, pdf, 'k', lw=2, label='GMM (2 components)')

for i in range(2):
    component = weights[i] * (1 / (np.sqrt(2*np.pi)*stds[i])) * \
                np.exp(-0.5 * ((x.flatten() - means[i]) / stds[i])**2)

    plt.plot(x, component, linestyle='--',
             label=f'Comp {i+1}: μ={means[i]:.3f}, σ={stds[i]:.3f}')

plt.xlabel('Difference (SWOT - HF radar) (m/s)')
plt.ylabel('Density')
plt.title('3-component Gaussian Mixture Fit')
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend()
plt.show()


print('---- Total surface velocity ----')


u1=NK_radial_hf[0,:,:]
u2=hf_vel.values[0,:,:]
res=u1-u2

mask_valid = np.isfinite(u1) & np.isfinite(u2)
corr = np.corrcoef(u1[mask_valid], u2[mask_valid])[0, 1]
rms = np.sqrt(np.mean(res[mask_valid]**2))
agree = np.mean(np.sign(u1[mask_valid]) == np.sign(u2[mask_valid]))

print(corr)
print(rms)
print(agree)


# gaussian dist analysisn 

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
plt.figure(figsize=(10,5))
plt.hist(data, bins=100, density=True, alpha=0.6,
         label='Residuals')

plt.plot(x, pdf, 'k', lw=2, label='GMM (1 components)')

for i in range(1):
    component = weights[i] * (1 / (np.sqrt(2*np.pi)*stds[i])) * \
                np.exp(-0.5 * ((x.flatten() - means[i]) / stds[i])**2)

    plt.plot(x, component, linestyle='--',
             label=f'Comp {i+1}: μ={means[i]:.3f}, σ={stds[i]:.3f}')

plt.xlabel('Difference (SWOT - HF radar) (m/s)')
plt.ylabel('Density')
plt.title('1-component Gaussian Mixture Fit')
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend()
plt.show()



#%%  Loop over two weeks 
from netCDF4 import  Dataset,num2date

# ----- LOAD HF ------ #

# ----------------------------
# HF radar
# ----------------------------
ds1 = xr.open_dataset('./hf/KRAK/KRAK_junejulyaugust2025_rgne.nc')

# SLAT
lat0 = np.deg2rad(59.9086667)
lon0 = np.deg2rad(5.0669167)

# KRAK
#lat0 = np.deg2rad(62.0329833)
#lon0 = np.deg2rad(4.9878833)

#FEDJ
lat0 = np.deg2rad(60.7759500)
lon0 = np.deg2rad(4.6946500)

hf_time = pd.to_datetime(ds1['time'].values)

HF_vel = ds1['velocity'].values
hf_lon = ds1['lon'].values[0]
hf_lat = ds1['lat'].values[0]
err_t=ds1['err_t'].values    # hf time std
err_s=ds1['err_s'].values    # hf sapce std

err=np.sqrt(ds1['err_t']**2+ds1['err_s']**2)

# -- if daily mean needed --- #
daily_mean = ds1['velocity'].groupby("time.date").mean("time")
valid = ds1['velocity'].groupby("time.date").count("time") == 24
hf_vel_mean = daily_mean.where(valid)
hf_dates = pd.to_datetime(hf_time).normalize().unique()
HF_vel_mean=hf_vel_mean.squeeze()


diff_total=[]
diff_geo=[]
time=[]
nk_tot=[]
nk_geo=[]
hf_hr=[]
hf_day=[]

start_date = hf_dates[0]  # or some other logic to define the start date
for t in hf_dates:
    
    date=t
    date_string= date.strftime("%Y%m%d")
    date_only = pd.Timestamp(date).normalize()
    hour= date.strftime("%H")
    hour = int(hour)
   
    # find hf data from that date + hr
    idx_hf=np.where(hf_time==date)
    idx_hf_2 = np.where(hf_dates == date_only)[0]

    hf_vel_mean=HF_vel_mean[idx_hf_2,:][0,:,:]
    hf_vel=HF_vel[idx_hf][0,:,:]

   

    print(date_string)

    # ------- LOAD NORKYST ----------- #

    #filename = f"./norkyst800/july/norkyst_{date_string}.nc"

    #try:
    #    ds = xr.open_dataset(filename)
    #except Exception as e:
    #    print(f"Could not open {filename}: {e}")
    #    continue

    # Access variables
    #zeta0 = ds["zeta"].isel(time=hour)
    #zeta_mean=ds['zeta'].groupby("time.date").mean("time").squeeze()
    #u = ds["u_eastward"].isel(time=hour)
    #v = ds["v_northward"].isel(time=hour)
    #lat = ds["lat"]
    #lon = ds["lon"]

    year=date.strftime("%Y")
    month=date.strftime("%m")
    d=date.strftime("%d")
    hr=date.strftime("%H")
    hr= int(hr)
    url = f"https://thredds.met.no/thredds/dodsC/romshindcast/norkyst_v3/zdepth/{year}/{month}/norkyst800-{date_string}.nc"
       
    try:
        nc = Dataset(url)
    except Exception as e:
        print(f"Could not open {url}: {e}")
        continue
        
    if t==start_date:
           
            lat_min, lat_max = 58, 64
            lon_min, lon_max = 1.5, 6

                
            lat_nk = nc.variables["lat"][:]
            lon_nk = nc.variables["lon"][:]
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
            
            lon=lon_sub
            lat=lat_sub
            
            # for interp
            NK_points = np.column_stack((lon_nk.ravel(), lat_nk.ravel()))
        

    
        
    #zeta0 = nc.variables["zeta"][:, y0:y1, x0:x1]
    zeta_mean=nc.variables["zeta"][:, y0:y1, x0:x1].mean(axis=0)
    u=nc.variables["u_eastward"][:,0, y0:y1, x0:x1].mean(axis=0)
    v=nc.variables["v_northward"][:,0, y0:y1, x0:x1].mean(axis=0)



    #generate geostrophic currents
    ug, vg, _=geostrophic_currents_stero(lat, lon, zeta_mean, equator_mask=1.0)

    # -- calc range velocities of norkyst --- #

    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)

    dlon = lonr - lon0

    theta = np.arctan2(
            np.sin(dlon) * np.cos(latr),
            np.cos(lat0) * np.sin(latr) -
            np.sin(lat0) * np.cos(latr) * np.cos(dlon)
    )

    norkyst_bear = (np.rad2deg(theta) + 360) % 360
    theta = np.deg2rad(norkyst_bear)


    # geostrophic
    NK_radial_g = ug * np.sin(theta) + vg * np.cos(theta)
    NK_radial_g=NK_radial_g
    NK_radial_g[np.abs(NK_radial_g) > 10] = np.nan
    NK_radial_g=mean_filter(NK_radial_g, ksize=1)

    # total velocities 
    NK_radial = u * np.sin(theta) + v * np.cos(theta)
    NK_radial=NK_radial
    NK_radial[np.abs(NK_radial) > 10] = np.nan

    # ---- Regrid Norkyst to HF-radar coordinats --- #
    from scipy.interpolate import griddata
    NK_points = np.column_stack((lon.ravel(), lat.ravel()))
    hf_points = np.column_stack((hf_lon.ravel(), hf_lat.ravel()))
    NK_vals_g = NK_radial_g.ravel()
    NK_vals = NK_radial.ravel()
    NK_radial_hf=griddata(NK_points,NK_vals,hf_points,method='linear').reshape(hf_lon.shape)
    NK_radial_hf_g=griddata(NK_points,NK_vals_g,hf_points,method='linear').reshape(hf_lon.shape)

    diff=NK_radial_hf-hf_vel
    diff_g=NK_radial_hf_g-hf_vel_mean

    diff_total.append(diff)
    diff_geo.append(diff_g)
    nk_tot.append(NK_radial_hf)
    nk_geo.append(NK_radial_hf_g)
    hf_hr.append(hf_vel)
    hf_day.append(hf_vel_mean)
    time.append(date)
#%% SAVE
import numpy as np

np.savez(
    "./fedj.npz",
    time=np.array(time),
    diff_total=np.array(diff_total),
    diff_geo=np.array(diff_geo),
    nk_tot=np.array(nk_tot),
    nk_geo=np.array(nk_geo),
    hf_hr=np.array(hf_hr),
    hf_day=np.array(hf_day),
)

#%% Global statisitc Analysis 

total_velocity = np.array(diff_total)
geo_velocity = np.array(diff_g)
hf=np.array(hf_hr) # not meaned hf data
hf_mean=np.array(hf_day) # daily mean hf data
nk_t=np.array(nk_tot) # total velocity swot on hf grid
nk_g=np.array(nk_geo) # geostropgic velocity from nk on hf grid 
t = np.array(time)


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
