# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:05:05 2026

@author: const
"""
#%% load
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fil_name=['EGG2015','NKG2015','XGM2019e']

#fil_name=['Norkyst MDT','DTUUH22','CNES-CLS22']

fil_name=['Filtered','Unfiltered']
# ----------------------------
# Load distance to coast
# ----------------------------
ds = xr.open_dataset(
    r"C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/dist2coast/dist2coast.nc"
)

dist2coast = ds["dist2coast"]
lon_d2c = ds["lon"].values
lat_d2c = ds["lat"].values

# ----------------------------
# Get all files
# ----------------------------
files = sorted(glob.glob("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/Github/norkyst_vs_swot/baseline/*.npz"))
#files = sorted(glob.glob("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/Github/norkyst_vs_swot/NKDA/*.npz"))
# ----------------------------
# Storage across files
# ----------------------------
u_sel_n_all = []
v_sel_n_all = []
u_sel_s_all = []
v_sel_s_all = []

lon_sel_all = []
lat_sel_all = []
dist_sel_all = []

adt_sel_n_all = []
adt_sel_s_all = []

grad_nx_all_files = []
grad_ny_all_files = []
grad_sx_all_files = []
grad_sy_all_files = []

vor_swath_s=[]
vor_swath_n=[]

# ----------------------------
# LOOP OVER FILES
# ----------------------------
for file_idx, file in enumerate(files):

    data = np.load(file, allow_pickle=True)
    
    print(file)

    swot = data["swot"]
    nk_int = data["nk_int"]

    lon = data["lon"]
    lat = data["lat"]

    lon_nk = data["lon_nk"]
    lat_nk = data["lat_nk"]

    u_s = data['swot_ugos']
    v_s = data['swot_vgos']

    u_n = data['nk_ugos_int']
    v_n = data['nk_vgos_int']

    grad_x_nk = []
    grad_y_nk = []
    grad_x_s = []
    grad_y_s = []
   
    

    

    # ----------------------------
    # Compute gradients per track
    # ----------------------------
    for i in range(len(lon)):

        LON = np.asarray(lon[i])
        LAT = np.asarray(lat[i])
        NK = nk_int[i]
        SWOT = swot[i]

        lat_rad = np.deg2rad(LAT)
        lon_rad = np.deg2rad(LON)

        dlat_y, dlat_x = np.gradient(lat_rad)
        dlon_y, dlon_x = np.gradient(lon_rad)

        orbit_angle = np.arctan2(dlon_y, dlat_y)

        # NK gradients
        grad_along, grad_cross = np.gradient(NK, 2000, 2000)

        grad_along_x = grad_along * np.sin(orbit_angle)
        grad_along_y = grad_along * np.cos(orbit_angle)

        grad_cross_x = grad_cross * np.cos(orbit_angle)
        grad_cross_y = -grad_cross * np.sin(orbit_angle)

        grad_x = grad_cross_x + grad_along_x
        grad_y = grad_cross_y + grad_along_y

        grad_x_nk.append(grad_x)
        grad_y_nk.append(grad_y)

        # SWOT gradients
        grad_along, grad_cross = np.gradient(SWOT, 2000, 2000)

        grad_along_x = grad_along * np.sin(orbit_angle)
        grad_along_y = grad_along * np.cos(orbit_angle)

        grad_cross_x = grad_cross * np.cos(orbit_angle)
        grad_cross_y = -grad_cross * np.sin(orbit_angle)

        grad_x = grad_cross_x + grad_along_x
        grad_y = grad_cross_y + grad_along_y

        grad_x_s.append(grad_x)
        grad_y_s.append(grad_y)

    # ----------------------------
    # Interpolate distance to coast
    # ----------------------------
    lon_2d, lat_2d = np.meshgrid(lon_d2c, lat_d2c)

    d2c_points = np.column_stack((lon_2d.ravel(), lat_2d.ravel()))
    d2c_values = dist2coast.values.ravel()

    dist2coast_s = []

    for i in range(len(lon)):

        LON = np.asarray(lon[i])
        LAT = np.asarray(lat[i])

        shape = LON.shape

        LON_flat = LON.ravel()
        LAT_flat = LAT.ravel()

        mask = ~np.isnan(LON_flat) & ~np.isnan(LAT_flat)

        swot_points = np.column_stack((LON_flat[mask], LAT_flat[mask]))

        dist_i = griddata(
            d2c_points,
            d2c_values,
            swot_points,
            method="linear"
        )

        out = np.full(LON_flat.shape, np.nan)
        out[mask] = dist_i

        dist2coast_s.append(out.reshape(shape))

    # ----------------------------
    # Flatten full file
    # ----------------------------
    lon_all = np.concatenate([np.asarray(x).ravel() for x in lon])
    lat_all = np.concatenate([np.asarray(x).ravel() for x in lat])

    u_all_n = np.concatenate([np.asarray(x).ravel() for x in u_n])
    v_all_n = np.concatenate([np.asarray(x).ravel() for x in v_n])

    u_all_s = np.concatenate([np.asarray(x).ravel() for x in u_s])
    v_all_s = np.concatenate([np.asarray(x).ravel() for x in v_s])

    dist_all = np.concatenate([np.asarray(x).ravel() for x in dist2coast_s])

    adt_all_s = np.concatenate([np.asarray(x).ravel() for x in swot])
    adt_all_n = np.concatenate([np.asarray(x).ravel() for x in nk_int])

    grad_nx_flat = np.concatenate([np.asarray(x).ravel() for x in grad_x_nk])
    grad_ny_flat = np.concatenate([np.asarray(x).ravel() for x in grad_y_nk])
    grad_sx_flat = np.concatenate([np.asarray(x).ravel() for x in grad_x_s])
    grad_sy_flat = np.concatenate([np.asarray(x).ravel() for x in grad_y_s])

    # ----------------------------
    # Mask
    # ----------------------------
    # zone 1: long back
    lon_min, lon_max = 3, 12
    lat_min, lat_max = 62, 69

    # zone 2: west coast
    #lon_min, lon_max = 3, 6
    #lat_min, lat_max = 58, 63
    
    
    lon_min,lon_max = 2,14
    lat_min,lat_max = 57,69



    mask_area = (
        (lon_all >= lon_min) & (lon_all <= lon_max) &
        (lat_all >= lat_min) & (lat_all <= lat_max) &
        (u_all_n != 0) &
        (v_all_n != 0) &
        (np.abs(u_all_n) < 100) &
        (np.abs(v_all_n) < 100) &
        (np.abs(u_all_s) < 100) &
        (np.abs(v_all_s) < 100)
    )

    # ----------------------------
    # STORE PER FILE
    # ----------------------------
    lon_sel_all.append(lon_all[mask_area])
    lat_sel_all.append(lat_all[mask_area])
    dist_sel_all.append(dist_all[mask_area])

    adt_sel_s_all.append(adt_all_s[mask_area])
    adt_sel_n_all.append(adt_all_n[mask_area])

    u_sel_n_all.append(u_all_n[mask_area])
    v_sel_n_all.append(v_all_n[mask_area])

    u_sel_s_all.append(u_all_s[mask_area])
    v_sel_s_all.append(v_all_s[mask_area])
    
    grad_nx_all_files.append(grad_nx_flat[mask_area])
    grad_ny_all_files.append(grad_ny_flat[mask_area])
    grad_sx_all_files.append(grad_sx_flat[mask_area])
    grad_sy_all_files.append(grad_sy_flat[mask_area])
    
 
    
    #---
    #plot
    #---
    n_data=len(lon)
    
    proj = ccrs.Orthographic(
        central_longitude=(lon_min+lon_max)/2,
        central_latitude=(lat_min+lat_max)/2
    )
    
    
    fig, ax = plt.subplots(figsize=(8,10),
                           subplot_kw={'projection': ccrs.PlateCarree()},dpi=200)
    
                                              
    ax.set_aspect(1/np.cos(np.mean(lat[i]*np.pi/180)))
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    
    for i in range(int(n_data/2)-20,int(n_data/2)):
            
        # global 
        #lon_min, lon_max = 1, 14
        #lat_min, lat_max = 56, 69

        
        sc = ax.scatter(
        lon[i],
        lat[i],
        c=np.sqrt(u_s[i]**2+v_s[i]**2).squeeze(),
        s=1,
        cmap='Reds',
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=1
        )
        
    plt.colorbar(sc, ax=ax, shrink=0.8, label='speed [m/s]')
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(8,10),
                           subplot_kw={'projection': ccrs.PlateCarree()},dpi=200)
    
                                              
    ax.set_aspect(1/np.cos(np.mean(lat[i]*np.pi/180)))
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    
    for i in range(int(n_data/2)-20,int(n_data/2)):
            
        # global 
        #lon_min, lon_max = 1, 14
        #lat_min, lat_max = 56, 69

        
        sc = ax.scatter(
        lon[i],
        lat[i],
        c=np.sqrt(u_n[i]**2+v_n[i]**2).squeeze(),
        s=1,
        cmap='Reds',
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=1
        )
        
    plt.colorbar(sc, ax=ax, shrink=0.8, label='speed [m/s]')
    plt.show()


speed_n = []
speed_s = []

for i in range(len(u_sel_n_all)):
    u = u_sel_n_all[i]
    v = v_sel_n_all[i]

    speed_n.append(np.sqrt(u**2 + v**2))

for i in range(len(u_sel_s_all)):
    u = u_sel_s_all[i]
    v = v_sel_s_all[i]

    speed_s.append(np.sqrt(u**2 + v**2))
    
#%% plot inital

n_files = len(lon_sel_all)
for i, (lon_i, lat_i, speed_i) in enumerate(zip(lon_sel_all, lat_sel_all, speed_n)):
        
    # global 
    #lon_min, lon_max = 1, 14
    #lat_min, lat_max = 56, 69
    
    
    # 
    
    proj = ccrs.Orthographic(
        central_longitude=(lon_min+lon_max)/2,
        central_latitude=(lat_min+lat_max)/2
    )
    
    
    fig, ax = plt.subplots(figsize=(8,10),
                           subplot_kw={'projection': ccrs.PlateCarree()},dpi=100)
    
                                              
    ax.set_aspect(1/np.cos(np.mean(lat_i)*np.pi/180))
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    
    sc = ax.scatter(
    lon_i,
    lat_i,
    c=speed_i.squeeze(),
    s=1,
    cmap='Reds',
    transform=ccrs.PlateCarree(),
    vmin=0,
    vmax=1
    )
    
    plt.colorbar(sc, ax=ax, shrink=0.8, label='speed [m/s]')
    plt.show()


#%% dist2coast vs velocities
n_files = len(lon_sel_all)
l = 5  # km bin size

bin_centers_all = []

swot_adt_mean_all = []
norkyst_adt_mean_all = []

swot_grad_mean_all = []
norkyst_grad_mean_all = []

swot_mean_all = []
norkyst_mean_all = []

swot_umean_all = []
norkyst_umean_all = []

swot_vmean_all = []
norkyst_vmean_all = []

counts_all = []

for k in range(n_files):
    print(k)
    print(dist_sel_all[k].shape)
    bins = np.arange(0, np.nanmax(dist_sel_all[k]) + l, l)  # 1 km bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Assign each point to a bin
    bin_idx = np.digitize(dist_sel_all[k], bins)

    # ----------------------------
    # Compute binned means
    # ----------------------------

    # ADT avg 

    swot_adt_mean = np.array([
        np.nanmean(adt_sel_s_all[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    norkyst_adt_mean = np.array([
        np.nanmean(adt_sel_n_all[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    # grad(ADT) avg 

    swot_grad_mean = np.array([
        np.nanmean(grad_sx_all_files[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    norkyst_grad_mean = np.array([
        np.nanmean(grad_nx_all_files[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    # speed
    swot_mean = np.array([
        np.nanmean(speed_s[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    norkyst_mean = np.array([
        np.nanmean(speed_n[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    # u

    swot_umean = np.array([
        np.nanmean(u_sel_s_all[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    norkyst_umean = np.array([
        np.nanmean(u_sel_n_all[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    # v

    swot_vmean = np.array([
        np.nanmean(v_sel_s_all[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    norkyst_vmean = np.array([
        np.nanmean(v_sel_n_all[k][bin_idx == i])
        for i in range(1, len(bins))
    ])

    # Optional: count points per bin (for quality control)
    counts = np.array([
        np.sum(bin_idx == i)
        for i in range(1, len(bins))
    ])

    # Mask bins with too few points
    min_points = 50
    swot_mean[counts < min_points] = np.nan
    norkyst_mean[counts < min_points] = np.nan
    
    bin_centers_all.append(bin_centers)
    
    swot_adt_mean_all.append(swot_adt_mean)
    norkyst_adt_mean_all.append(norkyst_adt_mean)
    
    swot_grad_mean_all.append(swot_grad_mean)
    norkyst_grad_mean_all.append(norkyst_grad_mean)
    
    swot_mean_all.append(swot_mean)
    norkyst_mean_all.append(norkyst_mean)
    
    swot_umean_all.append(swot_umean)
    norkyst_umean_all.append(norkyst_umean)
    
    swot_vmean_all.append(swot_vmean)
    norkyst_vmean_all.append(norkyst_vmean)
    
    counts_all.append(counts)
    


lon0 = lon_min
lat0 = lat_min

#%% stast vs dist2coast

bias_mean_all = []
bias_std_all = []
bias_rms_all = []
mean_ang_diff_deg_all = []
bin_centers_bias_all = []
counts_bias_all = []
mean_sign_all = []
mean_sign_mean = []

for file_idx in range(n_files): 
    bias = speed_s[file_idx] - speed_n[file_idx]
    
    bins = np.arange(0, np.nanmax(dist_sel_all[file_idx]) + l, l)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(dist_sel_all[file_idx], bins)
    
    bias_mean = np.array([
        np.nanmean(bias[bin_idx == b])
        for b in range(1, len(bins))
    ])
    
    bias_std = np.array([
        np.nanstd(bias[bin_idx == b])
        for b in range(1, len(bins))
    ])
    
    bias_rms = np.array([
        np.sqrt(np.nanmean(bias[bin_idx == b]**2))
        for b in range(1, len(bins))
    ])
    
    mean_u_s = np.array([
        np.nanmean(u_sel_s_all[file_idx][bin_idx == b])
        for b in range(1, len(bins))
    ])
    
    mean_v_s = np.array([
        np.nanmean(v_sel_s_all[file_idx][bin_idx == b])
        for b in range(1, len(bins))
    ])
    
    mean_u_n = np.array([
        np.nanmean(u_sel_n_all[file_idx][bin_idx == b])
        for b in range(1, len(bins))
    ])
    
    mean_v_n = np.array([
        np.nanmean(v_sel_n_all[file_idx][bin_idx == b])
        for b in range(1, len(bins))
    ])
    
    theta_s_mean = np.arctan2(mean_v_s, mean_u_s)
    theta_n_mean = np.arctan2(mean_v_n, mean_u_n)
    
    mean_vec_ang_diff = np.arctan2(
        np.sin(theta_s_mean - theta_n_mean),
        np.cos(theta_s_mean - theta_n_mean)
    )
    
    mean_ang_d = np.degrees(np.abs(mean_vec_ang_diff))
    
    # Angle of every individual vector
    theta_s = np.arctan2(u_sel_s_all[file_idx], v_sel_s_all[file_idx])
    theta_n = np.arctan2(u_sel_n_all[file_idx], v_sel_n_all[file_idx])
    
    # Angular difference for every sample (wrapped to [-pi, pi])
    ang_diff = np.arctan2(
        np.sin(theta_s - theta_n),
        np.cos(theta_s - theta_n)
    )
    
    # Mean absolute angular difference in each bin
    mean_ang_diff_deg = np.array([
        np.nanmean(np.degrees(np.abs(ang_diff[bin_idx == b])))
        for b in range(1, len(bins))
    ])
    
    counts = np.array([
        np.sum(bin_idx == b)
        for b in range(1, len(bins))
    ])
    
    agree_count = np.array([
    np.sum(
        (np.sign(u_sel_n_all[file_idx][bin_idx == b]) ==
         np.sign(u_sel_s_all[file_idx][bin_idx == b])) &
        (np.sign(v_sel_n_all[file_idx][bin_idx == b]) ==
         np.sign(v_sel_s_all[file_idx][bin_idx == b]))
    )
        for b in range(1, len(bins))
    ])
    
    agreement_fraction = agree_count / counts
    mean_sign_all.append(agreement_fraction)
    
    mask = counts < 50
    
    bias_mean[mask] = np.nan
    bias_std[mask] = np.nan
    bias_rms[mask] = np.nan
    mean_ang_diff_deg[mask] = np.nan

    # Store
    bin_centers_bias_all.append(bin_centers)
    bias_mean_all.append(bias_mean)
    bias_std_all.append(bias_std)
    bias_rms_all.append(bias_rms)
    mean_ang_diff_deg_all.append(mean_ang_diff_deg)
    counts_bias_all.append(counts)
    mean_sign_mean.append(mean_ang_d)


    
#%% heat maps 
import numpy as np
n_files=len(files)
# ----------------------------
# SETTINGS
# ----------------------------

import numpy as np

# ----------------------------
# SETTINGS
# ----------------------------
bin_size = 5  # km

lon0 = lon_min
lat0 = lat_min

# ----------------------------
# STORAGE ACROSS FILES
# ----------------------------
swot_map_all = []
u_swot_map_all = []
v_swot_map_all = []
norkyst_map_all = []
u_norkyst_map_all = []
v_norkyst_map_all = []

bias_map_all = []
std_map_all = []
rms_map_all = []
sign_map_all = []
angle_map_all = []
dist_map_all=[]

x_centers_all = []
y_centers_all = []

# ============================================================
# LOOP OVER FILES
# ============================================================
for file_idx in range(n_files):

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    lat_sel = lat_sel_all[file_idx]
    lon_sel = lon_sel_all[file_idx]

    u_s = u_sel_s_all[file_idx]
    v_s = v_sel_s_all[file_idx]
    u_n = u_sel_n_all[file_idx]
    v_n = v_sel_n_all[file_idx]
    dist_sel = dist_sel_all[file_idx]


    

    speed_swot = speed_s[file_idx]
    speed_norkyst = speed_n[file_idx]

    bias = speed_swot - speed_norkyst

    # ----------------------------
    # CONVERT TO KM COORDINATES
    # ----------------------------
    lat_ref = np.nanmean(lat_sel)

    x = (lon_sel - lon0) * 111.32 * np.cos(np.radians(lat_ref))  # east (km)
    y = (lat_sel - lat0) * 111.32                                # north (km)

    # ----------------------------
    # BUILD BINS (km grid)
    # ----------------------------
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + bin_size, bin_size)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y) + bin_size, bin_size)

    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])

    # ----------------------------
    # STORAGE FOR THIS FILE
    # ----------------------------
    swot_map = np.full((len(y_centers), len(x_centers)), np.nan)
    u_swot_map =np.full_like(swot_map, np.nan)
    v_swot_map =np.full_like(swot_map, np.nan)
    
    norkyst_map = np.full_like(swot_map, np.nan)
    u_norkyst_map =np.full_like(swot_map, np.nan)
    v_norkyst_map =np.full_like(swot_map, np.nan)

    bias_map = np.full_like(swot_map, np.nan)
    std_map = np.full_like(swot_map, np.nan)
    rms_map = np.full_like(swot_map, np.nan)
    sign_map = np.full_like(swot_map, np.nan)
    angle_map = np.full_like(swot_map, np.nan)
    dist_map = np.full_like(swot_map, np.nan)

    # ============================================================
    # 2D BINNING
    # ============================================================
    for i in range(len(y_bins) - 1):

        y_mask = (y >= y_bins[i]) & (y < y_bins[i + 1])

        for j in range(len(x_bins) - 1):

            m = y_mask & (x >= x_bins[j]) & (x < x_bins[j + 1])

            if np.sum(m) < 3:
                continue

            # ----------------------------
            # SPEED
            # ----------------------------
            swot_map[i, j] = np.nanmean(speed_swot[m])
            norkyst_map[i, j] = np.nanmean(speed_norkyst[m])

            # ----------------------------
            # BIAS STATS
            # ----------------------------
            bias_map[i, j] = np.nanmean(bias[m])
            std_map[i, j] = np.nanstd(bias[m])
            rms_map[i, j] = np.sqrt(np.nanmean(bias[m] ** 2))

            # ----------------------------
            # SIGN AGREEMENT
            # ----------------------------
            cond = (
                (np.sign(u_s[m]) == np.sign(u_n[m])) &
                (np.sign(v_s[m]) == np.sign(v_n[m]))
            )
            sign_map[i, j] = np.mean(cond)

            # ----------------------------
            # ANGLE DIFFERENCE
            # ----------------------------
            us = np.nanmean(u_s[m])
            vs = np.nanmean(v_s[m])
            un = np.nanmean(u_n[m])
            vn = np.nanmean(v_n[m])

            theta_s = np.arctan2(vs, us)
            theta_n = np.arctan2(vn, un)

            dtheta = np.arctan2(
                np.sin(theta_s - theta_n),
                np.cos(theta_s - theta_n)
            )

            angle_map[i, j] = np.degrees(np.abs(dtheta))
            
            dist_map[i, j] = np.nanmean(dist_sel[m])
            
            u_swot_map[i, j]=us
            v_swot_map[i, j]=vs
            
            u_norkyst_map[i, j]=un
            v_norkyst_map[i, j]=vn

    # ----------------------------
    # STORE RESULTS
    # ----------------------------
    swot_map_all.append(swot_map)
    u_swot_map_all.append(u_swot_map)
    v_swot_map_all.append(v_swot_map)
    norkyst_map_all.append(norkyst_map)
    u_norkyst_map_all.append(u_norkyst_map)
    v_norkyst_map_all.append(v_norkyst_map)

    bias_map_all.append(bias_map)
    std_map_all.append(std_map)
    rms_map_all.append(rms_map)
    sign_map_all.append(sign_map)
    angle_map_all.append(angle_map)

    x_centers_all.append(x_centers)
    y_centers_all.append(y_centers)
    
    dist_map_all.append(dist_map)

#%% Heat map mean + direction plot.

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

n_files=len(files)

def add_north_arrow(ax, x=0.92, y=0.1, length=0.15, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )


def format_lat(x, pos):
    return f"{abs(x):.0f}°{'N' if x >= 0 else 'S'}"

def format_lon(x, pos):
    return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"

# ============================================================
# Coordinate conversion
# ============================================================

lat_ref = lat0

km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return lon0 + x / km_per_deg_lon

def km_to_lat(y):
    return lat0 + y / km_per_deg_lat

# --------------------------------------------------
# Mean NK direction
# --------------------------------------------------
nk_direction = (
    np.degrees(
        np.arctan2(
            v_norkyst_map_all[0],
            u_norkyst_map_all[0]
        )
    ) + 360
) % 360

# --------------------------------------------------
# Figure size
# --------------------------------------------------
ncols = n_files + 1

fig, axes = plt.subplots(
    2,
    ncols,
    figsize=(8, 8),
    dpi=300,
    constrained_layout=True
)

# --------------------------------------------------
# Coordinate conversion
# --------------------------------------------------
lat_ref = lat0

km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return lon0 + x / km_per_deg_lon

def km_to_lat(y):
    return lat0 + y / km_per_deg_lat

# ==================================================
# FIRST COLUMN = NORKYST
# ==================================================

x = x_centers_all[0]
y = y_centers_all[0]

im_speed = axes[0,-1].pcolormesh(
    km_to_lon(x),
    km_to_lat(y),
    norkyst_map_all[0],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5
)

axes[0,-1].text(
    0.98, 0.01,
    'Norkyst',
    transform=axes[0,-1].transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

axes[0,-1].set_facecolor("black")

im_dir = axes[1,-1].pcolormesh(
    km_to_lon(x),
    km_to_lat(y),
    nk_direction,
    shading="auto",
    cmap="twilight",
    vmin=0,
    vmax=360
)

axes[1,-1].text(
    0.98, 0.01,
    'Norkyst',
    transform=axes[1,-1].transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

axes[1,-1].set_facecolor("black")

# ==================================================
# SWOT FILES
# ==================================================
# Desired SWOT order: CLS, STY, NK MDT
plot_order = [
    fil_name.index(fil_name[1]),
    fil_name.index(fil_name[0])
]

for col, i in enumerate(plot_order):

    x = x_centers_all[i]
    y = y_centers_all[i]

    direction = (
        np.degrees(
            np.arctan2(
                v_swot_map_all[i],
                u_swot_map_all[i]
            )
        ) + 360
    ) % 360

    # -------------------------
    # SPEED
    # -------------------------
    axes[0,col].pcolormesh(
        km_to_lon(x),
        km_to_lat(y),
        swot_map_all[i],
        shading="auto",
        cmap="Reds",
        vmin=0,
        vmax=0.5
    )

    axes[0,col].set_facecolor("black")

    axes[0,col].text(
        0.98, 0.01,
        fil_name[i],
        transform=axes[0,col].transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
    )

    # -------------------------
    # DIRECTION
    # -------------------------
    axes[1,col].pcolormesh(
        km_to_lon(x),
        km_to_lat(y),
        direction,
        shading="auto",
        cmap="twilight",
        vmin=0,
        vmax=360
    )

    axes[1,col].set_facecolor("black")

    axes[1,col].text(
        0.98, 0.01,
        fil_name[i],
        transform=axes[1,col].transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
    )

# ==================================================
# FORMAT
# ==================================================

for ax in axes.flat:

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(format_lon)
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(format_lat)
    )

    add_north_arrow(ax)

for ax in axes[:,1:].flat:
    ax.set_yticklabels([])

for ax in axes[0,:]:
    ax.set_xticklabels([])

# ==================================================
# COLORBARS
# ==================================================

cbar1 = fig.colorbar(
    im_speed,
    ax=axes[0,:],
    pad=0.02,
    aspect=30
)

cbar1.set_label("Speed [m/s]")

cbar2 = fig.colorbar(
    im_dir,
    ax=axes[1,:],
    pad=0.02,
    aspect=30
)

cbar2.set_label("Direction [towards]")

cbar2.set_ticks([0,90,180,270,360])
cbar2.set_ticklabels(["E","N","W","S","E"])

plt.show()
    

#%% PLOTS FOR BASELINE

    
#%% Final plots - velocities and stats (Zone 1)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0

colors = {
    "dtured": (0.6, 0.0, 0.0),
    "blue": (0.1843, 0.2431, 0.9176),
    "brightgreen": (0.1216, 0.8157, 0.5098),
    "navyblue": (0.0118, 0.0588, 0.3098),
    "yellow": (0.9647, 0.8157, 0.3019),
    "orange": (0.9882, 0.4627, 0.2039),
    "pink": (0.9686, 0.7333, 0.6941),
    "grey": (0.8549, 0.8549, 0.8549),
    "red": (0.9098, 0.2471, 0.2824),
    "green": (0.0, 0.5333, 0.2078),
    "purple": (0.4745, 0.1373, 0.5569),
}

colors = {
    "dtured": (0.6, 0.0, 0.0),
    "blue": (0.1843, 0.2431, 0.9176),
}


mark = ['-.', '--']
c = [colors['dtured'], colors['blue']]

lim = [0, 400]


# =============================================================================
# FIGURE SETUP
# =============================================================================

fig = plt.figure(figsize=(10, 14), dpi=300)

outer = fig.add_gridspec(
    2, 1,
    height_ratios=[4, 5],
    hspace=0.13
)

# =============================================================================
# TOP ROW (MAPS)
# =============================================================================

gs_top = outer[0].subgridspec(
    1, 3,
    wspace=0.1
)

file_idx = 1

# =============================================================================
# SWOT UNFILTERED
# =============================================================================

ax_map1 = fig.add_subplot(gs_top[0, 0])

im1 = ax_map1.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    swot_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map1.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map1.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map1.set_facecolor("black")
ax_map1.set_ylabel('Latitude')
ax_map1.set_xlabel('Longitude')
#ax_map1.set_xticklabels([])
#ax_map1.set_yticklabels([])



ax_map1.text(
    0.98, 0.01,
    fil_name[file_idx],
    transform=ax_map1.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# SWOT FILTERED
# =============================================================================

ax_map2 = fig.add_subplot(gs_top[0, 1])

im2 = ax_map2.pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    swot_map_all[0],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map2.contour(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    dist_map_all[0],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map2.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map2.set_facecolor("black")
#ax_map2.set_xticklabels([])
ax_map2.set_yticklabels([])
ax_map2.set_xlabel('Longitude')

ax_map2.text(
    0.98, 0.01,
    fil_name[0],
    transform=ax_map2.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# NORKYST
# =============================================================================

ax_map3 = fig.add_subplot(gs_top[0, 2])

im3 = ax_map3.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    norkyst_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map3.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map3.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map3.set_facecolor("black")
#ax_map3.set_xticklabels([])
ax_map3.set_yticklabels([])
ax_map3.set_xlabel('Longitude')

ax_map3.text(
    0.98, 0.01,
    "Norkyst",
    transform=ax_map3.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# COLORBAR (safe, no layout distortion)
cax = ax_map3.inset_axes([1.02, 0.0, 0.04, 1.0])
cbar = fig.colorbar(im3, cax=cax)
cbar.set_label("Velocity magnitude [m/s]")


def add_north_arrow(ax, x=0.92, y=0.08, length=0.15, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(ax_map1)
add_north_arrow(ax_map2)
add_north_arrow(ax_map3)

# =============================================================================
# BOTTOM BLOCK
# =============================================================================

gs_bot = outer[1].subgridspec(
    3, 2,
    hspace=0.2,
    wspace=0.2
)

ax_speed = fig.add_subplot(gs_bot[0, 0])
ax_u     = fig.add_subplot(gs_bot[1, 0], sharex=ax_speed)
ax_v     = fig.add_subplot(gs_bot[2, 0], sharex=ax_speed)

ax_bias = fig.add_subplot(gs_bot[0, 1])
ax_std  = fig.add_subplot(gs_bot[1, 1], sharex=ax_bias)
ax_dir  = fig.add_subplot(gs_bot[2, 1], sharex=ax_bias)

# =============================================================================
# STYLING SETUP (RESTORED)
# =============================================================================

file_idx = 1

# =============================================================================
# PLOTTING LOOP (RESTORED STYLE)
# =============================================================================

for i in range(n_files):

    ax_speed.plot(
        bin_centers_all[i],
        swot_mean_all[i],
        mark[i],
        color=c[i],
        label=fil_name[i]
    )
    
   
    ax_u.plot(bin_centers_all[i], swot_umean_all[i], mark[i], color=c[i])
    ax_v.plot(bin_centers_all[i], swot_vmean_all[i], mark[i], color=c[i])


    ax_bias.plot(bin_centers_bias_all[i], bias_mean_all[i], mark[i], color=c[i], label=fil_name[i])
    ax_std.plot(bin_centers_bias_all[i], bias_std_all[i], mark[i], color=c[i])

    ax_dir.plot(
        bin_centers_bias_all[i],
        mean_sign_mean[i],
        mark[i],
        color=c[i],
        label='_nolegend_'
    )

    

    
    

# =============================================================================
# NORKYST OVERLAY
# =============================================================================

ax_speed.plot(
    bin_centers_all[file_idx],
    norkyst_mean_all[file_idx],
    'k-',
    linewidth=2,
    label="Norkyst"
)

ax_u.plot(bin_centers_all[file_idx], norkyst_umean_all[file_idx], 'k-', linewidth=2)
ax_v.plot(bin_centers_all[file_idx], norkyst_vmean_all[file_idx], 'k-', linewidth=2)


# =============================================================================
# FORMATTING
# =============================================================================

for ax in [ax_speed, ax_u, ax_v,
           ax_bias, ax_std,  ax_dir]:
    ax.grid(True)
    ax.set_xlim(lim)

ax_speed.set_ylabel(r"$\overline{V_g}$ [m/s]")
ax_u.set_ylabel(r"$\overline{u_g}$ (m/s)")
ax_v.set_ylabel(r"$\overline{v_g}$ (m/s)")


ax_bias.set_ylabel("Mean bias (m/s)")
ax_std.set_ylabel("Std (m/s)")

ax_dir.set_ylabel("Direction diff (°)")
ax_dir.set_xlabel("Distance to coast (km)")
ax_v.set_xlabel("Distance to coast (km)")
ax_dir.set_ylim([0,180])
ax_std.set_ylim([0,0.3])
ax_v.set_ylim([-0.1,0.15])              
           


for ax in [ax_speed, ax_u, 
           ax_bias, ax_std]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_speed.legend(fontsize=8)
ax_bias.legend(fontsize=8)

# =============================================================================
# PANEL LABELS (RESTORED)
# =============================================================================

def panel_label(ax, label):
    ax.text(
        0.01, 0.99,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=1)
    )

panel_label(ax_map1, "(a)")
panel_label(ax_map2, "(b)")
panel_label(ax_map3, "(c)")
panel_label(ax_speed, "(d)")
panel_label(ax_u, "(e)")
panel_label(ax_v, "(f)")
panel_label(ax_bias, "(g)")
panel_label(ax_std, "(h)")
panel_label(ax_dir, "(i)")

plt.show()


#%% Final plots - spatial (zone 1)
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


fig, axs = plt.subplots(
    4, 2,
    figsize=(9, 11),
    dpi=300,
    sharex=True,
    sharey=True,
    width_ratios=[0.94,1]
)

# =====================================================
# COLUMN 0 = FILTERED (0)
# COLUMN 1 = UNFILTERED (1)
# COORDINATES CONVERTED TO lon/lat
# DATA TRANSPOSED (.T)
# =====================================================

# -----------------------------
# BIAS
# -----------------------------
im0 = axs[0, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    bias_map_all[1],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    bias_map_all[0],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

axs[0, 1].set_title(fil_name[0], fontsize=14, fontweight="bold", pad=12)
axs[0, 0].set_title(fil_name[1], fontsize=14, fontweight="bold", pad=12)


fig.colorbar(im1, ax=axs[0, 1], fraction=0.046,label="Mean Residual, Bias [m/s]")

# -----------------------------
# STD
# -----------------------------
im2 = axs[1, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    std_map_all[1],
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    std_map_all[0],
    cmap="Reds", vmin=0, vmax=0.25
)


fig.colorbar(im3, ax=axs[1, 1], fraction=0.046,label="Standard deviation [m/s]")

# -----------------------------
# ANGLE
# -----------------------------
im4 = axs[2, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    angle_map_all[1],
    cmap="Reds", vmin=0, vmax=180
)

im5 = axs[2, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    angle_map_all[0],
    cmap="Reds", vmin=0, vmax=180
)


fig.colorbar(im5, ax=axs[2, 1], fraction=0.046,label=r'Direction misalignemnt [$^\circ$]')

# -----------------------------
# AGREEMENT
# -----------------------------
im6 = axs[3, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    sign_map_all[1],
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    sign_map_all[0],
    cmap="Reds", vmin=0, vmax=1
)


fig.colorbar(im7, ax=axs[3, 1], fraction=0.046,label='Sign agreement ratio [-]')

# -----------------------------
# AXES LABELS
# -----------------------------
#for ax in axs.ravel():
 
axs[0, 0].set_ylabel("Longitude")
axs[1, 0].set_ylabel("Longitude")
axs[2, 0].set_ylabel("Longitude")
axs[3, 0].set_ylabel("Longitude")

axs[3, 0].set_xlabel("Latitude")
axs[3, 1].set_xlabel("Latitude")

axs[0, 0].set_facecolor("black")
axs[1, 0].set_facecolor("black")
axs[2, 0].set_facecolor("black")
axs[3, 0].set_facecolor("black")


axs[0, 1].set_facecolor("black")
axs[1, 1].set_facecolor("black")
axs[2, 1].set_facecolor("black")
axs[3, 1].set_facecolor("black")

labels = [
    "(a)", "(b)",
    "(c)", "(d)",
    "(e)", "(f)",
    "(g)", "(h)"
]

for ax, lab in zip(axs.ravel(), labels):
    ax.text(
        0.01, 0.99, lab,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=2)
    )



def add_north_arrow(ax, x=0.92, y=0.08, length=0.25, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(axs[0, 0])
add_north_arrow(axs[1, 0])
add_north_arrow(axs[2, 0])
add_north_arrow(axs[3, 0])


add_north_arrow(axs[0, 1])
add_north_arrow(axs[1, 1])
add_north_arrow(axs[2, 1])
add_north_arrow(axs[3, 1])

plt.tight_layout()
plt.show()

#%% Final plots - velocities and stats (Zone 2)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


colors = {
    "dtured": (0.6, 0.0, 0.0),
    "blue": (0.1843, 0.2431, 0.9176),
    "brightgreen": (0.1216, 0.8157, 0.5098),
    "navyblue": (0.0118, 0.0588, 0.3098),
    "yellow": (0.9647, 0.8157, 0.3019),
    "orange": (0.9882, 0.4627, 0.2039),
    "pink": (0.9686, 0.7333, 0.6941),
    "grey": (0.8549, 0.8549, 0.8549),
    "red": (0.9098, 0.2471, 0.2824),
    "green": (0.0, 0.5333, 0.2078),
    "purple": (0.4745, 0.1373, 0.5569),
}

mark = ['-.', '--','-*']
c = [colors['dtured'], colors['blue'], colors['purple']]

lim = [0, 125]


# =============================================================================
# FIGURE SETUP
# =============================================================================

fig = plt.figure(figsize=(10, 12), dpi=300)

outer = fig.add_gridspec(
    2, 1,
    height_ratios=[4, 5],
    hspace=0.11
)

# =============================================================================
# TOP ROW (MAPS)
# =============================================================================

gs_top = outer[0].subgridspec(
    1, 3,
    wspace=0.1
)

file_idx = 1

# =============================================================================
# SWOT UNFILTERED
# =============================================================================

ax_map1 = fig.add_subplot(gs_top[0, 0])

im1 = ax_map1.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    swot_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map1.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map1.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map1.set_facecolor("black")
ax_map1.set_ylabel('Latitude')
ax_map1.set_xlabel('Longitude')
#ax_map1.set_xticklabels([])
#ax_map1.set_yticklabels([])

ax_map1.text(
    0.98, 0.01,
    fil_name[1],
    transform=ax_map1.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# SWOT FILTERED
# =============================================================================

ax_map2 = fig.add_subplot(gs_top[0, 1])

im2 = ax_map2.pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    swot_map_all[0],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map2.contour(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    dist_map_all[0],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map2.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map2.set_facecolor("black")
ax_map2.set_xlabel('Longitude')
#ax_map2.set_xticklabels([])
ax_map2.set_yticklabels([])

ax_map2.text(
    0.98, 0.01,
    fil_name[0],
    transform=ax_map2.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# NORKYST
# =============================================================================

ax_map3 = fig.add_subplot(gs_top[0, 2])

im3 = ax_map3.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    norkyst_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map3.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map3.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map3.set_facecolor("black")
ax_map3.set_xlabel('Longitude')
#ax_map3.set_xticklabels([])
ax_map3.set_yticklabels([])

ax_map3.text(
    0.98, 0.01,
    "Norkyst",
    transform=ax_map3.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# COLORBAR (safe, no layout distortion)
cax = ax_map3.inset_axes([1.02, 0.0, 0.04, 1.0])
cbar = fig.colorbar(im3, cax=cax)
cbar.set_label("Velocity magnitude [m/s]")

# =============================================================================
# BOTTOM BLOCK
# =============================================================================

gs_bot = outer[1].subgridspec(
    4, 2,
    hspace=0.2,
    wspace=0.2
)

ax_speed = fig.add_subplot(gs_bot[0, 0])
ax_u     = fig.add_subplot(gs_bot[1, 0], sharex=ax_speed)
ax_v     = fig.add_subplot(gs_bot[2, 0], sharex=ax_speed)
ax_grad  = fig.add_subplot(gs_bot[3, 0], sharex=ax_speed)

ax_bias = fig.add_subplot(gs_bot[0, 1])
ax_std  = fig.add_subplot(gs_bot[1, 1], sharex=ax_bias)
ax_sign = fig.add_subplot(gs_bot[2, 1], sharex=ax_bias)
ax_dir  = fig.add_subplot(gs_bot[3, 1], sharex=ax_bias)

# =============================================================================
# STYLING SETUP (RESTORED)
# =============================================================================

file_idx = 1

# =============================================================================
# PLOTTING LOOP (RESTORED STYLE)
# =============================================================================

for i in range(n_files):

    ax_speed.plot(
        bin_centers_all[i],
        swot_mean_all[i],
        mark[i],
        color=c[i],
        label=fil_name[i]
    )

    ax_u.plot(bin_centers_all[i], swot_umean_all[i], mark[i], color=c[i])
    ax_v.plot(bin_centers_all[i], swot_vmean_all[i], mark[i], color=c[i])
    ax_grad.plot(bin_centers_all[i], swot_grad_mean_all[i] * 1e5, mark[i], color=c[i])

    ax_bias.plot(bin_centers_bias_all[i], bias_mean_all[i], mark[i], color=c[i], label=fil_name[i])
    ax_std.plot(bin_centers_bias_all[i], bias_std_all[i], mark[i], color=c[i])
    ax_sign.plot(bin_centers_bias_all[i], mean_sign_all[i], mark[i], color=c[i])
    ax_dir.plot(bin_centers_bias_all[i], mean_sign_mean[i], mark[i], color=c[i])

# =============================================================================
# NORKYST OVERLAY
# =============================================================================

ax_speed.plot(
    bin_centers_all[file_idx],
    norkyst_mean_all[file_idx],
    'k-',
    linewidth=2,
    label="Norkyst"
)

ax_u.plot(bin_centers_all[file_idx], norkyst_umean_all[file_idx], 'k-', linewidth=2)
ax_v.plot(bin_centers_all[file_idx], norkyst_vmean_all[file_idx], 'k-', linewidth=2)
ax_grad.plot(bin_centers_all[file_idx], norkyst_grad_mean_all[file_idx] * 1e5, 'k-', linewidth=2)

# =============================================================================
# FORMATTING
# =============================================================================

for ax in [ax_speed, ax_u, ax_v, ax_grad,
           ax_bias, ax_std, ax_sign, ax_dir]:
    ax.grid(True)
    ax.set_xlim(lim)

ax_speed.set_ylabel(r"$\overline{V_g}$ [m/s]")
ax_u.set_ylabel(r"$\overline{u_g}$ (m/s)")
ax_v.set_ylabel(r"$\overline{v_g}$ (m/s)")
ax_grad.set_ylabel(r"$\partial \mathrm{ADT} / \partial x$ (cm/km)")
ax_grad.set_xlabel("Distance to coast (km)")

ax_bias.set_ylabel("Mean bias (m/s)")
ax_std.set_ylabel("Std (m/s)")
ax_sign.set_ylabel("Sign agreement")
ax_dir.set_ylabel("Direction diff (°)")
ax_dir.set_xlabel("Distance to coast (km)")


ax_dir.set_ylim([0,180])
ax_sign.set_ylim([0,1])
ax_std.set_ylim([0,0.4])

for ax in [ax_speed, ax_u, ax_v,
           ax_bias, ax_std, ax_sign]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_speed.legend(fontsize=8)
ax_bias.legend(fontsize=8)


def add_north_arrow(ax, x=0.92, y=0.3, length=0.15, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(ax_map1)
add_north_arrow(ax_map2)
add_north_arrow(ax_map3)

# =============================================================================
# PANEL LABELS (RESTORED)
# =============================================================================

def panel_label(ax, label):
    ax.text(
        0.01, 0.99,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=1)
    )

panel_label(ax_map1, "(a)")
panel_label(ax_map2, "(b)")
panel_label(ax_map3, "(c)")
panel_label(ax_speed, "(d)")
panel_label(ax_u, "(e)")
panel_label(ax_v, "(f)")
panel_label(ax_grad, "(g)")
panel_label(ax_bias, "(h)")
panel_label(ax_std, "(i)")
panel_label(ax_sign, "(j)")
panel_label(ax_dir, "(k)")

plt.show()

#%% Final plots - spatial (zone 2)
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


fig, axs = plt.subplots(
    4, 2,
    figsize=(10, 11),
    dpi=300,
    sharex=True,
    sharey=True,
    width_ratios=[0.94,1]
)

# =====================================================
# COLUMN 0 = FILTERED (0)
# COLUMN 1 = UNFILTERED (1)
# COORDINATES CONVERTED TO lon/lat
# DATA TRANSPOSED (.T)
# =====================================================

# -----------------------------
# BIAS
# -----------------------------
im0 = axs[0, 0].pcolormesh(
    km_to_lat(y_centers_all[1]),
    km_to_lon(x_centers_all[1]),
    bias_map_all[1].T,
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 1].pcolormesh(
    km_to_lat(y_centers_all[0]),
    km_to_lon(x_centers_all[0]),
    bias_map_all[0].T,
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

axs[0, 1].set_title(fil_name[0], fontsize=14, fontweight="bold", pad=12)
axs[0, 0].set_title(fil_name[1], fontsize=14, fontweight="bold", pad=12)


fig.colorbar(im1, ax=axs[0, 1], fraction=0.046,label="Mean Residual, Bias [m/s]")

# -----------------------------
# STD
# -----------------------------
im2 = axs[1, 0].pcolormesh(
    km_to_lat(y_centers_all[1]),
    km_to_lon(x_centers_all[1]),
    std_map_all[1].T,
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 1].pcolormesh(
    km_to_lat(y_centers_all[0]),
    km_to_lon(x_centers_all[0]),
    std_map_all[0].T,
    cmap="Reds", vmin=0, vmax=0.25
)


fig.colorbar(im3, ax=axs[1, 1], fraction=0.046,label="Standard deviation [m/s]")

# -----------------------------
# ANGLE
# -----------------------------
im4 = axs[2, 0].pcolormesh(
    km_to_lat(y_centers_all[1]),
    km_to_lon(x_centers_all[1]),
    angle_map_all[1].T,
    cmap="Reds", vmin=0, vmax=180
)

im5 = axs[2, 1].pcolormesh(
    km_to_lat(y_centers_all[0]),
    km_to_lon(x_centers_all[0]),
    angle_map_all[0].T,
    cmap="Reds", vmin=0, vmax=180
)


fig.colorbar(im5, ax=axs[2, 1], fraction=0.046,label=r'Direction misalignemnt [$^\circ$]')

# -----------------------------
# AGREEMENT
# -----------------------------
im6 = axs[3, 0].pcolormesh(
    km_to_lat(y_centers_all[1]),
    km_to_lon(x_centers_all[1]),
    sign_map_all[1].T,
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 1].pcolormesh(
    km_to_lat(y_centers_all[0]),
    km_to_lon(x_centers_all[0]),
    sign_map_all[0].T,
    cmap="Reds", vmin=0, vmax=1
)


fig.colorbar(im7, ax=axs[3, 1], fraction=0.046,label='Sign agreement ratio [-]')

# -----------------------------
# AXES LABELS
# -----------------------------
#for ax in axs.ravel():
 
axs[0, 0].set_ylabel("Longitude")
axs[1, 0].set_ylabel("Longitude")
axs[2, 0].set_ylabel("Longitude")
axs[3, 0].set_ylabel("Longitude")

axs[3, 0].set_xlabel("Latitude")
axs[3, 1].set_xlabel("Latitude")

axs[0, 0].set_facecolor("black")
axs[1, 0].set_facecolor("black")
axs[2, 0].set_facecolor("black")
axs[3, 0].set_facecolor("black")


axs[0, 1].set_facecolor("black")
axs[1, 1].set_facecolor("black")
axs[2, 1].set_facecolor("black")
axs[3, 1].set_facecolor("black")

labels = [
    "(a)", "(b)",
    "(c)", "(d)",
    "(e)", "(f)",
    "(g)", "(h)"
]

for ax, lab in zip(axs.ravel(), labels):
    ax.text(
        0.01, 0.99, lab,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=2)
    )

def add_north_arrow(ax, x=60, y=5.5, length=1, color='white'):
    ax.annotate(
        'N',
        xy=(x + length, y),
        xytext=(x, y),
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )
    ax.set_aspect('equal')

add_north_arrow(axs[0, 0])
add_north_arrow(axs[1, 0])
add_north_arrow(axs[2, 0])
add_north_arrow(axs[3, 0])


add_north_arrow(axs[0, 1])
add_north_arrow(axs[1, 1])
add_north_arrow(axs[2, 1])
add_north_arrow(axs[3, 1])

plt.tight_layout()
plt.show()

#%% Bias plots - cartopy

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0

# -----------------------------
# two Cartopy plots side by side
# -----------------------------

lon_min, lon_max = 1.5, 6
lat_min, lat_max = 58.5, 63.5

proj = ccrs.Orthographic(
    central_longitude=(lon_min + lon_max) / 2,
    central_latitude=(lat_min + lat_max) / 2
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(5, 8),
    subplot_kw={'projection': proj},
    dpi=200
)

# convert km grid to lon/lat
lon_grid = km_to_lon(x_centers_all[1])
lat_grid = km_to_lat(y_centers_all[1])

# titles for each map
titles = [fil_name[0], fil_name[1]]

# store image handles for colorbar
ims = []

for i, ax in enumerate(axes):
    i = 1 - i   # axes[0] gets bias_map_all[1], axes[1] gets bias_map_all[0]

    ax.set_extent([3, lon_max, lat_min, 63])

    ax.add_feature(cfeature.LAND, color='black')
    ax.coastlines(resolution="10m", linewidth=0.8)

    # plot bias
    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        bias_map_all[i],
        cmap="RdBu_r",
        vmin=-0.25,
        vmax=0.25,
        shading="auto",
        transform=ccrs.PlateCarree()
    )

    ims.append(im)

    ax.set_title(
        titles[i],
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    # gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        alpha=0.5,
        linestyle="--"
    )

    gl.top_labels = False
    gl.right_labels = False

    # only left plot gets left labels
    if i == 0:
        gl.left_labels = False
    else:
        gl.left_labels = True

# common colorbar
cbar = fig.colorbar(
    ims[1],
    ax=axes,
    fraction=0.06,
    pad=0.05
)

cbar.set_label("Mean Residual, Bias [m/s]")

plt.show()

#%% PLOTS FOR MDT

#%% zone 2: dist2coast


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


colors = {
    "dtured": (0.6, 0.0, 0.0),
    "blue": (0.1843, 0.2431, 0.9176),
    "brightgreen": (0.1216, 0.8157, 0.5098),
    "navyblue": (0.0118, 0.0588, 0.3098),
    "yellow": (0.9647, 0.8157, 0.3019),
    "orange": (0.9882, 0.4627, 0.2039),
    "pink": (0.9686, 0.7333, 0.6941),
    "grey": (0.8549, 0.8549, 0.8549),
    "red": (0.9098, 0.2471, 0.2824),
    "green": (0.0, 0.5333, 0.2078),
    "purple": (0.4745, 0.1373, 0.5569),
}

mark = ['-', '--','-.']
c = [colors['purple'], colors['blue'], colors['dtured']]

lim = [0, 125]


# =============================================================================
# FIGURE SETUP
# =============================================================================

fig = plt.figure(figsize=(10, 12), dpi=300)

outer = fig.add_gridspec(
    2, 1,
    height_ratios=[4, 5],
    hspace=0.11
)

# =============================================================================
# TOP ROW (MAPS)
# =============================================================================

gs_top = outer[0].subgridspec(
    1, 4,
    wspace=0.1
)

file_idx = 1

# =============================================================================
# SWOT UNFILTERED
# =============================================================================

ax_map1 = fig.add_subplot(gs_top[0, 1])

im1 = ax_map1.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    swot_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map1.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map1.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map1.set_facecolor("black")
ax_map1.set_ylabel('Latitude')
ax_map1.set_xlabel('Longitude')
#ax_map1.set_xticklabels([])
ax_map1.set_yticklabels([])

ax_map1.text(
    0.98, 0.01,
    fil_name[1],
    transform=ax_map1.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# SWOT FILTERED
# =============================================================================

ax_map2 = fig.add_subplot(gs_top[0, 2])

im2 = ax_map2.pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    swot_map_all[0],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map2.contour(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    dist_map_all[0],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map2.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map2.set_facecolor("black")
ax_map2.set_xlabel('Longitude')
#ax_map2.set_xticklabels([])
ax_map2.set_yticklabels([])

ax_map2.text(
    0.98, 0.01,
    fil_name[0],
    transform=ax_map2.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)


# =============================================================================
# DTU
# =============================================================================

ax_map4 = fig.add_subplot(gs_top[0, 0])

im3 = ax_map4.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    swot_map_all[2],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map4.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map4.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map4.set_facecolor("black")
ax_map4.set_xlabel('Longitude')
#ax_map3.set_xticklabels([])
#ax_map4.set_yticklabels([])

ax_map4.text(
    0.98, 0.01,
    fil_name[2],
    transform=ax_map4.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# NORKYST
# =============================================================================

ax_map3 = fig.add_subplot(gs_top[0, 3])

im3 = ax_map3.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    norkyst_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map3.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map3.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map3.set_facecolor("black")
ax_map3.set_xlabel('Longitude')
#ax_map3.set_xticklabels([])
ax_map3.set_yticklabels([])

ax_map3.text(
    0.98, 0.01,
    "Norkyst",
    transform=ax_map3.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# COLORBAR (safe, no layout distortion)
cax = ax_map3.inset_axes([1.02, 0.0, 0.04, 1.0])
cbar = fig.colorbar(im3, cax=cax)
cbar.set_label("Velocity magnitude [m/s]")

# =============================================================================
# BOTTOM BLOCK
# =============================================================================

gs_bot = outer[1].subgridspec(
    4, 2,
    hspace=0.2,
    wspace=0.3,
    height_ratios=[1, 0.1, 1, 1]
)

ax_speed = fig.add_subplot(gs_bot[0:2, 0])
ax_u     = fig.add_subplot(gs_bot[2, 0], sharex=ax_speed)
ax_v     = fig.add_subplot(gs_bot[3, 0], sharex=ax_speed)


ax_bias = fig.add_subplot(gs_bot[0:2, 1])
ax_std  = fig.add_subplot(gs_bot[2, 1], sharex=ax_bias)
ax_dir  = fig.add_subplot(gs_bot[3, 1], sharex=ax_bias)

# =============================================================================
# STYLING SETUP (RESTORED)
# =============================================================================

file_idx = 1


# =============================================================================
# NORKYST OVERLAY
# =============================================================================


ax_speed.plot(
    bin_centers_all[file_idx],
    norkyst_mean_all[file_idx],
    'k-',
    linewidth=2,
    label="Norkyst"
)


ax_u.plot(bin_centers_all[file_idx], norkyst_umean_all[file_idx], 'k-', linewidth=2)
ax_v.plot(bin_centers_all[file_idx], norkyst_vmean_all[file_idx], 'k-', linewidth=2)


# =============================================================================
# PLOTTING LOOP (RESTORED STYLE)
# =============================================================================

for i in range(n_files):

    ax_speed.plot(
        bin_centers_all[i],
        swot_mean_all[i],
        mark[i],
        color=c[i],
        label=fil_name[i]
    )

    ax_u.plot(bin_centers_all[i], swot_umean_all[i], mark[i], color=c[i])
    ax_v.plot(bin_centers_all[i], swot_vmean_all[i], mark[i], color=c[i])
   
    ax_bias.plot(bin_centers_bias_all[i], bias_mean_all[i], mark[i], color=c[i], label=fil_name[i])
    ax_std.plot(bin_centers_bias_all[i], bias_std_all[i], mark[i], color=c[i])
   
    ax_dir.plot(bin_centers_bias_all[i], mean_sign_mean[i], mark[i], color=c[i])

# =============================================================================
# FORMATTING
# =============================================================================

for ax in [ax_speed, ax_u, ax_v,
           ax_bias, ax_std, ax_dir]:
    ax.grid(True)
    ax.set_xlim(lim)

ax_speed.set_ylabel(r"$\overline{V_g}$ [m/s]")
ax_u.set_ylabel(r"$\overline{u_g}$ [m/s]")
ax_v.set_ylabel(r"$\overline{v_g}$ [m/s]")
ax_v.set_xlabel("Distance to coast [km]")

ax_bias.set_ylabel("Mean bias [m/s]")
ax_std.set_ylabel("Std [m/s]")
ax_dir.set_ylabel("Direction diff [°]")
ax_dir.set_xlabel("Distance to coast [km]")


ax_dir.set_ylim([0,180])
ax_std.set_ylim([0.05,0.5])
ax_u.set_ylim([-0.06,0.05])

for ax in [ax_speed, ax_u,
           ax_bias, ax_std]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_speed.legend(fontsize=8)
ax_bias.legend(fontsize=8)


def add_north_arrow(ax, x=0.92, y=0.3, length=0.15, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(ax_map1)
add_north_arrow(ax_map2)
add_north_arrow(ax_map3)
add_north_arrow(ax_map4)

# =============================================================================
# PANEL LABELS (RESTORED)
# =============================================================================

def panel_label(ax, label):
    ax.text(
        0.01, 0.99,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=1)
    )

panel_label(ax_map4, "(a)")
panel_label(ax_map1, "(b)")
panel_label(ax_map2, "(c)")
panel_label(ax_map3, "(d)")
panel_label(ax_speed, "(e)")
panel_label(ax_u, "(f)")
panel_label(ax_v, "(g)")
panel_label(ax_bias, "(h)")
panel_label(ax_std, "(i)")
panel_label(ax_dir, "(j)")

plt.show()


#%% zone 2:spatial


# ----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


fig, axs = plt.subplots(
    4, 3,
    figsize=(9, 11),
    dpi=300,
    sharex=True,
    sharey=True,
    width_ratios=[0.94,1,1]
)

# =====================================================
# COLUMN 0 = FILTERED (0)
# COLUMN 1 = UNFILTERED (1)
# COORDINATES CONVERTED TO lon/lat
# DATA TRANSPOSED (.T)
# =====================================================

# -----------------------------
# BIAS
# -----------------------------
im0 = axs[0, 0].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    bias_map_all[2],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    bias_map_all[0],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 1].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    bias_map_all[1],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

axs[0, 0].set_title(fil_name[2], fontsize=14, fontweight="bold", pad=12)
axs[0, 1].set_title(fil_name[1], fontsize=14, fontweight="bold", pad=12)
axs[0, 2].set_title(fil_name[0], fontsize=14, fontweight="bold", pad=12)


fig.colorbar(im1, ax=axs[0, 2], fraction=0.046,label="Mean Residual, Bias [m/s]")

# -----------------------------
# STD
# -----------------------------
im2 = axs[1, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    std_map_all[1],
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    std_map_all[0],
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    std_map_all[2],
    cmap="Reds", vmin=0, vmax=0.25
)



fig.colorbar(im3, ax=axs[1, 2], fraction=0.046,label="Standard deviation [m/s]")

# -----------------------------
# ANGLE
# -----------------------------
im4 = axs[2, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    angle_map_all[1],
    cmap="Reds", vmin=0, vmax=180
)

im5 = axs[2, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    angle_map_all[0],
    cmap="Reds", vmin=0, vmax=180
)


im5 = axs[2, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    angle_map_all[2],
    cmap="Reds", vmin=0, vmax=180
)


fig.colorbar(im5, ax=axs[2, 2], fraction=0.046,label=r'Direction misalignemnt [$^\circ$]')

# -----------------------------
# AGREEMENT
# -----------------------------
im6 = axs[3, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    sign_map_all[1],
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    sign_map_all[0],
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    sign_map_all[2],
    cmap="Reds", vmin=0, vmax=1
)


fig.colorbar(im7, ax=axs[3, 2], fraction=0.046,label='Sign agreement ratio [-]')

# -----------------------------
# AXES LABELS
# -----------------------------
#for ax in axs.ravel():
 
axs[0, 0].set_ylabel("Longitude")
axs[1, 0].set_ylabel("Longitude")
axs[2, 0].set_ylabel("Longitude")
axs[3, 0].set_ylabel("Longitude")

axs[3, 0].set_xlabel("Latitude")
axs[3, 1].set_xlabel("Latitude")
axs[3, 2].set_xlabel("Latitude")

axs[0, 0].set_facecolor("black")
axs[1, 0].set_facecolor("black")
axs[2, 0].set_facecolor("black")
axs[3, 0].set_facecolor("black")


axs[0, 1].set_facecolor("black")
axs[1, 1].set_facecolor("black")
axs[2, 1].set_facecolor("black")
axs[3, 1].set_facecolor("black")

axs[0, 2].set_facecolor("black")
axs[1, 2].set_facecolor("black")
axs[2, 2].set_facecolor("black")
axs[3, 2].set_facecolor("black")

labels = [
    "(a)", "(b)",
    "(c)", "(d)",
    "(e)", "(f)",
    "(g)", "(h)",
    "(i)","(j)",
    "(k)", "(l)"
]

for ax, lab in zip(axs.ravel(), labels):
    ax.text(
        0.01, 0.99, lab,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=2)
    )



def add_north_arrow(ax, x=0.92, y=0.13, length=0.25, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(axs[0, 0])
add_north_arrow(axs[1, 0])
add_north_arrow(axs[2, 0])
add_north_arrow(axs[3, 0])


add_north_arrow(axs[0, 1])
add_north_arrow(axs[1, 1])
add_north_arrow(axs[2, 1])
add_north_arrow(axs[3, 1])

add_north_arrow(axs[0, 2])
add_north_arrow(axs[1, 2])
add_north_arrow(axs[2, 2])
add_north_arrow(axs[3, 2])

plt.tight_layout()
plt.show()


#%% zone 1: dist2coast


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


colors = {
    "dtured": (0.6, 0.0, 0.0),
    "blue": (0.1843, 0.2431, 0.9176),
    "brightgreen": (0.1216, 0.8157, 0.5098),
    "navyblue": (0.0118, 0.0588, 0.3098),
    "yellow": (0.9647, 0.8157, 0.3019),
    "orange": (0.9882, 0.4627, 0.2039),
    "pink": (0.9686, 0.7333, 0.6941),
    "grey": (0.8549, 0.8549, 0.8549),
    "red": (0.9098, 0.2471, 0.2824),
    "green": (0.0, 0.5333, 0.2078),
    "purple": (0.4745, 0.1373, 0.5569),
}

mark = ['-', '--','-.']
c = [colors['purple'], colors['blue'], colors['dtured']]

lim = [0, 400]


# =============================================================================
# FIGURE SETUP
# =============================================================================

fig = plt.figure(figsize=(10, 12), dpi=300)

outer = fig.add_gridspec(
    2, 1,
    height_ratios=[4, 5],
    hspace=0.11
)

# =============================================================================
# TOP ROW (MAPS)
# =============================================================================

gs_top = outer[0].subgridspec(
    1, 4,
    wspace=0.1
)

file_idx = 2

# =============================================================================
# CNES
# =============================================================================

ax_map1 = fig.add_subplot(gs_top[0, 0])

im1 = ax_map1.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    swot_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map1.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map1.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map1.set_facecolor("black")
ax_map1.set_ylabel('Latitude')
ax_map1.set_xlabel('Longitude')
#ax_map1.set_xticklabels([])
#ax_map1.set_yticklabels([])

ax_map1.text(
    0.98, 0.01,
    fil_name[2],
    transform=ax_map1.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# DTU
# =============================================================================

ax_map2 = fig.add_subplot(gs_top[0, 1])

im2 = ax_map2.pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    swot_map_all[1],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map2.contour(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    dist_map_all[1],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map2.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map2.set_facecolor("black")
ax_map2.set_xlabel('Longitude')
#ax_map2.set_xticklabels([])
ax_map2.set_yticklabels([])

ax_map2.text(
    0.98, 0.01,
    fil_name[1],
    transform=ax_map2.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)


# =============================================================================
# Norkyst
# =============================================================================

ax_map4 = fig.add_subplot(gs_top[0, 2])

im3 = ax_map4.pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    swot_map_all[0],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map4.contour(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    dist_map_all[0],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map4.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map4.set_facecolor("black")
ax_map4.set_xlabel('Longitude')
#ax_map3.set_xticklabels([])
ax_map4.set_yticklabels([])

ax_map4.text(
    0.98, 0.01,
    fil_name[0],
    transform=ax_map4.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# NORKYST
# =============================================================================

ax_map3 = fig.add_subplot(gs_top[0, 3])

im3 = ax_map3.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    norkyst_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map3.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map3.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map3.set_facecolor("black")
ax_map3.set_xlabel('Longitude')
#ax_map3.set_xticklabels([])
ax_map3.set_yticklabels([])

ax_map3.text(
    0.98, 0.01,
    "Norkyst",
    transform=ax_map3.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# COLORBAR (safe, no layout distortion)
cax = ax_map3.inset_axes([1.02, 0.0, 0.04, 1.0])
cbar = fig.colorbar(im3, cax=cax)
cbar.set_label("Velocity magnitude [m/s]")

# =============================================================================
# BOTTOM BLOCK
# =============================================================================

gs_bot = outer[1].subgridspec(
    4, 2,
    hspace=0.2,
    wspace=0.3,
    height_ratios=[1,0.2,1,1]
)

ax_speed = fig.add_subplot(gs_bot[0:2, 0])
ax_u     = fig.add_subplot(gs_bot[2, 0], sharex=ax_speed)
ax_v     = fig.add_subplot(gs_bot[3, 0], sharex=ax_speed)


ax_bias = fig.add_subplot(gs_bot[0:2, 1])
ax_std  = fig.add_subplot(gs_bot[2, 1], sharex=ax_bias)
ax_dir  = fig.add_subplot(gs_bot[3, 1], sharex=ax_bias)

# =============================================================================
# STYLING SETUP (RESTORED)
# =============================================================================

file_idx = 1

# =============================================================================
# PLOTTING LOOP (RESTORED STYLE)
# =============================================================================

for i in range(n_files):

    ax_speed.plot(
        bin_centers_all[i],
        swot_mean_all[i],
        mark[i],
        color=c[i],
        label=fil_name[i]
    )

    ax_u.plot(bin_centers_all[i], swot_umean_all[i], mark[i], color=c[i])
    ax_v.plot(bin_centers_all[i], swot_vmean_all[i], mark[i], color=c[i])
   
    ax_bias.plot(bin_centers_bias_all[i], bias_mean_all[i], mark[i], color=c[i], label=fil_name[i])
    ax_std.plot(bin_centers_bias_all[i], bias_std_all[i], mark[i], color=c[i])
   
    ax_dir.plot(bin_centers_bias_all[i], mean_sign_mean[i], mark[i], color=c[i])

# =============================================================================
# NORKYST OVERLAY
# =============================================================================

ax_speed.plot(
    bin_centers_all[file_idx],
    norkyst_mean_all[file_idx],
    'k-',
    linewidth=2,
    label="Norkyst"
)

ax_u.plot(bin_centers_all[file_idx], norkyst_umean_all[file_idx], 'k-', linewidth=2)
ax_v.plot(bin_centers_all[file_idx], norkyst_vmean_all[file_idx], 'k-', linewidth=2)

# =============================================================================
# FORMATTING
# =============================================================================

for ax in [ax_speed, ax_u, ax_v,
           ax_bias, ax_std, ax_dir]:
    ax.grid(True)
    ax.set_xlim(lim)

ax_speed.set_ylabel(r"$\overline{V_g}$ [m/s]")
ax_u.set_ylabel(r"$\overline{u_g}$ [m/s]")
ax_v.set_ylabel(r"$\overline{v_g}$ [m/s]")
ax_v.set_xlabel("Distance to coast [km]")

ax_bias.set_ylabel("Mean bias [m/s]")
ax_std.set_ylabel("Std [m/s]")
ax_dir.set_ylabel("Direction diff [°]")
ax_dir.set_xlabel("Distance to coast [km]")


ax_dir.set_ylim([0,180])
ax_std.set_ylim([0.1,0.26])
ax_u.set_ylim([-0.1,0.1])
ax_v.set_ylim([-0.1,0.2])

for ax in [ax_speed, ax_u,
           ax_bias, ax_std]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_speed.legend(fontsize=8)
ax_bias.legend(fontsize=8)


def add_north_arrow(ax, x=0.92, y=0.1, length=0.15, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(ax_map1)
add_north_arrow(ax_map2)
add_north_arrow(ax_map3)
add_north_arrow(ax_map4)

# =============================================================================
# PANEL LABELS (RESTORED)
# =============================================================================

def panel_label(ax, label):
    ax.text(
        0.01, 0.99,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=1)
    )

panel_label(ax_map1, "(a)")
panel_label(ax_map4, "(c)")
panel_label(ax_map2, "(b)")
panel_label(ax_map3, "(d)")
panel_label(ax_speed, "(e)")
panel_label(ax_u, "(f)")
panel_label(ax_v, "(g)")
panel_label(ax_bias, "(h)")
panel_label(ax_std, "(i)")
panel_label(ax_dir, "(j)")

plt.show()


#%% zone 2:spatial


# ----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


fig, axs = plt.subplots(
    4, 3,
    figsize=(9, 11),
    dpi=300,
    sharex=True,
    sharey=True,
    width_ratios=[0.94,1,1]
)

# =====================================================
# COLUMN 0 = FILTERED (0)
# COLUMN 1 = UNFILTERED (1)
# COORDINATES CONVERTED TO lon/lat
# DATA TRANSPOSED (.T)
# =====================================================

# -----------------------------
# BIAS
# -----------------------------
im0 = axs[0, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    bias_map_all[1],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    bias_map_all[0],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 1].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    bias_map_all[2],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

axs[0, 0].set_title(fil_name[1], fontsize=14, fontweight="bold", pad=12)
axs[0, 1].set_title(fil_name[2], fontsize=14, fontweight="bold", pad=12)
axs[0, 2].set_title(fil_name[0], fontsize=14, fontweight="bold", pad=12)


fig.colorbar(im1, ax=axs[0, 2], fraction=0.046,label="Mean Residual, Bias [m/s]")

# -----------------------------
# STD
# -----------------------------
im2 = axs[1, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    std_map_all[1],
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    std_map_all[0],
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 1].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    std_map_all[2],
    cmap="Reds", vmin=0, vmax=0.25
)



fig.colorbar(im3, ax=axs[1, 2], fraction=0.046,label="Standard deviation [m/s]")

# -----------------------------
# ANGLE
# -----------------------------
im4 = axs[2, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    angle_map_all[1],
    cmap="Reds", vmin=0, vmax=180
)

im5 = axs[2, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    angle_map_all[0],
    cmap="Reds", vmin=0, vmax=180
)


im5 = axs[2, 1].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    angle_map_all[2],
    cmap="Reds", vmin=0, vmax=180
)


fig.colorbar(im5, ax=axs[2, 2], fraction=0.046,label=r'Direction misalignemnt [$^\circ$]')

# -----------------------------
# AGREEMENT
# -----------------------------
im6 = axs[3, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    sign_map_all[1],
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    sign_map_all[0],
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 1].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    sign_map_all[2],
    cmap="Reds", vmin=0, vmax=1
)


fig.colorbar(im7, ax=axs[3, 2], fraction=0.046,label='Sign agreement ratio [-]')

# -----------------------------
# AXES LABELS
# -----------------------------
#for ax in axs.ravel():
 
axs[0, 0].set_ylabel("Longitude")
axs[1, 0].set_ylabel("Longitude")
axs[2, 0].set_ylabel("Longitude")
axs[3, 0].set_ylabel("Longitude")

axs[3, 0].set_xlabel("Latitude")
axs[3, 1].set_xlabel("Latitude")
axs[3, 2].set_xlabel("Latitude")

axs[0, 0].set_facecolor("black")
axs[1, 0].set_facecolor("black")
axs[2, 0].set_facecolor("black")
axs[3, 0].set_facecolor("black")


axs[0, 1].set_facecolor("black")
axs[1, 1].set_facecolor("black")
axs[2, 1].set_facecolor("black")
axs[3, 1].set_facecolor("black")

axs[0, 2].set_facecolor("black")
axs[1, 2].set_facecolor("black")
axs[2, 2].set_facecolor("black")
axs[3, 2].set_facecolor("black")

labels = [
    "(a)", "(b)",
    "(c)", "(d)",
    "(e)", "(f)",
    "(g)", "(h)"
]

for ax, lab in zip(axs.ravel(), labels):
    ax.text(
        0.01, 0.99, lab,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=2)
    )



def add_north_arrow(ax, x=0.92, y=0.13, length=0.25, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(axs[0, 0])
add_north_arrow(axs[1, 0])
add_north_arrow(axs[2, 0])
add_north_arrow(axs[3, 0])


add_north_arrow(axs[0, 1])
add_north_arrow(axs[1, 1])
add_north_arrow(axs[2, 1])
add_north_arrow(axs[3, 1])

add_north_arrow(axs[0, 2])
add_north_arrow(axs[1, 2])
add_north_arrow(axs[2, 2])
add_north_arrow(axs[3, 2])

plt.tight_layout()
plt.show()

#%% PLOTS FOR GEOID

#%% zone 2: dist2coast

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


colors = {
    "dtured": (0.6, 0.0, 0.0),
    "blue": (0.1843, 0.2431, 0.9176),
    "brightgreen": (0.1216, 0.8157, 0.5098),
    "navyblue": (0.0118, 0.0588, 0.3098),
    "yellow": (0.9647, 0.8157, 0.3019),
    "orange": (0.9882, 0.4627, 0.2039),
    "pink": (0.9686, 0.7333, 0.6941),
    "grey": (0.8549, 0.8549, 0.8549),
    "red": (0.9098, 0.2471, 0.2824),
    "green": (0.0, 0.5333, 0.2078),
    "purple": (0.4745, 0.1373, 0.5569),
}


def format_lat(x, pos):
    return f"{abs(x):.0f}°{'N' if x >= 0 else 'S'}"

def format_lon(x, pos):
    return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"

mark = ['-', '--','-.']
c = [colors['purple'], colors['blue'], colors['dtured']]

lim = [0, 125]


# =============================================================================
# FIGURE SETUP
# =============================================================================

fig = plt.figure(figsize=(10, 12), dpi=300)

outer = fig.add_gridspec(
    2, 1,
    height_ratios=[4, 5],
    hspace=0.08,
    wspace=0.2
)

# =============================================================================
# TOP ROW (MAPS)
# =============================================================================

gs_top = outer[0].subgridspec(
    1, 4,
    wspace=0.1
)

file_idx = 1

# =============================================================================
# SWOT UNFILTERED
# =============================================================================

ax_map1 = fig.add_subplot(gs_top[0, 1])

im1 = ax_map1.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    swot_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map1.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map1.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map1.set_facecolor("black")
#ax_map1.set_xticklabels([])
ax_map1.set_yticklabels([])

ax_map1.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)




ax_map1.text(
    0.98, 0.01,
    fil_name[1],
    transform=ax_map1.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# SWOT FILTERED
# =============================================================================

ax_map2 = fig.add_subplot(gs_top[0, 2])

im2 = ax_map2.pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    swot_map_all[0],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map2.contour(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    dist_map_all[0],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map2.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map2.set_facecolor("black")
#ax_map2.set_xticklabels([])
ax_map2.set_yticklabels([])
ax_map2.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)



ax_map2.text(
    0.98, 0.01,
    fil_name[0],
    transform=ax_map2.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)


# =============================================================================
# DTU
# =============================================================================

ax_map4 = fig.add_subplot(gs_top[0, 0])

im3 = ax_map4.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    swot_map_all[2],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map4.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map4.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map4.set_facecolor("black")
#ax_map3.set_xticklabels([])
#ax_map4.set_yticklabels([])

ax_map4.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)

ax_map4.yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)

ax_map4.text(
    0.98, 0.01,
    fil_name[2],
    transform=ax_map4.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# NORKYST
# =============================================================================

ax_map3 = fig.add_subplot(gs_top[0, 3])

im3 = ax_map3.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    norkyst_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map3.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map3.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map3.set_facecolor("black")
#ax_map3.set_xticklabels([])
ax_map3.set_yticklabels([])

ax_map3.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)


ax_map3.text(
    0.98, 0.01,
    "Norkyst",
    transform=ax_map3.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# COLORBAR (safe, no layout distortion)
cax = ax_map3.inset_axes([1.02, 0.0, 0.04, 1.0])
cbar = fig.colorbar(im3, cax=cax)
cbar.set_label("Velocity magnitude [m/s]")

# =============================================================================
# BOTTOM BLOCK
# =============================================================================

gs_bot = outer[1].subgridspec(
    4, 2,
    hspace=0.2,
    wspace=0.3,
    height_ratios=[1, 0.1, 1, 1]
)

ax_speed = fig.add_subplot(gs_bot[0:2, 0])
ax_u     = fig.add_subplot(gs_bot[2, 0], sharex=ax_speed)
ax_v     = fig.add_subplot(gs_bot[3, 0], sharex=ax_speed)


ax_bias = fig.add_subplot(gs_bot[0:2, 1])
ax_std  = fig.add_subplot(gs_bot[2, 1], sharex=ax_bias)
ax_dir  = fig.add_subplot(gs_bot[3, 1], sharex=ax_bias)

# =============================================================================
# STYLING SETUP (RESTORED)
# =============================================================================

file_idx = 1


# =============================================================================
# NORKYST OVERLAY
# =============================================================================


ax_speed.plot(
    bin_centers_all[file_idx],
    norkyst_mean_all[file_idx],
    'k-',
    linewidth=2,
    label="Norkyst"
)


ax_u.plot(bin_centers_all[file_idx], norkyst_umean_all[file_idx], 'k-', linewidth=2)
ax_v.plot(bin_centers_all[file_idx], norkyst_vmean_all[file_idx], 'k-', linewidth=2)


# =============================================================================
# PLOTTING LOOP (RESTORED STYLE)
# =============================================================================

for i in range(n_files):

    ax_speed.plot(
        bin_centers_all[i],
        swot_mean_all[i],
        mark[i],
        color=c[i],
        label=fil_name[i]
    )

    ax_u.plot(bin_centers_all[i], swot_umean_all[i], mark[i], color=c[i])
    ax_v.plot(bin_centers_all[i], swot_vmean_all[i], mark[i], color=c[i])
   
    ax_bias.plot(bin_centers_bias_all[i], bias_mean_all[i], mark[i], color=c[i], label=fil_name[i])
    ax_std.plot(bin_centers_bias_all[i], bias_std_all[i], mark[i], color=c[i])
   
    ax_dir.plot(bin_centers_bias_all[i], mean_sign_mean[i], mark[i], color=c[i])

# =============================================================================
# FORMATTING
# =============================================================================

for ax in [ax_speed, ax_u, ax_v,
           ax_bias, ax_std, ax_dir]:
    ax.grid(True)
    ax.set_xlim(lim)

ax_speed.set_ylabel(r"$\overline{V_g}$ [m/s]")
ax_u.set_ylabel(r"$\overline{u_g}$ [m/s]")
ax_v.set_ylabel(r"$\overline{v_g}$ [m/s]")
ax_v.set_xlabel("Distance to coast [km]")

ax_bias.set_ylabel("Mean bias [m/s]")
ax_std.set_ylabel("Std [m/s]")
ax_dir.set_ylabel("Direction diff [°]")
ax_dir.set_xlabel("Distance to coast [km]")


ax_dir.set_ylim([0,180])
ax_std.set_ylim([0.05,0.5])
ax_u.set_ylim([-0.06,0.05])

for ax in [ax_speed, ax_u,
           ax_bias, ax_std]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_speed.legend(fontsize=8)
ax_bias.legend(fontsize=8)


def add_north_arrow(ax, x=0.92, y=0.3, length=0.15, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(ax_map1)
add_north_arrow(ax_map2)
add_north_arrow(ax_map3)
add_north_arrow(ax_map4)

# =============================================================================
# PANEL LABELS (RESTORED)
# =============================================================================

def panel_label(ax, label):
    ax.text(
        0.01, 0.99,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=1)
    )

panel_label(ax_map4, "(a)")
panel_label(ax_map1, "(b)")
panel_label(ax_map2, "(c)")
panel_label(ax_map3, "(d)")
panel_label(ax_speed, "(e)")
panel_label(ax_u, "(f)")
panel_label(ax_v, "(g)")
panel_label(ax_bias, "(h)")
panel_label(ax_std, "(i)")
panel_label(ax_dir, "(j)")

plt.show()


#%% zone 2:spatial


# ----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


fig, axs = plt.subplots(
    4, 3,
    figsize=(9, 11),
    dpi=300,
    sharex=True,
    sharey=True,
    width_ratios=[0.94,1,1]
)

# =====================================================
# COLUMN 0 = FILTERED (0)
# COLUMN 1 = UNFILTERED (1)
# COORDINATES CONVERTED TO lon/lat
# DATA TRANSPOSED (.T)
# =====================================================

# -----------------------------
# BIAS
# -----------------------------
im0 = axs[0, 0].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    bias_map_all[2],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    bias_map_all[0],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 1].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    bias_map_all[1],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

axs[0, 0].set_title(fil_name[2], fontsize=14, fontweight="bold", pad=12)
axs[0, 1].set_title(fil_name[1], fontsize=14, fontweight="bold", pad=12)
axs[0, 2].set_title(fil_name[0], fontsize=14, fontweight="bold", pad=12)


fig.colorbar(im1, ax=axs[0, 2], fraction=0.046,label="Mean Residual, Bias [m/s]")

# -----------------------------
# STD
# -----------------------------
im2 = axs[1, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    std_map_all[1],
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    std_map_all[0],
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    std_map_all[2],
    cmap="Reds", vmin=0, vmax=0.25
)



fig.colorbar(im3, ax=axs[1, 2], fraction=0.046,label="Standard deviation [m/s]")

# -----------------------------
# ANGLE
# -----------------------------
im4 = axs[2, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    angle_map_all[1],
    cmap="Reds", vmin=0, vmax=180
)

im5 = axs[2, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    angle_map_all[0],
    cmap="Reds", vmin=0, vmax=180
)


im5 = axs[2, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    angle_map_all[2],
    cmap="Reds", vmin=0, vmax=180
)


fig.colorbar(im5, ax=axs[2, 2], fraction=0.046,label=r'Direction misalignemnt [$^\circ$]')

# -----------------------------
# AGREEMENT
# -----------------------------
im6 = axs[3, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    sign_map_all[1],
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    sign_map_all[0],
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 1].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    sign_map_all[2],
    cmap="Reds", vmin=0, vmax=1
)


fig.colorbar(im7, ax=axs[3, 2], fraction=0.046,label='Sign agreement ratio [-]')

# -----------------------------
# AXES LABELS
# -----------------------------
#for ax in axs.ravel():
 
axs[0, 0].yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)
axs[1, 0].yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)
axs[2, 0].yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)
axs[3, 0].yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)

axs[3, 0].xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)
axs[3, 1].xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)
axs[3, 2].xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)

ax_map4.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)

ax_map4.yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)

axs[0, 0].set_facecolor("black")
axs[1, 0].set_facecolor("black")
axs[2, 0].set_facecolor("black")
axs[3, 0].set_facecolor("black")


axs[0, 1].set_facecolor("black")
axs[1, 1].set_facecolor("black")
axs[2, 1].set_facecolor("black")
axs[3, 1].set_facecolor("black")

axs[0, 2].set_facecolor("black")
axs[1, 2].set_facecolor("black")
axs[2, 2].set_facecolor("black")
axs[3, 2].set_facecolor("black")

labels = [
    "(a)", "(b)",
    "(c)", "(d)",
    "(e)", "(f)",
    "(g)", "(h)",
    "(i)","(j)",
    "(k)", "(l)"
]

for ax, lab in zip(axs.ravel(), labels):
    ax.text(
        0.01, 0.99, lab,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=2)
    )



def add_north_arrow(ax, x=0.92, y=0.13, length=0.25, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(axs[0, 0])
add_north_arrow(axs[1, 0])
add_north_arrow(axs[2, 0])
add_north_arrow(axs[3, 0])


add_north_arrow(axs[0, 1])
add_north_arrow(axs[1, 1])
add_north_arrow(axs[2, 1])
add_north_arrow(axs[3, 1])

add_north_arrow(axs[0, 2])
add_north_arrow(axs[1, 2])
add_north_arrow(axs[2, 2])
add_north_arrow(axs[3, 2])

plt.tight_layout()
plt.show()


#%% zone 1: dist2coast


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


colors = {
    "dtured": (0.6, 0.0, 0.0),
    "blue": (0.1843, 0.2431, 0.9176),
    "brightgreen": (0.1216, 0.8157, 0.5098),
    "navyblue": (0.0118, 0.0588, 0.3098),
    "yellow": (0.9647, 0.8157, 0.3019),
    "orange": (0.9882, 0.4627, 0.2039),
    "pink": (0.9686, 0.7333, 0.6941),
    "grey": (0.8549, 0.8549, 0.8549),
    "red": (0.9098, 0.2471, 0.2824),
    "green": (0.0, 0.5333, 0.2078),
    "purple": (0.4745, 0.1373, 0.5569),
}

mark = ['-', '--','-.']
c = [colors['purple'], colors['blue'], colors['dtured']]

lim = [0, 400]


# =============================================================================
# FIGURE SETUP
# =============================================================================

fig = plt.figure(figsize=(10, 12), dpi=300)

outer = fig.add_gridspec(
    2, 1,
    height_ratios=[4, 5],
    hspace=0.11
)

# =============================================================================
# TOP ROW (MAPS)
# =============================================================================

gs_top = outer[0].subgridspec(
    1, 4,
    wspace=0.1
)

file_idx = 2

# =============================================================================
# CNES
# =============================================================================

ax_map1 = fig.add_subplot(gs_top[0, 0])

im1 = ax_map1.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    swot_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map1.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map1.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map1.set_facecolor("black")

#ax_map1.set_xticklabels([])
#ax_map1.set_yticklabels([])

ax_map1.yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)

ax_map1.xaxis.set_major_formatter(
        mticker.FuncFormatter(format_lon)
    )

ax_map1.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)

ax_map1.text(
    0.98, 0.01,
    fil_name[2],
    transform=ax_map1.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# DTU
# =============================================================================

ax_map2 = fig.add_subplot(gs_top[0, 1])

im2 = ax_map2.pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    swot_map_all[1],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map2.contour(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    dist_map_all[1],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map2.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map2.set_facecolor("black")
#ax_map2.set_xticklabels([])
ax_map2.set_yticklabels([])
ax_map2.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)

ax_map2.text(
    0.98, 0.01,
    fil_name[1],
    transform=ax_map2.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)


# =============================================================================
# Norkyst
# =============================================================================

ax_map4 = fig.add_subplot(gs_top[0, 2])

im3 = ax_map4.pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    swot_map_all[0],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map4.contour(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    dist_map_all[0],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map4.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map4.set_facecolor("black")
#ax_map3.set_xticklabels([])
ax_map4.set_yticklabels([])

ax_map4.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)

ax_map4.text(
    0.98, 0.01,
    fil_name[0],
    transform=ax_map4.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# =============================================================================
# NORKYST
# =============================================================================

ax_map3 = fig.add_subplot(gs_top[0, 3])

im3 = ax_map3.pcolormesh(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    norkyst_map_all[file_idx],
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map3.contour(
    km_to_lon(x_centers_all[file_idx]),
    km_to_lat(y_centers_all[file_idx]),
    dist_map_all[file_idx],
    levels=[50, 100, 150, 200, 250, 300,350,400],
    colors='black',
    linewidths=2,
)

ax_map3.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map3.set_facecolor("black")

ax_map3.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)
#ax_map3.set_xticklabels([])
ax_map3.set_yticklabels([])

ax_map3.text(
    0.98, 0.01,
    "Norkyst",
    transform=ax_map3.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)

# COLORBAR (safe, no layout distortion)
cax = ax_map3.inset_axes([1.02, 0.0, 0.04, 1.0])
cbar = fig.colorbar(im3, cax=cax)
cbar.set_label("Velocity magnitude [m/s]")

# =============================================================================
# BOTTOM BLOCK
# =============================================================================

gs_bot = outer[1].subgridspec(
    4, 2,
    hspace=0.2,
    wspace=0.3,
    height_ratios=[1,0.2,1,1]
)

ax_speed = fig.add_subplot(gs_bot[0:2, 0])
ax_u     = fig.add_subplot(gs_bot[2, 0], sharex=ax_speed)
ax_v     = fig.add_subplot(gs_bot[3, 0], sharex=ax_speed)


ax_bias = fig.add_subplot(gs_bot[0:2, 1])
ax_std  = fig.add_subplot(gs_bot[2, 1], sharex=ax_bias)
ax_dir  = fig.add_subplot(gs_bot[3, 1], sharex=ax_bias)

# =============================================================================
# STYLING SETUP (RESTORED)
# =============================================================================

file_idx = 1

# =============================================================================
# PLOTTING LOOP (RESTORED STYLE)
# =============================================================================

for i in range(n_files):

    ax_speed.plot(
        bin_centers_all[i],
        swot_mean_all[i],
        mark[i],
        color=c[i],
        label=fil_name[i]
    )

    ax_u.plot(bin_centers_all[i], swot_umean_all[i], mark[i], color=c[i])
    ax_v.plot(bin_centers_all[i], swot_vmean_all[i], mark[i], color=c[i])
   
    ax_bias.plot(bin_centers_bias_all[i], bias_mean_all[i], mark[i], color=c[i], label=fil_name[i])
    ax_std.plot(bin_centers_bias_all[i], bias_std_all[i], mark[i], color=c[i])
   
    ax_dir.plot(bin_centers_bias_all[i], mean_sign_mean[i], mark[i], color=c[i])

# =============================================================================
# NORKYST OVERLAY
# =============================================================================

ax_speed.plot(
    bin_centers_all[file_idx],
    norkyst_mean_all[file_idx],
    'k-',
    linewidth=2,
    label="Norkyst"
)

ax_u.plot(bin_centers_all[file_idx], norkyst_umean_all[file_idx], 'k-', linewidth=2)
ax_v.plot(bin_centers_all[file_idx], norkyst_vmean_all[file_idx], 'k-', linewidth=2)

# =============================================================================
# FORMATTING
# =============================================================================

for ax in [ax_speed, ax_u, ax_v,
           ax_bias, ax_std, ax_dir]:
    ax.grid(True)
    ax.set_xlim(lim)

ax_speed.set_ylabel(r"$\overline{V_g}$ [m/s]")
ax_u.set_ylabel(r"$\overline{u_g}$ [m/s]")
ax_v.set_ylabel(r"$\overline{v_g}$ [m/s]")
ax_v.set_xlabel("Distance to coast [km]")

ax_bias.set_ylabel("Mean bias [m/s]")
ax_std.set_ylabel("Std [m/s]")
ax_dir.set_ylabel("Direction diff [°]")
ax_dir.set_xlabel("Distance to coast [km]")


ax_dir.set_ylim([0,180])
ax_std.set_ylim([0.1,0.26])
ax_u.set_ylim([-0.1,0.1])
ax_v.set_ylim([-0.1,0.2])

for ax in [ax_speed, ax_u,
           ax_bias, ax_std]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_speed.legend(fontsize=8)
ax_bias.legend(fontsize=8)


def add_north_arrow(ax, x=0.92, y=0.1, length=0.15, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(ax_map1)
add_north_arrow(ax_map2)
add_north_arrow(ax_map3)
add_north_arrow(ax_map4)

# =============================================================================
# PANEL LABELS (RESTORED)
# =============================================================================

def panel_label(ax, label):
    ax.text(
        0.01, 0.99,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=1)
    )

panel_label(ax_map1, "(a)")
panel_label(ax_map4, "(c)")
panel_label(ax_map2, "(b)")
panel_label(ax_map3, "(d)")
panel_label(ax_speed, "(e)")
panel_label(ax_u, "(f)")
panel_label(ax_v, "(g)")
panel_label(ax_bias, "(h)")
panel_label(ax_std, "(i)")
panel_label(ax_dir, "(j)")

plt.show()


#%% zone 2:spatial


# ----------------------------
# conversion constants
# -----------------------------
lat_ref = np.mean(lat0)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_ref))

def km_to_lon(x):
    return x / km_per_deg_lon + lon0

def km_to_lat(y):
    return y / km_per_deg_lat + lat0


fig, axs = plt.subplots(
    4, 3,
    figsize=(9, 11),
    dpi=300,
    sharex=True,
    sharey=True,
    width_ratios=[0.94,1,1]
)

# =====================================================
# COLUMN 0 = FILTERED (0)
# COLUMN 1 = UNFILTERED (1)
# COORDINATES CONVERTED TO lon/lat
# DATA TRANSPOSED (.T)
# =====================================================

# -----------------------------
# BIAS
# -----------------------------
im0 = axs[0, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    bias_map_all[1],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    bias_map_all[0],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

im1 = axs[0, 1].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    bias_map_all[2],
    cmap="RdBu_r", vmin=-0.25, vmax=0.25
)

axs[0, 0].set_title(fil_name[1], fontsize=14, fontweight="bold", pad=12)
axs[0, 1].set_title(fil_name[2], fontsize=14, fontweight="bold", pad=12)
axs[0, 2].set_title(fil_name[0], fontsize=14, fontweight="bold", pad=12)


fig.colorbar(im1, ax=axs[0, 2], fraction=0.046,label="Mean Residual, Bias [m/s]")

# -----------------------------
# STD
# -----------------------------
im2 = axs[1, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    std_map_all[1],
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    std_map_all[0],
    cmap="Reds", vmin=0, vmax=0.25
)

im3 = axs[1, 1].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    std_map_all[2],
    cmap="Reds", vmin=0, vmax=0.25
)



fig.colorbar(im3, ax=axs[1, 2], fraction=0.046,label="Standard deviation [m/s]")

# -----------------------------
# ANGLE
# -----------------------------
im4 = axs[2, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    angle_map_all[1],
    cmap="Reds", vmin=0, vmax=180
)

im5 = axs[2, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    angle_map_all[0],
    cmap="Reds", vmin=0, vmax=180
)


im5 = axs[2, 1].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    angle_map_all[2],
    cmap="Reds", vmin=0, vmax=180
)


fig.colorbar(im5, ax=axs[2, 2], fraction=0.046,label=r'Direction misalignemnt [$^\circ$]')

# -----------------------------
# AGREEMENT
# -----------------------------
im6 = axs[3, 0].pcolormesh(
    km_to_lon(x_centers_all[1]),
    km_to_lat(y_centers_all[1]),
    sign_map_all[1],
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 2].pcolormesh(
    km_to_lon(x_centers_all[0]),
    km_to_lat(y_centers_all[0]),
    sign_map_all[0],
    cmap="Reds", vmin=0, vmax=1
)

im7 = axs[3, 1].pcolormesh(
    km_to_lon(x_centers_all[2]),
    km_to_lat(y_centers_all[2]),
    sign_map_all[2],
    cmap="Reds", vmin=0, vmax=1
)


fig.colorbar(im7, ax=axs[3, 2], fraction=0.046,label='Sign agreement ratio [-]')

# -----------------------------
# AXES LABELS
# -----------------------------
#for ax in axs.ravel():
 

axs[0, 0].yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)
axs[1, 0].yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)
axs[2, 0].yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)
axs[3, 0].yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)

axs[3, 0].xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)
axs[3, 1].xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)
axs[3, 2].xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)

ax_map4.xaxis.set_major_formatter(
    mticker.FuncFormatter(format_lon)
)

ax_map4.yaxis.set_major_formatter(
    mticker.FuncFormatter(format_lat)
)

axs[0, 0].set_facecolor("black")
axs[1, 0].set_facecolor("black")
axs[2, 0].set_facecolor("black")
axs[3, 0].set_facecolor("black")


axs[0, 1].set_facecolor("black")
axs[1, 1].set_facecolor("black")
axs[2, 1].set_facecolor("black")
axs[3, 1].set_facecolor("black")

axs[0, 2].set_facecolor("black")
axs[1, 2].set_facecolor("black")
axs[2, 2].set_facecolor("black")
axs[3, 2].set_facecolor("black")

labels = [
    "(a)", "(b)",
    "(c)", "(d)",
    "(e)", "(f)",
    "(g)", "(h)"
]

for ax, lab in zip(axs.ravel(), labels):
    ax.text(
        0.01, 0.99, lab,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=1, pad=2)
    )



def add_north_arrow(ax, x=0.92, y=0.13, length=0.25, color='white'):
    ax.annotate(
        'N',
        xy=(x, y + length),   # arrow tip
        xytext=(x, y),        # arrow base
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        color=color,
        arrowprops=dict(
            arrowstyle='-|>',
            color=color,
            lw=2
        )
    )

add_north_arrow(axs[0, 0])
add_north_arrow(axs[1, 0])
add_north_arrow(axs[2, 0])
add_north_arrow(axs[3, 0])


add_north_arrow(axs[0, 1])
add_north_arrow(axs[1, 1])
add_north_arrow(axs[2, 1])
add_north_arrow(axs[3, 1])

add_north_arrow(axs[0, 2])
add_north_arrow(axs[1, 2])
add_north_arrow(axs[2, 2])
add_north_arrow(axs[3, 2])

plt.tight_layout()
plt.show()

