# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:27:21 2026

@author: const
"""

#%% Visual 

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# -----------------------
# Load data
# -----------------------

model="fullNO_NK_dt300_coarse"
data = np.load(
    f"./Github/QGBFN/comparison/{model}_swot_NK_QG_day.npz",
    allow_pickle=True
)

lon = data["lon"]
lat = data["lat"]

swot = data["swot"]
nk = data["nk_int"]
qg = data["qg_int"]


swot_u = data["swot_ugos"]
swot_v = data["swot_vgos"]

nk_u = data["nk_ugos_int"]
nk_v = data["nk_vgos_int"]

qg_u = data["qg_ugos_int"]
qg_v = data["qg_vgos_int"]

# Compute speed for each pass


time = data["time"]


start = datetime.strptime("2025-07-20", "%Y-%m-%d")
end = datetime.strptime("2025-07-27", "%Y-%m-%d")

mask = (time >= start) & (time <= end)

lon = lon[mask]
lat = lat[mask]

swot = swot[mask]
nk = nk[mask]+ 0.2-0.049
qg = qg[mask]

swot_u = swot_u[mask]
swot_v = swot_v[mask]

nk_u = nk_u[mask]
nk_v = nk_v[mask]

qg_u = qg_u[mask]
qg_v = qg_v[mask]


time = time[mask]


# -----------------------
# Figure
# -----------------------

vmin = -0.5
vmax = 0

fig, axs = plt.subplots(
    1, 3,
    figsize=(8,7),
    dpi=200,
    subplot_kw={"projection": ccrs.PlateCarree()}
)


titles = ["SWOT", "NorKyst", "QG"]
fields = [swot, nk, qg]


lonax=0
for ax, field, title in zip(axs, fields, titles):

    ax.set_extent([2, 13.9, 57.5, 68.5], ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="black")
    ax.coastlines("10m")

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    if lonax==0:
        gl.left_labels = True
        lonax=1
    else:
        gl.left_labels = False
        

    mappable = None

    for lo, la, spd in zip(lon, lat, field):

        mappable = ax.scatter(
            lo.ravel(),
            la.ravel(),
            c=spd.ravel(),
            s=1,
            cmap="Blues_r",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree()
        )

    ax.set_aspect(1/np.cos(np.deg2rad(62)))
    ax.text(
        0.03,0.985,title,
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
    

fig.colorbar(
    mappable,
    ax=axs,
    orientation="horizontal",
    pad=0.05,
    aspect=40,
    label="ADT [m]"
)


plt.show()

# -----------------------
# Common color limits
# -----------------------
vmin = -1
vmax = 1


swot_speed = [np.sqrt(u**2 + v**2) for u, v in zip(swot_u, swot_v)]
nk_speed   = [np.sqrt(u**2 + v**2) for u, v in zip(nk_u, nk_v)]
qg_speed   = [np.sqrt(u**2 + v**2) for u, v in zip(qg_u, qg_v)]


fields = [swot_speed, nk_speed, qg_speed]
titles = ["SWOT", "NorKyst", "QG"]

vmin = 0
vmax = 1     # adjust if needed

fig, axs = plt.subplots(
    1, 3,
    figsize=(8,7),
    dpi=200,
    subplot_kw={"projection": ccrs.PlateCarree()}
)

lonax=0
for ax, field, title in zip(axs, fields, titles):

    ax.set_extent([2, 13.9, 57.5, 68.5], ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="black")
    ax.coastlines("10m")

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    if lonax==0:
        gl.left_labels = True
        lonax=1
    else:
        gl.left_labels = False
        

    mappable = None

    for lo, la, spd in zip(lon, lat, field):

        mappable = ax.scatter(
            lo.ravel(),
            la.ravel(),
            c=spd.ravel(),
            s=1,
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree()
        )

    ax.set_aspect(1/np.cos(np.deg2rad(62)))
    ax.text(
        0.03,0.985,title,
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
    

fig.colorbar(
    mappable,
    ax=axs,
    orientation="horizontal",
    pad=0.05,
    aspect=40,
    label="Surface geostrophic speed [m/s]"
)

plt.show()

#%% load data


# ============================================================
# LOAD SINGLE SWOT-NK-QG FILE AND DISTANCE STATISTICS
# ============================================================

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ------------------------------------------------------------
# Load comparison file
# ------------------------------------------------------------

file = "./Github/QGBFN/comparison/fullNO_NK_dt300_coarse_V2_swot_NK_QG_day.npz"

data = np.load(file, allow_pickle=True)


swot = data["swot"]
nk = data["nk_int"]
qg = data["qg_int"]

lon = data["lon"]
lat = data["lat"]

u_s = data["swot_ugos"]
v_s = data["swot_vgos"]

u_n = data["nk_ugos_int"]
v_n = data["nk_vgos_int"]

u_q = data["qg_ugos_int"]
v_q = data["qg_vgos_int"]


# convert object arrays
swot = list(swot)
nk = list(nk)
qg = list(qg)

lon = list(lon)
lat = list(lat)

u_s = list(u_s)
v_s = list(v_s)

u_n = list(u_n)
v_n = list(v_n)

u_q = list(u_q)
v_q = list(v_q)



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



#%
# ------------------------------------------------------------
# Speed
# ------------------------------------------------------------

speed_s = [
    np.sqrt(u**2 + v**2)
    for u,v in zip(u_s,v_s)
]

speed_n = [
    np.sqrt(u**2 + v**2)
    for u,v in zip(u_n,v_n)
]

speed_q = [
    np.sqrt(u**2 + v**2)
    for u,v in zip(u_q,v_q)
]



# ------------------------------------------------------------
# Distance to coast
# ------------------------------------------------------------

ds = xr.open_dataset(
    "./data/dist2coast/dist2coast.nc"
)

dist2coast = ds["dist2coast"]

lon_d2c = ds["lon"].values
lat_d2c = ds["lat"].values


lon2d,lat2d = np.meshgrid(
    lon_d2c,
    lat_d2c
)

d2c_points = np.column_stack(
    (
        lon2d.ravel(),
        lat2d.ravel()
    )
)

d2c_values = dist2coast.values.ravel()



dist_swot=[]


for lo,la in zip(lon,lat):

    lo=np.asarray(lo)
    la=np.asarray(la)

    shape=lo.shape

    mask = (
        np.isfinite(lo) &
        np.isfinite(la)
    )


    points=np.column_stack(
        (
            lo[mask],
            la[mask]
        )
    )


    dist=griddata(
        d2c_points,
        d2c_values,
        points,
        method="linear"
    )


    out=np.full(lo.size,np.nan)

    out[mask.ravel()] = dist

    dist_swot.append(
        out.reshape(shape)
    )



# ------------------------------------------------------------
# Flatten all passes
# ------------------------------------------------------------

lon_all=np.concatenate(
    [np.asarray(x).ravel() for x in lon]
)

lat_all=np.concatenate(
    [np.asarray(x).ravel() for x in lat]
)


dist_all=np.concatenate(
    [np.asarray(x).ravel() for x in dist_swot]
)


# fields

speed_s_all=np.concatenate(
    [np.asarray(x).ravel() for x in speed_s]
)

speed_n_all=np.concatenate(
    [np.asarray(x).ravel() for x in speed_n]
)

speed_q_all=np.concatenate(
    [np.asarray(x).ravel() for x in speed_q]
)


u_s_all=np.concatenate(
    [np.asarray(x).ravel() for x in u_s]
)

v_s_all=np.concatenate(
    [np.asarray(x).ravel() for x in v_s]
)

u_n_all=np.concatenate(
    [np.asarray(x).ravel() for x in u_n]
)

v_n_all=np.concatenate(
    [np.asarray(x).ravel() for x in v_n]
)

u_q_all=np.concatenate(
    [np.asarray(x).ravel() for x in u_q]
)

v_q_all=np.concatenate(
    [np.asarray(x).ravel() for x in v_q]
)



# ------------------------------------------------------------
# AREA MASK
# same as your old code
# ------------------------------------------------------------

# zone 1

lon_min,lon_max = 3,12
lat_min,lat_max = 62,69


# zone 2
lon_min,lon_max = 3,6
lat_min,lat_max = 58,63

# full NO
lon_min,lon_max = 2,13.9
lat_min,lat_max = 57.5,68.5



mask_area=(

    (lon_all>=lon_min) &
    (lon_all<=lon_max) &

    (lat_all>=lat_min) &
    (lat_all<=lat_max) &


    np.isfinite(speed_s_all) &
    np.isfinite(speed_n_all) &
    np.isfinite(speed_q_all) &


    (speed_s_all<100) &
    (speed_n_all<100) &
    (speed_q_all<100)

)



# apply mask

lon_sel = lon_all[mask_area]
lat_sel = lat_all[mask_area]

dist_sel = dist_all[mask_area]


speed_s_sel=speed_s_all[mask_area]
speed_n_sel=speed_n_all[mask_area]
speed_q_sel=speed_q_all[mask_area]


u_s_sel=u_s_all[mask_area]
v_s_sel=v_s_all[mask_area]

u_n_sel=u_n_all[mask_area]
v_n_sel=v_n_all[mask_area]

u_q_sel=u_q_all[mask_area]
v_q_sel=v_q_all[mask_area]


print("points:",len(dist_sel))

#%% dist2coast (Like old analysis)
#% calc all means 
# =====================================================
# Statistics: SWOT vs NK and SWOT vs QG
# =====================================================

# ---------- containers ----------

# Distance binning
# ============================

bin_size = 5   # km  <-- ONLY INPUT
stats = {
    "NK": {},
    "QG": {}
}


# =====================================================
# Function for comparison statistics
# =====================================================

def velocity_stats(
        u_ref, v_ref, speed_ref,
        u_mod, v_mod, speed_mod,
        bin_idx,
        n_bins,
        min_points=50):


    # -----------------------------
    # Speed statistics
    # -----------------------------

    bias = speed_ref - speed_mod


    bias_mean = np.array([
        np.nanmean(bias[bin_idx == i])
        for i in range(1, n_bins)
    ])


    bias_std = np.array([
        np.nanstd(bias[bin_idx == i])
        for i in range(1, n_bins)
    ])


    bias_rms = np.array([
        np.sqrt(np.nanmean(bias[bin_idx == i]**2))
        for i in range(1, n_bins)
    ])



    # -----------------------------
    # Mean vector direction
    # -----------------------------

    u_ref_mean = np.array([
        np.nanmean(u_ref[bin_idx == i])
        for i in range(1, n_bins)
    ])

    v_ref_mean = np.array([
        np.nanmean(v_ref[bin_idx == i])
        for i in range(1, n_bins)
    ])


    u_mod_mean = np.array([
        np.nanmean(u_mod[bin_idx == i])
        for i in range(1, n_bins)
    ])

    v_mod_mean = np.array([
        np.nanmean(v_mod[bin_idx == i])
        for i in range(1, n_bins)
    ])



    theta_ref_mean = np.arctan2(
        v_ref_mean,
        u_ref_mean
    )

    theta_mod_mean = np.arctan2(
        v_mod_mean,
        u_mod_mean
    )


    mean_vec_diff = np.arctan2(
        np.sin(theta_ref_mean - theta_mod_mean),
        np.cos(theta_ref_mean - theta_mod_mean)
    )


    mean_ang_d = np.degrees(
        np.abs(mean_vec_diff)
    )



    # -----------------------------
    # Individual vector angle diff
    # -----------------------------

    theta_ref = np.arctan2(
        v_ref,
        u_ref
    )


    theta_mod = np.arctan2(
        v_mod,
        u_mod
    )


    ang_diff = np.arctan2(
        np.sin(theta_ref-theta_mod),
        np.cos(theta_ref-theta_mod)
    )


    mean_ang_diff_deg = np.array([
        np.nanmean(
            np.degrees(
                np.abs(ang_diff[bin_idx == i])
            )
        )
        for i in range(1, n_bins)
    ])



    # -----------------------------
    # Sign agreement
    # -----------------------------
    
    counts = np.array([
        np.sum(bin_idx == b)
        for b in range(1, len(bins))
    ])

    agree_count = np.array([

        np.sum(
            (np.sign(u_ref[bin_idx == i]) ==
             np.sign(u_mod[bin_idx == i]))
            &
            (np.sign(v_ref[bin_idx == i]) ==
             np.sign(v_mod[bin_idx == i]))
        )

        for i in range(1,n_bins)

    ])


    agreement_fraction = (
        agree_count / counts
    )



    # -----------------------------
    # Remove low statistics bins
    # -----------------------------

    mask = counts < min_points


    for arr in [
        bias_mean,
        bias_std,
        bias_rms,
        mean_ang_d,
        mean_ang_diff_deg,
        agreement_fraction
    ]:
        arr[mask] = np.nan



    return {
        "bias_mean": bias_mean,
        "bias_std": bias_std,
        "bias_rms": bias_rms,

        "mean_vector_angle": mean_ang_d,
        "individual_angle": mean_ang_diff_deg,

        "agreement": agreement_fraction,

        "u_ref_mean": u_ref_mean,
        "v_ref_mean": v_ref_mean,
        "u_mod_mean": u_mod_mean,
        "v_mod_mean": v_mod_mean,

        "counts": counts
    }


# ============================

bins = np.arange(
    0,
    np.nanmax(dist_sel) + bin_size,
    bin_size
)

bin_centers = (
    bins[:-1] + bins[1:]
) / 2

bin_idx = np.digitize(
    dist_sel,
    bins
)

n_bins = len(bins)


# =====================================================
# Calculate SWOT-NK
# =====================================================

stats["NK"] = velocity_stats(
    u_s_sel,
    v_s_sel,
    speed_s_sel,

    u_n_sel,
    v_n_sel,
    speed_n_sel,

    bin_idx,
    n_bins
)



# =====================================================
# Calculate SWOT-QG
# =====================================================

stats["QG"] = velocity_stats(
    u_s_sel,
    v_s_sel,
    speed_s_sel,

    u_q_sel,
    v_q_sel,
    speed_q_sel,

    bin_idx,
    n_bins
)

# Example access:

bias_NK = stats["NK"]["bias_mean"]
bias_QG = stats["QG"]["bias_mean"]

angle_NK = stats["NK"]["individual_angle"]
angle_QG = stats["QG"]["individual_angle"]

agreement_NK = stats["NK"]["agreement"]
agreement_QG = stats["QG"]["agreement"]

#%% heat map
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

bin_size = 5  # km

lon0 = lon_min
lat0 = lat_min


# ============================================================
# INPUT DATA
# ============================================================

lat = lat_sel
lon = lon_sel

# velocities
u_s = u_s_sel
v_s = v_s_sel

u_n = u_n_sel
v_n = v_n_sel

u_q = u_q_sel
v_q = v_q_sel

# speed
speed_s = speed_s_sel
speed_n = speed_n_sel
speed_q = speed_q_sel

dist = dist_sel


# ============================================================
# CONVERT TO KM COORDINATES
# ============================================================

lat_ref = np.nanmean(lat)

x = (lon - lon0) * 111.32 * np.cos(np.radians(lat_ref))
y = (lat - lat0) * 111.32


# ============================================================
# CREATE 5 KM GRID
# ============================================================

x_bins = np.arange(
    np.nanmin(x),
    np.nanmax(x) + bin_size,
    bin_size
)

y_bins = np.arange(
    np.nanmin(y),
    np.nanmax(y) + bin_size,
    bin_size
)


x_centers = 0.5*(x_bins[:-1] + x_bins[1:])
y_centers = 0.5*(y_bins[:-1] + y_bins[1:])


shape = (len(y_centers), len(x_centers))


# ============================================================
# STORAGE
# ============================================================

swot_map = np.full(shape,np.nan)
nk_map   = np.full(shape,np.nan)
qg_map   = np.full(shape,np.nan)


u_swot_map = np.full(shape,np.nan)
v_swot_map = np.full(shape,np.nan)

u_nk_map = np.full(shape,np.nan)
v_nk_map = np.full(shape,np.nan)

u_qg_map = np.full(shape,np.nan)
v_qg_map = np.full(shape,np.nan)


# SWOT-NK
bias_nk_map = np.full(shape,np.nan)
std_nk_map  = np.full(shape,np.nan)
rms_nk_map  = np.full(shape,np.nan)
sign_nk_map = np.full(shape,np.nan)
angle_nk_map = np.full(shape,np.nan)
corr_nk_map = np.full(shape, np.nan)


# SWOT-QG
bias_qg_map = np.full(shape,np.nan)
std_qg_map  = np.full(shape,np.nan)
rms_qg_map  = np.full(shape,np.nan)
sign_qg_map = np.full(shape,np.nan)
angle_qg_map = np.full(shape,np.nan)
corr_qg_map = np.full(shape, np.nan)

dist_map = np.full(shape,np.nan)

count_map = np.zeros(shape)


# ============================================================
# 2D BINNING
# ============================================================

for i in range(len(y_bins)-1):

    for j in range(len(x_bins)-1):

        mask = (
            (x >= x_bins[j]) &
            (x < x_bins[j+1]) &
            (y >= y_bins[i]) &
            (y < y_bins[i+1])
        )


        n = np.sum(mask)

        # minimum number of observations
        if n < 3:
            continue


        count_map[i,j] = n


        # ----------------------------------------------------
        # Mean speeds
        # ----------------------------------------------------

        swot_map[i,j] = np.nanmean(speed_s[mask])
        nk_map[i,j]   = np.nanmean(speed_n[mask])
        qg_map[i,j]   = np.nanmean(speed_q[mask])
        
        u_swot_map[i,j] = np.nanmean(u_s[mask])
        v_swot_map[i,j] = np.nanmean(v_s[mask])
        
        u_nk_map[i,j] = np.nanmean(u_n[mask])
        v_nk_map[i,j] = np.nanmean(v_n[mask])
        
        u_qg_map[i,j] = np.nanmean(u_q[mask])
        v_qg_map[i,j] = np.nanmean(v_q[mask])

        # ----------------------------------------------------
        # Distance
        # ----------------------------------------------------

        dist_map[i,j] = np.nanmean(dist[mask])


        # ====================================================
        # SWOT - NK
        # ====================================================

        bias = speed_s[mask] - speed_n[mask]

        bias_nk_map[i,j] = np.nanmean(bias)
        std_nk_map[i,j]  = np.nanstd(bias)
        rms_nk_map[i,j]  = np.sqrt(np.nanmean(bias**2))


        # sign agreement

        cond = (
            (np.sign(u_s[mask]) == np.sign(u_n[mask])) &
            (np.sign(v_s[mask]) == np.sign(v_n[mask]))
        )

        sign_nk_map[i,j] = np.mean(cond)


        # angle difference of mean vectors

        us = np.nanmean(u_s[mask])
        vs = np.nanmean(v_s[mask])

        un = np.nanmean(u_n[mask])
        vn = np.nanmean(v_n[mask])


        theta_s = np.arctan2(vs,us)
        theta_n = np.arctan2(vn,un)


        dtheta = np.arctan2(
            np.sin(theta_s-theta_n),
            np.cos(theta_s-theta_n)
        )

        angle_nk_map[i,j] = np.degrees(np.abs(dtheta))
        
        
        
        corr_nk_map[i, j] = np.corrcoef(
                speed_s[mask],
                speed_n[mask]
            )[0, 1]



        # ====================================================
        # SWOT - QG
        # ====================================================

        bias = speed_s[mask] - speed_q[mask]

        bias_qg_map[i,j] = np.nanmean(bias)
        std_qg_map[i,j]  = np.nanstd(bias)
        rms_qg_map[i,j]  = np.sqrt(np.nanmean(bias**2))


        cond = (
            (np.sign(u_s[mask]) == np.sign(u_q[mask])) &
            (np.sign(v_s[mask]) == np.sign(v_q[mask]))
        )

        sign_qg_map[i,j] = np.mean(cond)


        uq = np.nanmean(u_q[mask])
        vq = np.nanmean(v_q[mask])


        theta_q = np.arctan2(vq,uq)


        dtheta = np.arctan2(
            np.sin(theta_s-theta_q),
            np.cos(theta_s-theta_q)
        )

        angle_qg_map[i,j] = np.degrees(np.abs(dtheta))
        
        corr_qg_map[i, j] = np.corrcoef(
                speed_s[mask],
                speed_q[mask]
            )[0, 1]


# ============================================================
# REMOVE LOW COUNT BINS
# ============================================================

min_points = 3

bad = count_map < min_points


maps = [
    swot_map,nk_map,qg_map,
    u_swot_map,v_swot_map,
    u_nk_map,v_nk_map,
    u_qg_map,v_qg_map,
    bias_nk_map,std_nk_map,rms_nk_map,
    sign_nk_map,angle_nk_map,corr_nk_map,
    bias_qg_map,std_qg_map,rms_qg_map,
    sign_qg_map,angle_qg_map,corr_qg_map,
    dist_map
]


for m in maps:
    m[bad] = np.nan
    
#%% dist2coast plot zone 1

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.ticker as mticker

def format_lat(x, pos):
    return f"{abs(x):.0f}°{'N' if x >= 0 else 'S'}"

def format_lon(x, pos):
    return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"

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
    1, 3,
    wspace=0.1
)

file_idx = 2

# =============================================================================
# SWOT
# =============================================================================

ax_map1 = fig.add_subplot(gs_top[0, 0])

im1 = ax_map1.pcolormesh(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    swot_map,
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map1.contour(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    dist_map,
    levels=[50, 100, 150, 200, 250, 300, 350, 400],
    colors='black',
    linewidths=2,
)

ax_map1.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map1.set_facecolor("black")
ax_map1.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
ax_map1.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
#ax_map1.set_xticklabels([])
#ax_map1.set_yticklabels([])

ax_map1.text(
    0.98, 0.01,
    'SWOT',
    transform=ax_map1.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)
# =============================================================================
# QG
# =============================================================================

ax_map2 = fig.add_subplot(gs_top[0, 1])

im2 = ax_map2.pcolormesh(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    qg_map,
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map2.contour(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    dist_map,
    levels=[50, 100, 150, 200, 250, 300, 350, 400],
    colors='black',
    linewidths=2,
)

ax_map2.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map2.set_facecolor("black")
#ax_map2.set_xticklabels([])
ax_map2.set_yticklabels([])
ax_map2.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
ax_map2.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))

ax_map2.text(
    0.98, 0.01,
    'QG-model',
    transform=ax_map2.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)


# =============================================================================
# NorKyst
# =============================================================================

ax_map4 = fig.add_subplot(gs_top[0, 2])

im3 = ax_map4.pcolormesh(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    nk_map,
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map4.contour(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    dist_map,
    levels=[50, 100, 150, 200, 250, 300, 350, 400],
    colors='black',
    linewidths=2,
)
ax_map4.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map4.set_facecolor("black")
#ax_map3.set_xticklabels([])
ax_map4.set_yticklabels([])
ax_map4.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
ax_map4.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))

ax_map4.text(
    0.98, 0.01,
    'Norkyst',
    transform=ax_map4.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)


# =============================================================================
# BOTTOM BLOCK
# =============================================================================

gs_bot = outer[1].subgridspec(
    4, 2,
    hspace=0.1,
    wspace=0.2,
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


# speed

ax_speed.plot(
    bin_centers,
    np.sqrt(stats["QG"]["u_mod_mean"]**2+stats["QG"]["v_mod_mean"]**2),
    '-',
    color=c[1],
    label="QG-model",

)

ax_speed.plot(
    bin_centers,
    np.sqrt(stats["NK"]["u_mod_mean"]**2+stats["NK"]["v_mod_mean"]**2),
    '-',
    color='black',
    label="Norkyst",
  
)

ax_speed.plot(
    bin_centers,
    np.sqrt(stats["NK"]["u_ref_mean"]**2+stats["NK"]["v_ref_mean"]**2),
    '-',
    color=c[2],
    label="SWOT",
   
)

# zonal


ax_u.plot(
    bin_centers,
    stats["QG"]["u_mod_mean"],
    '-',
    color=c[1],
    label="QG-model",

)
ax_u.plot(
    bin_centers,
    stats["NK"]["u_mod_mean"],
    '-',
    color='black',
    label="Norkyst",

)

ax_u.plot(
    bin_centers,
    stats["NK"]["u_ref_mean"],
    '-',
    color=c[2],

)

# meridonal


ax_v.plot(
    bin_centers,
    stats["QG"]["v_mod_mean"],
    '-',
    color=c[1],
    label="QG-model",

)
ax_v.plot(
    bin_centers,
    stats["NK"]["v_mod_mean"],
    '-',
    color='black',
    label="Norkyst",
 
)

ax_v.plot(
    bin_centers,
    stats["NK"]["v_ref_mean"],
    '-',
    color=c[2],

)

# =============================================================================
# STATISTICS: SWOT - NorKyst and SWOT - QG
# =============================================================================

ax_bias.plot(
    bin_centers,
    stats["NK"]["bias_mean"],
    '--',
    color=c[0],
    label="SWOT-Norkyst"
)

ax_bias.plot(
    bin_centers,
    stats["QG"]["bias_mean"],
    '-.',
    color=c[1],
    label="SWOT-QG"
)


ax_std.plot(
    bin_centers,
    stats["NK"]["bias_std"],
    '-.',
    color=c[0],
    label="SWOT-Norkyst"
)

ax_std.plot(
    bin_centers,
    stats["QG"]["bias_std"],
    '--',
    color=c[1],
    label="SWOT-QG"
)

ax_dir.plot(
    bin_centers,
    stats["NK"]["mean_vector_angle"],
    '--',
    color=c[0],
    label="SWOT-Norkyst"
)
ax_dir.plot(
    bin_centers,
    stats["QG"]["mean_vector_angle"],
    '--',
    color=c[1],
    label="SWOT-QG"
)
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
ax_std.set_ylim([0,0.26])
ax_u.set_ylim([-0.1,0.05])
ax_v.set_ylim([-0.1,0.15])

for ax in [ax_speed, ax_u,
           ax_bias, ax_std]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_speed.legend(fontsize=8,loc='upper right')
ax_bias.legend(fontsize=8,loc='upper right')


add_north_arrow(ax_map1)
add_north_arrow(ax_map2)
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
panel_label(ax_speed, "(d)")
panel_label(ax_u, "(e)")
panel_label(ax_v, "(f)")
panel_label(ax_bias, "(g)")
panel_label(ax_std, "(h)")
panel_label(ax_dir, "(i)")

plt.show()

#%% dist2coast plot zone 2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.ticker as mticker

def format_lat(x, pos):
    return f"{abs(x):.0f}°{'N' if x >= 0 else 'S'}"

def format_lon(x, pos):
    return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"

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

lim =  [0, 125]


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

file_idx = 2

# =============================================================================
# SWOT
# =============================================================================

ax_map1 = fig.add_subplot(gs_top[0, 0])

im1 = ax_map1.pcolormesh(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    swot_map,
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map1.contour(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    dist_map,
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map1.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map1.set_facecolor("black")
ax_map1.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
ax_map1.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
#ax_map1.set_xticklabels([])
#ax_map1.set_yticklabels([])

ax_map1.text(
    0.98, 0.01,
    'SWOT',
    transform=ax_map1.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)
# =============================================================================
# QG
# =============================================================================

ax_map2 = fig.add_subplot(gs_top[0, 1])

im2 = ax_map2.pcolormesh(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    qg_map,
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map2.contour(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    dist_map,
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)

ax_map2.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map2.set_facecolor("black")
#ax_map2.set_xticklabels([])
ax_map2.set_yticklabels([])
ax_map2.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))

ax_map2.text(
    0.98, 0.01,
    'QG-model',
    transform=ax_map2.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)


# =============================================================================
# NorKyst
# =============================================================================

ax_map4 = fig.add_subplot(gs_top[0, 2])

im3 = ax_map4.pcolormesh(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    nk_map,
    shading="auto",
    cmap="Reds",
    vmin=0,
    vmax=0.5,
)

cs = ax_map4.contour(
    km_to_lon(x_centers),
    km_to_lat(y_centers),
    dist_map,
    levels=[20, 40, 60, 80,100, 120],
    colors='black',
    linewidths=2,
)
ax_map4.clabel(cs, inline=True, fontsize=9, fmt='%d km')
ax_map4.set_facecolor("black")
#ax_map3.set_xticklabels([])
ax_map4.set_yticklabels([])
ax_map4.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))

ax_map4.text(
    0.98, 0.01,
    'Norkyst',
    transform=ax_map4.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=11,
    fontweight="bold",
    bbox=dict(facecolor="black", edgecolor="none", alpha=0.85, pad=2)
)


# =============================================================================
# BOTTOM BLOCK
# =============================================================================

gs_bot = outer[1].subgridspec(
    4, 2,
    hspace=0.1,
    wspace=0.2,
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


# speed

ax_speed.plot(
    bin_centers,
    np.sqrt(stats["QG"]["u_mod_mean"]**2+stats["QG"]["v_mod_mean"]**2),
    '-',
    color=c[1],
    label="QG-model",

)

ax_speed.plot(
    bin_centers,
    np.sqrt(stats["NK"]["u_mod_mean"]**2+stats["NK"]["v_mod_mean"]**2),
    '-',
    color='black',
    label="Norkyst",
  
)

ax_speed.plot(
    bin_centers,
    np.sqrt(stats["NK"]["u_ref_mean"]**2+stats["NK"]["v_ref_mean"]**2),
    '-',
    color=c[2],
    label="SWOT",
   
)

# zonal


ax_u.plot(
    bin_centers,
    stats["QG"]["u_mod_mean"],
    '-',
    color=c[1],
    label="QG-model",

)
ax_u.plot(
    bin_centers,
    stats["NK"]["u_mod_mean"],
    '-',
    color='black',
    label="Norkyst",

)

ax_u.plot(
    bin_centers,
    stats["NK"]["u_ref_mean"],
    '-',
    color=c[2],

)

# meridonal


ax_v.plot(
    bin_centers,
    stats["QG"]["v_mod_mean"],
    '-',
    color=c[1],
    label="QG-model",

)
ax_v.plot(
    bin_centers,
    stats["NK"]["v_mod_mean"],
    '-',
    color='black',
    label="Norkyst",
 
)

ax_v.plot(
    bin_centers,
    stats["NK"]["v_ref_mean"],
    '-',
    color=c[2],

)

# =============================================================================
# STATISTICS: SWOT - NorKyst and SWOT - QG
# =============================================================================

ax_bias.plot(
    bin_centers,
    stats["NK"]["bias_mean"],
    '--',
    color=c[0],
    label="SWOT-Norkyst"
)

ax_bias.plot(
    bin_centers,
    stats["QG"]["bias_mean"],
    '-.',
    color=c[1],
    label="SWOT-QG"
)


ax_std.plot(
    bin_centers,
    stats["NK"]["bias_std"],
    '-.',
    color=c[0],
    label="SWOT-Norkyst"
)

ax_std.plot(
    bin_centers,
    stats["QG"]["bias_std"],
    '--',
    color=c[1],
    label="SWOT-QG"
)

ax_dir.plot(
    bin_centers,
    stats["NK"]["mean_vector_angle"],
    '--',
    color=c[0],
    label="SWOT-Norkyst"
)
ax_dir.plot(
    bin_centers,
    stats["QG"]["mean_vector_angle"],
    '--',
    color=c[1],
    label="SWOT-QG"
)
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

ax_bias.set_ylabel("Bias [m/s]")
ax_std.set_ylabel("Std [m/s]")
ax_dir.set_ylabel("Direction diff [°]")
ax_dir.set_xlabel("Distance to coast [km]")

ax_bias.set_ylim([-0.23,0.2])
ax_dir.set_ylim([0,180])
ax_std.set_ylim([0,0.26])
ax_u.set_ylim([-0.05,0.07])
ax_v.set_ylim([-0.05,0.40])

for ax in [ax_speed, ax_u,
           ax_bias, ax_std]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_speed.legend(fontsize=8,loc='upper right')
ax_bias.legend(fontsize=8,loc='upper right')


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
add_north_arrow(ax_map4)

# =============================================================================
# PANEL LABELS (RESTORED)
# =============================================================================

def panel_label(ax, label):
    ax.text(
        0.01, 0.995,
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
panel_label(ax_speed, "(d)")
panel_label(ax_u, "(e)")
panel_label(ax_v, "(f)")
panel_label(ax_bias, "(g)")
panel_label(ax_std, "(h)")
panel_label(ax_dir, "(i)")

plt.show()

#%% Correlation matrix:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Create profiles
# ============================================================

speed_profiles = pd.DataFrame({

    "SWOT":
        np.sqrt(
            stats["NK"]["u_ref_mean"]**2 +
            stats["NK"]["v_ref_mean"]**2
        ),

    "NorKyst":
        np.sqrt(
            stats["NK"]["u_mod_mean"]**2 +
            stats["NK"]["v_mod_mean"]**2
        ),

    "QG":
        np.sqrt(
            stats["QG"]["u_mod_mean"]**2 +
            stats["QG"]["v_mod_mean"]**2
        )

})


u_profiles = pd.DataFrame({

    "SWOT":
        stats["NK"]["u_ref_mean"],

    "NorKyst":
        stats["NK"]["u_mod_mean"],

    "QG":
        stats["QG"]["u_mod_mean"]

})


v_profiles = pd.DataFrame({

    "SWOT":
        stats["NK"]["v_ref_mean"],

    "NorKyst":
        stats["NK"]["v_mod_mean"],

    "QG":
        stats["QG"]["v_mod_mean"]

})


# ============================================================
# Function
# ============================================================

def calculate_corr(data):

    data = data.dropna()

    return data.corr()



corr_speed = calculate_corr(speed_profiles)
corr_u = calculate_corr(u_profiles)
corr_v = calculate_corr(v_profiles)



print("Speed correlation")
print(corr_speed)

print("\nU correlation")
print(corr_u)

print("\nV correlation")
print(corr_v)



# ============================================================
# Plot all
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12,4),
    dpi=300
)


corrs = [
    corr_speed,
    corr_u,
    corr_v
]


titles = [
    "(a) Speed",
    "(b) Zonal",
    "(c) Meridional"
]

lol=0
for ax, corr, title in zip(
        axes,
        corrs,
        titles):


    im = ax.imshow(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )


    labels = corr.columns


    ax.set_xticks(
        np.arange(len(labels))
    )

    ax.set_yticks(
        np.arange(len(labels))
    )


    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    
    if lol==0:
        ax.set_yticklabels(labels)
        lol=1


    for i in range(len(labels)):
        for j in range(len(labels)):

            ax.text(
                j,
                i,
                f"{corr.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=11
            )


    ax.text(
    0.5, -0.2, title,
    transform=ax.transAxes,
    ha='center',
    va='top'
    )
    


# common colorbar

cbar = fig.colorbar(
    im,
    ax=axes,
    shrink=0.75,
    pad=0.01,
    aspect=20,
)

cbar.set_label(
    "Correlation [-]"
)

plt.show()

#%% Heat maps with stats 


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


fig, axes = plt.subplots(
    4,
    3,
    figsize=(12,16),
    dpi=300
)

plt.subplots_adjust(
    wspace=0.05,
    hspace=0.05
)


# ============================================================
# Data
# ============================================================

plot_data = [

    # Speed
    swot_map,
    nk_map,
    qg_map,

    # Speed differences
    swot_map - nk_map,
    swot_map - qg_map,
    nk_map - qg_map,

    # Direction differences
    angle_nk_map,
    angle_qg_map,
    angle_nk_map - angle_qg_map,

    # RMS
    rms_nk_map,
    rms_qg_map,
    rms_nk_map - rms_qg_map

]


titles = [

    # Speed
    "SWOT speed",
    "NorKyst speed",
    "QG-model speed",

    # Speed differences
    "SWOT - NorKyst speed",
    "SWOT - QG speed",
    "NorKyst - QG speed",

    # Direction
    "SWOT - NorKyst direction",
    "SWOT - QG direction",
    "Direction difference change",

    # RMS
    "SWOT - NorKyst RMS",
    "SWOT - QG RMS",
    "RMS difference"

]


cmaps = [

    # Speed
    "Reds",
    "Reds",
    "Reds",

    # Speed difference
    "RdBu_r",
    "RdBu_r",
    "RdBu_r",

    # Direction
    "RdBu_r",
    "RdBu_r",
    "RdBu_r",

    # RMS
    "viridis",
    "viridis",
    "viridis"

]


vmins = [

    # Speed
    0,0,0,

    # Speed difference
    -0.25,-0.25,-0.25,

    # Direction
    -180,-180,-180,

    # RMS
    -0.3,-0.3,-0.3

]


vmaxs = [

    # Speed
    0.5,0.5,0.5,

    # Speed difference
    0.25,0.25,0.25,

    # Direction
    180,180,180,

    # RMS
    0.3,0.3,0.3

]


# ============================================================
# Plot
# ============================================================

for ax, data, title, cmap, vmin, vmax in zip(
    axes.flat,
    plot_data,
    titles,
    cmaps,
    vmins,
    vmaxs
):

    im = ax.pcolormesh(
        km_to_lon(x_centers),
        km_to_lat(y_centers),
        data,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )


    ax.set_facecolor("black")


    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(format_lat)
    )

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(format_lon)
    )


    ax.text(
        0.98,
        0.02,
        title,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            facecolor="black",
            edgecolor="none",
            alpha=0.85,
            pad=2
        )
    )


# ============================================================
# Remove repeated labels
# ============================================================

for ax in axes[:,1:].flat:
    ax.set_yticklabels([])

for ax in axes[:-1,:].flat:
    ax.set_xticklabels([])


# North arrows
for ax in axes[0,:]:
    add_north_arrow(ax)


# ============================================================
# Colorbars
# ============================================================

fig.colorbar(
    axes[0,0].collections[0],
    ax=axes[0,:],
    shrink=1,
    pad=0.01,
    aspect=25,
    label="Speed [m/s]"
)

fig.colorbar(
    axes[1,0].collections[0],
    ax=axes[1,:],
    shrink=1,
    pad=0.01,
    aspect=25,
    label="Speed difference [m/s]"
)

fig.colorbar(
    axes[2,0].collections[0],
    ax=axes[2,:],
    shrink=1,
    pad=0.01,
    aspect=25,
    label="Direction difference [deg]"
)

fig.colorbar(
    axes[3,0].collections[0],
    ax=axes[3,:],
    shrink=1,
    pad=0.01,
    aspect=25,
    label="RMS [m/s]"
)


plt.show()

#%% heat map with differences

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ============================================================
# Calculate mean speed and direction from u,v components
# ============================================================

# Speed magnitude
speed_swot = np.sqrt(
    u_swot_map**2 + v_swot_map**2
)

speed_nk = np.sqrt(
    u_nk_map**2 + v_nk_map**2
)

speed_qg = np.sqrt(
    u_qg_map**2 + v_qg_map**2
)


# Direction (degrees)
# atan2 gives correct quadrant
direction_swot = np.degrees(
    np.arctan2(v_swot_map, u_swot_map)
)

direction_nk = np.degrees(
    np.arctan2(v_nk_map, u_nk_map)
)

direction_qg = np.degrees(
    np.arctan2(v_qg_map, u_qg_map)
)


# Optional: convert from -180,180 to 0,360
direction_swot = (direction_swot + 360) % 360
direction_nk = (direction_nk + 360) % 360
direction_qg = (direction_qg + 360) % 360

#%

# ============================================================
# Improvement metrics
# ============================================================

# Positive = QG has lower RMS error
rms_improvement = (
    rms_qg_map - rms_nk_map
)


# Positive = QG has smaller directional error
direction_improvement = (
    np.abs(angle_qg_map) -
    np.abs(angle_nk_map)
)


# Positive = QG correlates better
correlation_improvement = (
    corr_qg_map -
    corr_nk_map
)



# ============================================================
# Data
# ============================================================

maps = [

    # Row 1 - Mean speed
    speed_swot,
    speed_nk,
    speed_qg,


    # Row 2 - Mean direction
    direction_swot,
    direction_nk,
    direction_qg,


    # Row 3 - QG improvement
    rms_improvement,
    direction_improvement,
    correlation_improvement,

]


titles = [

    # Speed
    "SWOT speed",
    "NorKyst speed",
    "QG-model speed",


    # Direction
    "SWOT direction",
    "NorKyst direction",
    "QG-model direction",


    # Improvement
    "RMS difference\n(QG-model - Norkyst)",
    "Direction difference\n(QG-model - Norkyst)",
    "Correlation difference\n(QG-model - NorKyst)",

]


cmaps = [

    # Speed
    "Reds",
    "Reds",
    "Reds",

    # Direction
    "twilight",
    "twilight",
    "twilight",

    # Improvement
    "RdBu_r",
    "RdBu_r",
    "RdBu_r",

]


vmins = [

    # Speed
    0,
    0,
    0,

    # Direction
    0,
    0,
    0,

    # Improvement
    -0.1,
    -30,
    -1,

]


vmaxs = [

    # Speed
    0.5,
    0.5,
    0.5,

    # Direction
    360,
    360,
    360,

    # Improvement
    0.1,
    30,
    1,

]



# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(10,10),
    dpi=300,
    constrained_layout=True
)


plt.subplots_adjust(
    wspace=0.05,
    hspace=0.05
)



# ============================================================
# Plot
# ============================================================

for ax, data, title, cmap, vmin, vmax in zip(
    axes.flat,
    maps,
    titles,
    cmaps,
    vmins,
    vmaxs
):

    im = ax.pcolormesh(
        km_to_lon(x_centers),
        km_to_lat(y_centers),
        data,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )


    ax.set_facecolor("black")


    ax.text(
        0.97,
        0.03,
        title,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            facecolor="black",
            edgecolor="none",
            alpha=0.8,
            pad=2
        )
    )


    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(format_lon)
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(format_lat)
    )



# ============================================================
# Remove repeated labels
# ============================================================

for ax in axes[:,1:].flat:
    ax.set_yticklabels([])


for ax in axes[:-1,:].flat:
    ax.set_xticklabels([])



# ============================================================
# North arrows
# ============================================================

for ax in axes[0,:]:
    add_north_arrow(ax)



# ============================================================
# Colorbars
# ============================================================

# Speed
fig.colorbar(
    axes[0,0].collections[0],
    ax=axes[0,:],
    shrink=1,
    pad=0.01,
    aspect=25,
    label="Speed [m/s]"
)

cbar = fig.colorbar(
    axes[1,0].collections[0],
    ax=axes[1,:],
    shrink=1,
    pad=0.01,
    aspect=25,
    label="Current direction [towards]"
)

cbar.set_ticks([0, 90, 180, 270, 360])
cbar.set_ticklabels([
    "E",
    "N",
    "W",
    "S",
    "E"
])

# ============================================================
# Improvement colorbars (one per panel, bottom)
# ============================================================

fig.colorbar(
    axes[2,0].collections[0],
    ax=axes[2,0],
    orientation="horizontal",
    shrink=0.9,
    pad=0.05,
    aspect=30,
    label="RMS difference [m/s]"
)


fig.colorbar(
    axes[2,1].collections[0],
    ax=axes[2,1],
    orientation="horizontal",
    shrink=0.9,
    pad=0.05,
    aspect=30,
    label="Direction difference [$^\circ$]"
)


fig.colorbar(
    axes[2,2].collections[0],
    ax=axes[2,2],
    orientation="horizontal",
    shrink=0.9,
    pad=0.05,
    aspect=30,
    label="Correlation difference [-]"
)


plt.show()

#%% Heat map only speed and direction


# ============================================================
# Figure (speed + direction only)
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(9,10),
    dpi=300,
    constrained_layout=True
)


# ============================================================
# Plot
# ============================================================

for ax, data, title, cmap, vmin, vmax in zip(
    axes.flat,
    maps[:6],       # only speed + direction
    titles[:6],
    cmaps[:6],
    vmins[:6],
    vmaxs[:6]
):

    im = ax.pcolormesh(
        km_to_lon(x_centers),
        km_to_lat(y_centers),
        data,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    #ax.set_aspect(1/np.cos(62*np.pi/180))


    ax.set_facecolor("black")


    ax.text(
        0.99,
        0.01,
        title,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            facecolor="black",
            edgecolor="none",
            alpha=0.8,
            pad=2
        )
    )


    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(format_lon)
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(format_lat)
    )


# ============================================================
# Remove repeated labels
# ============================================================

for ax in axes[:,1:].flat:
    ax.set_yticklabels([])

for ax in axes[:-1,:].flat:
    ax.set_xticklabels([])


# ============================================================
# North arrows
# ============================================================

for ax in axes[0,:]:
    add_north_arrow(ax)


# ============================================================
# Colorbars
# ============================================================

# Speed
fig.colorbar(
    axes[0,0].collections[0],
    ax=axes[0,:],
    shrink=1,
    pad=0.02,
    aspect=25,
    label="Speed [m/s]"
)


# Direction
cbar = fig.colorbar(
    axes[1,0].collections[0],
    ax=axes[1,:],
    shrink=1,
    pad=0.02,
    aspect=25,
    label="Current direction [towards]"
)

cbar.set_ticks([0, 90, 180, 270, 360])
cbar.set_ticklabels([
    "E",
    "N",
    "W",
    "S",
    "E"
])


plt.show()


#%% Only speed heat map

# ============================================================
# Figure (speed only)
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(8, 5),
    dpi=300,
    constrained_layout=True
)


# ============================================================
# Plot
# ============================================================

for ax, data, title, cmap, vmin, vmax in zip(
    axes.flat,
    maps[:3],       # only speed maps
    titles[:3],
    cmaps[:3],
    vmins[:3],
    vmaxs[:3]
):

    im = ax.pcolormesh(
        km_to_lon(x_centers),
        km_to_lat(y_centers),
        data,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    

    ax.set_facecolor("black")
    

    ax.text(
        0.99,
        0.01,
        title,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            facecolor="black",
            edgecolor="none",
            alpha=0.8,
            pad=2
        )
    )

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(format_lon)
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(format_lat)
    )
    ax.set_aspect(1/np.cos(62*np.pi/180))

# ============================================================
# Remove repeated labels
# ============================================================

for ax in axes[1:]:
    ax.set_yticklabels([])


# ============================================================
# North arrows
# ============================================================

for ax in axes:
    add_north_arrow(ax)


# ============================================================
# Speed colorbar
# ============================================================

fig.colorbar(
    axes[0].collections[0],
    ax=axes,
    shrink=1,
    pad=0.02,
    aspect=25,
    label="Mean speed [m/s]"
)


plt.show()