"""
Memory-efficient MP4 animation from monthly NetCDF outputs
3-panel Cartopy plots: ADT, Geostrophic Current, Relative Vorticity
Author: Adapted for HPC streaming
"""
#%%

import os
import glob
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature
import imageio.v2 as imageio
import gc                                  # <-- new import


# -----------------------
# User settings
# -----------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "outputs", "fullNO_NK_dt300_fine_V23")  # <-- update this path as needed
video_path = os.path.join(base_dir, "plot", "fullNO_NK_dt300_coarse_V23.mp4")
#model_dir = os.path.join(base_dir, "outputs", "06-08_018x037_-1_8_58_65_K_0_5")
#video_path = os.path.join(base_dir, "plots", "animation_6_8_2025_K_05_V2.mp4")
os.makedirs(os.path.dirname(video_path), exist_ok=True)

# -----------------------

# Domain and constants

# -----------------------
lon_min, lon_max = 2, 13.9
lat_min, lat_max = 57.5, 68.5
g = 9.82
Omega = 7.292e-5
dlat = 0.018*2
extent = [lon_min, lon_max, lat_min, lat_max]

proj = ccrs.Orthographic(
    central_longitude=(lon_min+lon_max)/2,
    central_latitude=(lat_min+lat_max)/2
)

# -----------------------
# MP4 writer (append mode)
# -----------------------
fps = 10
writer = imageio.get_writer(video_path, fps=fps, mode='I')
print("Writing animation to:", video_path)

# -----------------------
# Find all model files
# -----------------------
nc_files = sorted(glob.glob(os.path.join(model_dir, "*.nc")))
model_times = [datetime.strptime(f[-20:-6],'y%Ym%md%dh%H') for f in nc_files]

# -----------------------
# Define months to process
# -----------------------
months = [(2025,6),(2025,7),(2025,8)]  

# ------------------------------------------------------------------
# prepare a single figure and axes; reuse them inside the loop
# ------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(11, 6), dpi=160,
    subplot_kw={'projection': proj}
)
for ax in (ax1, ax2, ax3):
    ax.coastlines(resolution='10m')
    ax.add_feature(cartopy.feature.LAND, color='0.92')
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor('0.4')
    gl = ax.gridlines(draw_labels=True, dms=False,
                      x_inline=False, y_inline=False,
                      alpha=0.5, linestyle="--")
    gl.right_labels = False

# placeholders for the pcolormesh objects
sc1 = sc2 = sc3 = None

# -----------------------
# Process each month separately
# -----------------------
for year, month in months:
    start = datetime(year, month, 1, 0)
    if month == 12:
        end = datetime(year+1,1,1,0)
    else:
        end = datetime(year, month+1,1,0)
    
    td = end - start
    totalhours = int(td.total_seconds() / 3600)
    dt_array = [start + timedelta(hours=3*i) for i in range(int(totalhours/3))]

    # -----------------------
    # Process each time step
    # -----------------------
    for dt in dt_array:

        # Find closest model file
        model_idx = np.argmin(abs(np.array(model_times) - dt))
        if dt != model_times[model_idx]:
            print("No model file for", dt)
            continue
        print("Processing", dt)

        # -----------------------
        # Load only the subsection of the file we need
        # -----------------------
        with nc.Dataset(nc_files[model_idx]) as ds:
            lon_full = ds.variables['lon'][:]
            lat_full = ds.variables['lat'][:]
            i1 = np.searchsorted(lon_full, lon_min, 'left')
            i2 = np.searchsorted(lon_full, lon_max, 'right')
            j1 = np.searchsorted(lat_full, lat_min, 'left')
            j2 = np.searchsorted(lat_full, lat_max, 'right')
            ssh = ds.variables['ssh'][0, j1:j2, i1:i2].filled(np.nan)

        # prepare coordinate arrays
        lon = lon_full[i1:i2]
        lat = lat_full[j1:j2]
        lon, lat = np.meshgrid(lon, lat)
        dlon = abs(lon[0,0] - lon[0,1])

        # -----------------------
        # Geostrophic currents & vorticity
        # -----------------------
        grad_x = np.gradient(ssh, axis=1) / (
            dlon * (111.320 * np.cos(lat * np.pi / 180)))
        grad_y = np.gradient(ssh, axis=0) / (dlat * 110.574)
        f = 2 * Omega * np.sin(np.deg2rad(lat))
        gcurr_u = -g / f * grad_y / 1000
        gcurr_v = g / f * grad_x / 1000
        gcurr_mag = np.hypot(gcurr_u, gcurr_v)
        grad_v_x = np.gradient(gcurr_v, axis=1) / (
            dlon * (111.320 * np.cos(lat * np.pi / 180)) * 1000)
        grad_u_y = np.gradient(gcurr_u, axis=0) / (dlat * 110.574 * 1000)
        rel_vort = (grad_v_x - grad_u_y) / f

        # -----------------------
        # update/create pcolormesh objects
        # -----------------------
        if sc1 is None:
            sc1 = ax1.pcolormesh(lon, lat, ssh, vmin=-0.4, vmax=0.4,
                                 cmap='RdYlBu_r',
                                 transform=ccrs.PlateCarree(),
                                 rasterized=True)
            sc2 = ax2.pcolormesh(lon, lat, gcurr_mag, vmin=0, vmax=0.55,
                                 cmap='turbo',
                                 transform=ccrs.PlateCarree(),
                                 rasterized=True)
            sc3 = ax3.pcolormesh(lon, lat, rel_vort, vmin=-0.35, vmax=0.35,
                                 cmap='RdBu_r',
                                 transform=ccrs.PlateCarree(),
                                 rasterized=True)
            cb1 = fig.colorbar(sc1, ax=ax1, orientation='horizontal', pad=0.05)
            cb1.set_label("ADT [m]")
            cb2 = fig.colorbar(sc2, ax=ax2, orientation='horizontal', pad=0.05)
            cb2.set_label("Current Speed [m/s]")
            cb3 = fig.colorbar(sc3, ax=ax3, orientation='horizontal', pad=0.05)
            cb3.set_label("Relative Vorticity / f")
        else:
            sc1.set_array(ssh.ravel())
            sc2.set_array(gcurr_mag.ravel())
            sc3.set_array(rel_vort.ravel())

        ax2.set_title(dt.strftime('%d %b %Y %H:00'), fontsize=16)

        # -----------------------
        # Capture frame from canvas
        # -----------------------
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        writer.append_data(buf.reshape(h, w, 4)[:, :, :3])

        # free per‑timestep arrays immediately
        for arr in (ssh, gcurr_u, gcurr_v, gcurr_mag,
                    grad_x, grad_y, grad_v_x, grad_u_y, rel_vort,
                    lon, lat):
            del arr
        gc.collect()

# -----------------------
# Finish MP4
# -----------------------
writer.close()
print("Animation finished successfully!")