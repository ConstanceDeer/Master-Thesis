# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:27:08 2026

@author: const
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs



# Load random Norkyst file

T = "2023-01-01"

day = pd.Timestamp(T)

url = (
    f"https://thredds.met.no/thredds/dodsC/"
    f"romshindcast/norkyst_v3/zdepth/"
    f"{day:%Y}/{day:%m}/norkyst800-{day:%Y%m%d}.nc"
)

print(url)

ds = nc.Dataset(url)

# get inital mask form zeta (ocean mask)

lat = ds.variables["lat"][:].filled(np.nan)
lon = ds.variables["lon"][:].filled(np.nan)

zeta = ds.variables["zeta"][0,:,:]

ocean = (~np.ma.getmaskarray(zeta)).astype(float)



# Interpolate onto regular grid of size 0.01x0.01 deg

ddeg = 0.01

lat_reg = np.arange(
    np.nanmin(lat),
    np.nanmax(lat),
    ddeg
)

lon_reg = np.arange(
    np.nanmin(lon),
    np.nanmax(lon),
    ddeg
)

lon_grid, lat_grid = np.meshgrid(
    lon_reg,
    lat_reg
)


ocean_grid = griddata(
    (
        lon.ravel(),
        lat.ravel()
    ),
    ocean.ravel(),
    (
        lon_grid,
        lat_grid
    ),
    method="linear"
)



# Convert to landmask 
#RIGHT NOW: 1 = ocean, 0 = land, NaN = outside Norkyst domain
#AFTER: 0 = ocean, 1 = land

landmask = np.where(
    ocean_grid >= 0.5,
    0,
    1
).astype(np.uint8)



# smooth
from scipy import ndimage

landmask0=landmask
# Convert to boolean
land = landmask.astype(bool)



# Fill narrow fjords and bays


land = ndimage.binary_closing(
    land,
    structure=np.ones((10,10))
)



# Remove small isolated islands


land = ndimage.binary_opening(
    land,
    structure=np.ones((5,5))
)


# Fill enclosed water bodies


land = ndimage.binary_fill_holes(
    land
)



landmask = land.astype(np.uint8)


# Save

da = xr.DataArray(
    landmask,
    dims=("lat","lon"),
    coords={
        "lat": lat_reg,
        "lon": lon_reg
    },
    name="landmask"
)


da.attrs["description"] = (
    "Norkyst zeta-derived land mask on regular lat/lon grid"
)

da.attrs["values"] = "0=ocean, 1=land"

#%
da.to_netcdf(
    "./data/norkyst_landmask_regular_clean.nc"
)

#% Plot test 
# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

#Ocean=0, land=1
fig = plt.figure(figsize=(8, 8))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)


ax.pcolormesh(
    lon_grid,
    lat_grid,
    landmask,
    cmap="Reds",
    shading="auto",
    vmin=0,
    vmax=1,
    transform=ccrs.PlateCarree()
)



ax.set_extent(
    [1, 14, 57, 69],
    crs=ccrs.PlateCarree()
)

plt.title("Norkyst land mask")
plt.show()

#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}, dpi=200
)

# Original interpolated landmask
ax = axes[0]

ax.pcolormesh(
    lon_grid,
    lat_grid,
    landmask0,
    cmap="Greys",
    shading="auto",
    transform=ccrs.PlateCarree()
)

ax.set_extent(
    [1, 14, 57, 69],
    crs=ccrs.PlateCarree()
)

ax.text(
    0.99,
    0.01,
    "(a) Before filtering",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=12,
    fontweight="bold",
    bbox=dict(
        facecolor="black",
        edgecolor="none",
        alpha=0.8,
        pad=2
    )
)

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="black", edgecolor="black", label="Land"),
    Patch(facecolor="white", edgecolor="black", label="Ocean")
]

ax.legend(
    handles=legend_elements,
    loc="lower left"
)


# Simplified landmask
ax = axes[1]

ax.pcolormesh(
    lon_grid,
    lat_grid,
    landmask,
    cmap="Greys",
    shading="auto",
    transform=ccrs.PlateCarree()
)

ax.set_extent(
    [1, 14, 57, 69],
    crs=ccrs.PlateCarree()
)




ax.text(
    0.99,
    0.01,
    "(b) After filtering",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    color="white",
    fontsize=12,
    fontweight="bold",
    bbox=dict(
        facecolor="black",
        edgecolor="none",
        alpha=0.8,
        pad=2
    )
)


legend_elements = [
    Patch(facecolor="black", edgecolor="black", label="Land"),
    Patch(facecolor="white", edgecolor="black", label="Ocean")
]

ax.legend(
    handles=legend_elements,
    loc="lower left"
)

plt.tight_layout()
plt.show()
