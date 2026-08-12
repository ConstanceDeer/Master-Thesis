import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
print("numpy:", np.__version__)
print("rasterio:", rasterio.__version__)

#%% Tif - files XGM

with rasterio.open("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/raw/XGM2019e.tiff") as src:
    geoid = src.read(1)
    transform = src.transform
    height, width = geoid.shape

rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
lon, lat = rasterio.transform.xy(transform, rows, cols)


lon = np.array(lon).reshape(height, width)
lat = np.array(lat).reshape(height, width)

lon_min, lon_max = 1, 14
lat_min, lat_max = 56, 69

proj = ccrs.Orthographic(
    central_longitude=(lon_min+lon_max)/2,
    central_latitude=(lat_min+lat_max)/2
)


fig = plt.figure(figsize=(20,15),dpi=300)
ax = fig.add_subplot(2,1,1, projection=proj)

ax.coastlines(resolution='10m')
gl1 = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,alpha=0.5,linestyle="--")
gl1.top_labels = False
gl1.right_labels = False
ax.set_extent([1, 14, 56, 69], crs=ccrs.PlateCarree())

sc = ax.pcolormesh(
    lon,
    lat[:],
    geoid[:,:].squeeze(),
    vmin=36,
    vmax=52,
    cmap='RdBu_r',      # your colormap
    shading='auto',
    transform=ccrs.PlateCarree()
)

plt.colorbar(sc,shrink=1,label='ssha [m]')
plt.title('Geoid: EGG2015')

plt.show()
#fig.savefig(os.path.join('./data/Duacs/figures/',dir_list[i][0:-3]+'.png'),dpi=100)
#plt.close() 

geoid_sub=geoid
lat_sub = lat[:, 0]
lon_sub = lon[0, :]


#%% lOAD matlab file EGG

from scipy.io import loadmat
import xarray as xr
mat = loadmat("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/raw/egg2015.mat")

egg = mat["egg2015"]

lat = egg["bi"][0, 0].squeeze()
lon = egg["li"][0, 0].squeeze()
geoid = egg["zeta"][0, 0]

print(lat.shape)
print(lon.shape)
print(geoid.shape)

lon_min, lon_max = 1, 14
lat_min, lat_max = 56, 69

lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]

lat_sub = lat[lat_idx]
lon_sub = lon[lon_idx]

geoid_sub = geoid[np.ix_(lat_idx, lon_idx)]


ds = xr.Dataset(
    data_vars={
        "geoid": (
            ("latitude", "longitude"),
            geoid_sub,
        )
    },
    coords={
        "latitude": lat_sub,
        "longitude": lon_sub,
    },
)

ds.geoid.attrs["units"] = "m"
ds.geoid.attrs["long_name"] = "Geoid height"

ds.latitude.attrs["units"] = "degrees_north"
ds.longitude.attrs["units"] = "degrees_east"

ds.to_netcdf("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/final/EGG2015.nc")
#%% txt-files


import numpy as np
import xarray as xr

# path to your file
file_path = "C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/raw/NKG2015.txt"


with open(file_path, "r") as f:
    for i in range(10):
        print(f.readline().strip())
 
# read the file
with open(file_path, "r") as f:
    lines = f.readlines()

# find header lines
header_start = 0
header_end = 0
for i, line in enumerate(lines):
    if "begin_of_head" in line:
        header_start = i
    if "end_of_head" in line:
        header_end = i
        break

# parse header info
header = lines[header_start+1:header_end]
header_dict = {}
for line in header:
    if ":" in line:
        key, val = line.split(":", 1)
        header_dict[key.strip()] = val.strip()
    elif "=" in line:
        key, val = line.split("=", 1)
        header_dict[key.strip()] = val.strip()

# extract needed info
lat_min = float(header_dict["lat min"])
lat_max = float(header_dict["lat max"])
lon_min = float(header_dict["lon min"])
lon_max = float(header_dict["lon max"])
delta_lat = float(header_dict["delta lat"])
delta_lon = float(header_dict["delta lon"])
nrows = int(header_dict["nrows"])
ncols = int(header_dict["ncols"])
nodata = float(header_dict["nodata"])

# remaining lines after header
data_lines = lines[header_end+1:]

# flatten all numbers into a single list
data = []
for line in data_lines:
    data.extend([float(x) for x in line.split()])

# reshape into 2D array
geoid = np.array(data).reshape((nrows, ncols))

lat = np.linspace(lat_max, lat_min, nrows)  # top-to-bottom
lon = np.linspace(lon_min, lon_max, ncols)  # left-to-right

# replace nodata values with NaN
geoid[geoid == nodata] = np.nan

lon_min, lon_max = 1, 14
lat_min, lat_max = 56, 69

lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]

lat_sub = lat[lat_idx]
lon_sub = lon[lon_idx]

geoid_sub = geoid[np.ix_(lat_idx, lon_idx)]

# make 2D grids
lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)

proj = ccrs.Orthographic(
    central_longitude=(lon_min+lon_max)/2,
    central_latitude=(lat_min+lat_max)/2
)


fig = plt.figure(figsize=(20,15), dpi=300)
ax = fig.add_subplot(2,1,1, projection=proj)

ax.coastlines(resolution="10m")

gl = ax.gridlines(draw_labels=True,
                  x_inline=False,
                  y_inline=False,
                  linestyle="--",
                  alpha=0.5)

gl.top_labels = False
gl.right_labels = False

ax.set_extent([1, 14, 56, 69], crs=ccrs.PlateCarree())

sc = ax.pcolormesh(
    lon2d,
    lat2d,
    geoid_sub,
    vmin=36,
    vmax=52,
    cmap="RdBu_r",
    shading="auto",
    transform=ccrs.PlateCarree()
)

plt.colorbar(sc, shrink=0.8, label="Geoid height [m]")
plt.title("Geoid: NKG2015")

plt.show()

#%% .dat file NKG

import pandas as pd
import numpy as np
import xarray as xr

# path to your file
file_path = "C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/raw/NKG2015.dat"

# Load xyz table
data = np.loadtxt(file_path)

lat = data[:, 0]
lon = data[:, 1]
geoid = data[:, 2]

print(data.shape)

lon_min, lon_max = 1, 14
lat_min, lat_max = 56, 69

mask = (
    (lat >= lat_min) &
    (lat <= lat_max) &
    (lon >= lon_min) &
    (lon <= lon_max)
)

lat_sub = lat[mask]
lon_sub = lon[mask]
geoid_sub = geoid[mask]

lat_unique = np.unique(lat_sub)
lon_unique = np.unique(lon_sub)

geoid_grid = geoid_sub.reshape(
    len(lat_unique),
    len(lon_unique)
)

df = pd.DataFrame({
    "latitude": lat_sub,
    "longitude": lon_sub,
    "geoid": geoid_sub
})

grid = df.pivot(
    index="latitude",
    columns="longitude",
    values="geoid"
)

geoid_sub = grid.values

lat_sub = grid.index.values
lon_sub = grid.columns.values


# make 2D grids
lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)

proj = ccrs.Orthographic(
    central_longitude=(lon_min+lon_max)/2,
    central_latitude=(lat_min+lat_max)/2
)


fig = plt.figure(figsize=(20,15), dpi=300)
ax = fig.add_subplot(2,1,1, projection=proj)

ax.coastlines(resolution="10m")

gl = ax.gridlines(draw_labels=True,
                  x_inline=False,
                  y_inline=False,
                  linestyle="--",
                  alpha=0.5)

gl.top_labels = False
gl.right_labels = False

ax.set_extent([1, 14, 56, 69], crs=ccrs.PlateCarree())

sc = ax.pcolormesh(
    lon2d,
    lat2d,
    geoid_sub,
    vmin=36,
    vmax=52,
    cmap="RdBu_r",
    shading="auto",
    transform=ccrs.PlateCarree()
)

plt.colorbar(sc, shrink=0.8, label="Geoid height [m]")
plt.title("Geoid: NKG2015")

plt.show()

#%% save 
import xarray as xr
import numpy as np

ds = xr.Dataset(
     {
         "geoid": (["latitude", "longitude"], geoid_sub[:,:])
     },
     coords={
         "latitude": lat_sub[:],
         "longitude": lon_sub
     }
 )

ds.geoid.attrs["units"] = "m"
ds.geoid.attrs["long_name"] = "Geoid height"

ds.latitude.attrs["units"] = "degrees_north"
ds.longitude.attrs["units"] = "degrees_east"

ds.to_netcdf("C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/final/XGM2019e.nc")


#%% plot geoids and differences [Code generated with AI]

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import RectBivariateSpline


# -------------------------------------------------------
# Load datasets
# -------------------------------------------------------

folder = "C:/Users/const/Documents/kandidatspeciale/BFN-QG_code/thesis/data/geoid/final/"

files = {
    "NKG2015": folder + "NKG2015.nc",
    "EGG2015": folder + "EGG2015.nc",
    "XGM2019e": folder + "XGM2019e.nc"
}

datasets = {}

for name, file in files.items():
    ds = xr.open_dataset(file)
    datasets[name] = {
        "geoid": ds["geoid"].values,
        "lat": ds["latitude"].values,
        "lon": ds["longitude"].values
    }


# -------------------------------------------------------
# Define target grid (XGM2019e)
# -------------------------------------------------------

target = datasets["XGM2019e"]

lat_target = target["lat"]
lon_target = target["lon"]


# Create mesh for plotting
lon2d, lat2d = np.meshgrid(lon_target, lat_target)


# -------------------------------------------------------
# Spline interpolation function
# -------------------------------------------------------

def spline_interpolate(data, lat, lon, new_lat, new_lon):

    # spline requires increasing coordinates
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        data = data[:, ::-1]

    spline = RectBivariateSpline(
        lat,
        lon,
        data,
        kx=3,
        ky=3
    )

    return spline(new_lat, new_lon)


# -------------------------------------------------------
# Interpolate all datasets onto XGM grid
# -------------------------------------------------------

geoid_interp = {}

for name in datasets:

    if name == "XGM2019e":
        geoid_interp[name] = datasets[name]["geoid"]

    else:
        geoid_interp[name] = spline_interpolate(
            datasets[name]["geoid"],
            datasets[name]["lat"],
            datasets[name]["lon"],
            lat_target,
            lon_target
        )


# -------------------------------------------------------
# Differences
# -------------------------------------------------------

diff_NKG_XGM = geoid_interp["NKG2015"] - geoid_interp["XGM2019e"]
diff_EGG_XGM = geoid_interp["EGG2015"] - geoid_interp["XGM2019e"]
diff_NKG_EGG = geoid_interp["NKG2015"] - geoid_interp["EGG2015"]


# -------------------------------------------------------
# Plot function
# -------------------------------------------------------

lon_min, lon_max = 1, 14
lat_min, lat_max = 56, 69


def plot_map(field, title, cmap="RdBu_r", vmin=None, vmax=None):

    proj = ccrs.Orthographic(
        central_longitude=(lon_min+lon_max)/2,
        central_latitude=(lat_min+lat_max)/2
    )

    fig = plt.figure(figsize=(8,7), dpi=150)

    ax = fig.add_subplot(111, projection=proj)

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

    im = ax.pcolormesh(
        lon2d,
        lat2d,
        field,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )

    plt.colorbar(
        im,
        shrink=0.8,
        label="Geoid height, [m]"
    )

    plt.title(title)
    plt.show()



# -------------------------------------------------------
# Plot geoid models
# -------------------------------------------------------

plot_map(
    geoid_interp["XGM2019e"],
    "XGM2019e Geoid height",
    vmin=36,
    vmax=52
)

plot_map(
    geoid_interp["NKG2015"],
    "NKG2015 interpolated to XGM grid",
    vmin=36,
    vmax=52
)

plot_map(
    geoid_interp["EGG2015"],
    "EGG2015 interpolated to XGM grid",
    vmin=36,
    vmax=52
)


# -------------------------------------------------------
# Plot differences
# -------------------------------------------------------

plot_map(
    diff_NKG_XGM,
    "NKG2015 - XGM2019e",
    cmap="RdBu_r",
    vmin=-0.5,
    vmax=0.5
)

plot_map(
    diff_EGG_XGM,
    "EGG2015 - XGM2019e",
    cmap="RdBu_r",
    vmin=-0.5,
    vmax=0.5
)

plot_map(
    diff_NKG_EGG,
    "NKG2015 - EGG2015",
    cmap="RdBu_r",
    vmin=-0.5,
    vmax=0.5
)