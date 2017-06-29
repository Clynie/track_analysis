# Need to add day steps onto plots. Combine with plot_track_data

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import cartopy.crs as ccrs
from datetime import datetime


def track_data(filename):
    dataset = Dataset(filename)
    lats =  dataset.variables['lat_for_mapping'][:]
    lons = dataset.variables['lon_for_mapping'][:]
    return lats,lons


def main():
    filename = 'ibtracs/hagupit_ibtracs.nc'
    lats,lons = track_data(filename)

    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    plt.plot(lons[0][:],lats[0][:],'-')
    ax.set_ylim([-5.9,23.8])
    ax.set_xlim([110,167])
    ax.coastlines(resolution='10m', color='k', linewidth=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.75, color='k', linestyle=':')
    plt.show()

if __name__=='__main__':
    main()
