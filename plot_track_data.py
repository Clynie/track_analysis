#!/usr/local/sci/bin/python2.7

import os
#import matplotlib
#matplotlib.use('Agg')
import iris
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from plot_storm_track import track_data

def main():
    j=0
    filename = 'ibtracs/hagupit_ibtracs.nc'
    act_lats, act_lons = track_data(filename)
    Latitude=act_lats[0][:]; Longitude=act_lons[0][:]
    
    monthday = [1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208]
    tz = [0, 12]
    
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    best_track = plt.plot(act_lons[0][:],act_lats[0][:],'-')
    ax1.set_xlim([110,167])
    ax1.set_ylim([-5.9,23.8])
    ax1.coastlines(resolution='10m', color='k', linewidth=1)
    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.75, color='k', linestyle=':')
    gl1.xlabels_top = False
    gl1.ylabels_right = False    
    gl1.xlocator = mticker.MultipleLocator(base=5.0)
    gl1.ylocator = mticker.MultipleLocator(base=5.0)

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    best_track = plt.plot(act_lons[0][:],act_lats[0][:],'-')
    ax2.set_xlim([110,167])
    ax2.set_ylim([-5.9,23.8])
    ax2.coastlines(resolution='10m', color='k', linewidth=1)
    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.75, color='k', linestyle=':')
    gl2.xlabels_top = False
    gl2.ylabels_right = False    
    gl2.xlocator = mticker.MultipleLocator(base=5.0)
    gl2.ylocator = mticker.MultipleLocator(base=5.0)

    
    for md in monthday:
        for TT in tz:
            if not (TT==12 and md==1208):
                track_data_file = '/nfs/a37/scjea/suite-runs/Hagupit/data/fricon_track_data_{0:04d}{1:02d}00.npy.npz'.format(md,TT)
                track_data2 = np.load(track_data_file)
                lat = track_data2['arr_0']
                lon = track_data2['arr_1']
                


                if j<8:
                    plt.figure(2)
                    plt.plot(lon,lat,'r-',linewidth=0.5)
                elif j>11: 
                    plt.figure(2)
                    plt.plot(lon[0:72],lat[0:72],'r-',linewidth=0.5)
                else:
                    plt.figure(2)
                    plt.plot(lon,lat,'r-',linewidth=0.5)
                j=j+1
    plt.show()

if __name__ == '__main__':
    main()
