#!/usr/local/sci/bin/python2.7
import os
#import matplotlib
#matplotlib.use('Agg')
import iris
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from convert_coordinates import find_centre
from convert_coordinates import calculate_distance
from plot_best_track import track_data
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def calculate_speed():

def make_same_size(array, sarray):
    if array.shape == sarray.shape:
        return sarray
    else:
        sarray = np.append(sarray,0)
        return sarray


def main():
    ############## Hagupit ####################
    monthday = [1201, 1202, 1203, 1204, 1205, 1206, 1207]
    times= [0, 12]
    i=0
    for md in monthday:
        for TT in times:
            if not (md == 1207 and TT == 12):
                dataname = '/nfs/a37/scjea/suite-runs/Hagupit/data/tctracker_data/{0:04d}_{1:02d}Z'.format(md,hr)
                ft1, omw1, lat1, lon1, dpe1, fvort1, fcp1, fmw1, rmw1 = read_tracker_output(dataname)
                if i==0:
                    # Tracker code stops when storm reaches end of domain, hence for each time the arrays are of
                    # different sizes. ft0,lat0,lon0 keeps track of the size of original array.
                    ft, ft0 = ft1, ft1; lat, lat0 = lat1, lat1; lon, lon0 = lon1, lon1
                else:
                    # make same size appends the array with a 0 which must be masked later on.
                    ft1 = make_same_size(ft0,ft1); lat1 = make_same_size(lat0,lat1); lon1 = make_same_size(lon0,lon1);
                    ft = np.vstack([ft,ft1]); lat = np.vstack([lat,lat1]); lon = np.vstack([lon,lon1])

    # Now have 2D array of forecast times in hours and central positions of storms.
    for i in np.arange(ft.shape[0]):  #i.e. the number of forecasts (13 for Hagupit)
        k = 0
        for j in np.arange(ft[0].shape[0] - 1): #i.e. the number of times in forecast
            lat0 = lat[i][j]; lat1 = lat[i][j+1];
            lon0 = lon[i][j]; lon1 = lon[i][j+1];
            dt = ft[i][j+1]-ft[i][j]
            new_speed = calculate_speed(lat0, lat1, lon0, lon1, dt)
            if k==0:
                speed = new_speed
            else:
                speed = np.append((speed,new_speed))


if __name__ == '__main__':
    main()
