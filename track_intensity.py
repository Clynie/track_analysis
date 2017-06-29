#!/usr/local/sci/bin/python2.7

import os
#import matplotlib
#matplotlib.use('Agg')
import iris
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from get_time import get_time
from read_tctracker_data import *
from plot_best_track import track_data
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import itertools

def plot_track(lats,lons,plt_labels):
    lats = np.ma.masked_where(lats==0,lats)
    lons = np.ma.masked_where(lons==0,lons)
    print lons
    print lons.shape
    filename = './hagupit_ibtracs.nc'
    act_lats, act_lons = track_data(filename)
    Latitude=act_lats[0][:]; Longitude=act_lons[0][:]
    print Latitude

    fname = '/nfs/a37/scjea/suite-runs/Hagupit/data/20141203T1200Z_Hagupit_4p4_L80_singv_2p1_new_pa000.pp'
    orog = iris.load_cube(fname)
    print orog
    orogX = orog.coord('longitude').points
    orogY = orog.coord('latitude').points
    X,Y = np.meshgrid(orogX,orogY)

    orog.data = np.ma.masked_less(orog.data,10)
    orog.data = np.ma.masked_greater(orog.data,3000)

    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    #orog_plt = ax.pcolormesh(X,Y,orog.data)
    #plt.colorbar(orog_plt)
    best_track = plt.plot(Longitude,Latitude,'k-',linewidth=2.5,label='IBTrACS Data')
    ax.set_xlim([130,137])
    ax.set_ylim([7.5, 12])
    #ax.set_xlim([110,155])
    #ax.set_ylim([-2.5, 17.5])
    daybesttrack = plt.plot(Longitude[3:len(Longitude):4],Latitude[3:len(Latitude):4],'gv',markersize=10,label='Position at 0Z')
    ax.annotate('Time-lagged track ensemble for Typhoon Hagupit',xy=(0.5,0.99), size='xx-large', xycoords='axes fraction', horizontalalignment='center', verticalalignment='top')
    ax.annotate('5 day forecasts initialised 12 hrs apart', xy=(0.5,0.95), size='xx-large', xycoords='axes fraction', horizontalalignment='center', verticalalignment='top')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.coastlines(resolution='10m', color='k', linewidth=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.75, color='k', linestyle=':')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style= {'size':20}
    gl.ylabel_style= {'size':20}
    gl.xlocator = mticker.MultipleLocator(base=5.0)
    gl.ylocator = mticker.MultipleLocator(base=5.0)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    for i,j,k in zip(Longitude[3:len(Longitude):4],Latitude[3:len(Latitude):4],np.arange(len(Latitude[3:len(Latitude):4]))):
        if k==0:
            day=1
            month=12
            ax.annotate('{0:02d}/{1:02d}'.format(day,month),xy=(i,j-0.4),xycoords='data', horizontalalignment='right',size='x-large')
        else:
            day = k+1
            month=12
            if k<5:
                #ax.annotate('{0:02d}/{1:02d}'.format(day,month),xy=(i,j-0.8),xycoords='data', horizontalalignment='center',size='x-large')
                ax.annotate('{0:02d}/{1:02d}'.format(day,month),xy=(i,j-0.2),xycoords='data', horizontalalignment='center',size='x-large')
            elif k==5:
                ax.annotate('{0:02d}/{1:02d}'.format(day,month),xy=(i,j+0.6),xycoords='data', horizontalalignment='center',size='x-large')

            else:
                ax.annotate('{0:02d}/{1:02d}'.format(day,month),xy=(i,j+0.4),xycoords='data', horizontalalignment='center',size='x-large')

    colors = ['b','g','r','c','m','y','chocolate','b','g','r','c','m','y']
    linestyles = ['-','-','-','-','-','-','-','--','--','--','--','--','--']

    for i in np.arange(lats.shape[0]):
        if i==3:
            plt.figure(1)
            plt.plot(lons[i,5],lats[i,5],'ro',markersize=15,label='03/12, 18Z')
            plt.plot(lons[i,9],lats[i,9],'ms',markersize=15,label='04/12, 18Z')
            plt.plot(lons[i,:],lats[i,:],label=plt_labels[i],linestyle=linestyles[i],color=colors[i],linewidth=3)

        elif i==5:
            plt.figure(1)
            plt.plot(lons[i,:],lats[i,:],label=plt_labels[i],linestyle=linestyles[i],color=colors[i],linewidth=3)
            plt.plot(lons[i,1],lats[i,1],'ro',markersize=15)
            plt.plot(lons[i,5],lats[i,5],'ms',markersize=15)
        elif i==7:
            plt.figure(1)
            plt.plot(lons[i,:],lats[i,:],label=plt_labels[i],linestyle=linestyles[i],color=colors[i],linewidth=3)
            plt.plot(lons[i,1],lats[i,1],'ms',markersize=15)
    plt.legend(loc='lower left')
    plt.show()
    plt.close()

def plot_ws(ft, fmw, plt_labels):
    fmw = np.ma.masked_where(fmw==0,fmw)
    ft[:,1:-1] = np.ma.masked_where(ft[:,1:-1]==0,ft[:,1:-1])
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    obw = [30, 30, 35, 35, 40, 45, 50, 55, 60, 70, 75, 80, 85, 95, 105, 115, 115, 115, 110, 110, 110, 100, 95, 95, 90, 90, 85, 85, 75, 65, 60, 55, 45, 40, 35, 35, 40, 40, 40, 40, 40, 40, 35, 30, 30, 30, 30, 30]
    obw = np.asarray(obw)
    obw = obw*0.5144444
    time = np.arange(48)*6
    plt.plot(time,obw,'k-',linewidth=2,label='Best track data')

    colors = ['b','g','r','c','m','y','chocolate','b','g','r','c','m','y']
    linestyles = ['-','-','-','-','-','-','-','--','--','--','--','--','--']

    for i in np.arange(fmw.shape[0]):
        plt.figure(1)
        vt = ft[i,:] + (i+1)*12
        plt.plot(vt,fmw[i,:],label=plt_labels[i],linewidth=1.5,linestyle=linestyles[i],color=colors[i])

    intervals = np.arange(12,300,24)
    labels = ['01/12', '02/12', '03/12', '04/12', '05/12', '06/12', '07/12',  '08/12',  '09/12', '10/12', '11/12',  '12/12',  '13/12']
    plt.xticks(intervals,labels,rotation='horizontal')
    plt.tick_params(labelsize=20)
    ax.set_xlabel('Date',fontsize='x-large')
    ax.set_ylabel('Windspeed, ms$^{-1}$', fontsize='x-large')
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()

def plot_mslp(ft, fcp, plt_labels):
    fcp = np.ma.masked_where(fcp==0,fcp)
    ft[:,1:-1] = np.ma.masked_where(ft[:,1:-1]==0,ft[:,1:-1])
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)

    Pressure = [1006, 1004, 1000, 998, 994, 990, 985, 980, 975, 965, 955, 950, 945, 935, 915, 905, 905, 905, 910, 910, 910, 925, 935, 935, 945, 945, 955, 955, 965, 975, 980, 985, 990, 990, 994, 994, 990, 990, 990, 990, 990, 990, 996, 1002, 1004, 1006, 1008, 1008]
    Pressure = np.asarray(Pressure)
    time = np.arange(48)*6
    plt.plot(time,Pressure,'k-',linewidth=2,label='Best track data')
    colors = ['b','g','r','c','m','y','chocolate','b','g','r','c','m','y']
    linestyles = ['-','-','-','-','-','-','-','--','--','--','--','--','--']

    for i in np.arange(fcp.shape[0]):
        plt.figure(1)
        vt = ft[i,:] + (i+1)*12
        plt.plot(vt,fcp[i,:],label=plt_labels[i],linewidth=1.5,color=colors[i], linestyle=linestyles[i])
    intervals = np.arange(12,300,24)
    labels = ['01/12', '02/12', '03/12', '04/12', '05/12', '06/12', '07/12',  '08/12',  '09/12', '10/12', '11/12',  '12/12',  '13/12']
    plt.xticks(intervals,labels,rotation='horizontal')
    plt.tick_params(labelsize=20)
    ax.set_xlabel('Date',fontsize='x-large')
    ax.set_ylabel('Minimum sea level pressure, hPa', fontsize='x-large')
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()

def plot_poserror_dpe1(ft,dpe,plt_labels):
    dpe = np.ma.masked_where(dpe==0,dpe)
    ft[:,1:-1] = np.ma.masked_where(ft[:,1:-1]==0,ft[:,1:-1])
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    colors = ['b','g','r','c','m','y','chocolate','b','g','r','c','m','y']
    linestyles = ['-','-','-','-','-','-','-','--','--','--','--','--','--']

    for i in np.arange(dpe.shape[0]):
        plt.figure(1)
        vt = ft[i,:]
        plt.plot(vt,dpe[i,:],label=plt_labels[i],linewidth=2,linestyle=linestyles[i],color=colors[i])
    plt.grid()
    ax.set_xlabel('Forecast time (hr)',fontsize='x-large')
    ax.set_ylabel('Direct poistional error, km', fontsize='x-large')
    plt.legend(loc='upper left')
    plt.show()


def plot_poserror_dpe2(ft,dpe,plt_labels):
    dpe = np.ma.masked_where(dpe==0,dpe)
    ft[:,1:-1] = np.ma.masked_where(ft[:,1:-1]==0,ft[:,1:-1])
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    colors = ['b','g','r','c','m','y','chocolate','b','g','r','c','m','y']
    linestyles = ['-','-','-','-','-','-','-','--','--','--','--','--','--']

    for i in np.arange(dpe.shape[0]):
        plt.figure(1)
        vt = ft[i,:] + (i+1)*12
        plt.plot(vt,dpe[i,:],label=plt_labels[i],linewidth=2,color=colors[i],linestyle=linestyles[i])
    plt.xlim(xmin=0)
    intervals = np.arange(12,300,24)
    labels = ['01/12', '02/12', '03/12', '04/12', '05/12', '06/12', '07/12',  '08/12',  '09/12', '10/12', '11/12',  '12/12',  '13/12']
    plt.xticks(intervals,labels,rotation='horizontal')
    plt.tick_params(labelsize=20)
    ax.set_xlabel('Date',fontsize='x-large')
    ax.set_ylabel('Direct positional error, km', fontsize='x-large')
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()

def make_same_size(array, sarray):
    if array.shape == sarray.shape:
        return sarray
    else:
        sarray = np.append(sarray,0)
        return sarray

def main():
##################################################################################

    monthday = [1201, 1202, 1203, 1204, 1205, 1206, 1207]
    hours = [0,12]
    i=0
    plt_labels = ['01/12, 00Z','01/12, 12Z','02/12, 00Z','02/12, 12Z','03/12, 00Z','03/12, 12Z','04/12, 00Z','04/12, 12Z','05/12, 00Z','05/12, 12Z','06/12, 00Z','06/12, 12Z','07/12, 00Z']
    for md in monthday:
        for hr in hours:
            if not (md == 1207 and hr == 12):
                dataname = '/nfs/a37/scjea/suite-runs/Hagupit/data/tctracker_data/{0:04d}_{1:02d}Z'.format(md,hr)
                ft1, omw1, lat1, lon1, dpe1, fvort1, fcp1, fmw1, rmw1 = read_tracker_output(dataname)
                print ft1.shape
                if i==0:
                    ft, ft0 = ft1, ft1; omw, omw0 = omw1, omw1; lat, lat0=lat1, lat1; lon, lon0 = lon1, lon1; dpe, dpe0 = dpe1, dpe1; fvort, fvort0 = fvort1, fvort1
                    fmw, fmw0 = fmw1, fmw1; rmw, rmw0 = rmw1, rmw1; fcp, fcp0 = fcp1, fcp1;
                    i=i+1
                else:
                    ft1 = make_same_size(ft0,ft1); omw1 = make_same_size(omw0,omw1); lat1 = make_same_size(lat0,lat1); lon1 = make_same_size(lon0,lon1); dpe1 = make_same_size(dpe0,dpe1);
                    fvort1 = make_same_size(fvort0,fvort1); fmw1 = make_same_size(fmw0,fmw1); rmw1 = make_same_size(rmw0,rmw1); fcp1 = make_same_size(fcp0,fcp1);
                    ft = np.vstack([ft,ft1]); omw = np.vstack([omw, omw1]); lat = np.vstack([lat,lat1]); lon = np.vstack([lon,lon1]); dpe = np.vstack([dpe,dpe1]); fvort = np.vstack([fvort,fvort1])
                    fmw = np.vstack([fmw, fmw1]); rmw = np.vstack([rmw,rmw1]); fcp = np.vstack([fcp,fcp1])

    fmw = fmw*0.5144444
    plot_track(lat,lon,plt_labels)
    plot_ws(ft,fmw,plt_labels)
    plot_mslp(ft,fcp,plt_labels)
    plot_poserror_dpe1(ft,dpe,plt_labels)
    plot_poserror_dpe2(ft,dpe,plt_labels)

if __name__ == '__main__':
    main()
