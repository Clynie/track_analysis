import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from netCDF4 import Dataset
import iris
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def find_centre(slp):
    index = np.argmin(slp.data)
    indices= np.unravel_index(index,slp.data.shape)
    cenlat = slp.coord('latitude').points[indices[0]] # Find the coordinates for this central position
    cenlon = slp.coord('longitude').points[indices[1]]
    minpress = slp.data[indices]
    return cenlat, cenlon, minpress

def reduce_slp(slp, minlat, maxlat, minlon, maxlon):
    longitude_constraint1 = iris.Constraint(longitude = lambda cell:cell<maxlon)
    longitude_constraint2 = iris.Constraint(longitude = lambda cell:cell>minlon)
    latitude_constraint1 = iris.Constraint(latitude = lambda cell:cell<maxlat)
    latitude_constraint2 = iris.Constraint(latitude = lambda cell:cell>minlat)
    slp_reduced = slp.extract(longitude_constraint1&longitude_constraint2&latitude_constraint1&latitude_constraint2)
    return slp_reduced

def main():
    monthday = [1104, 1105, 1106, 1107, 1108]
    times = [0, 12]
    NN = np.arange(20)*6
    
    plt_labels = ['4/11, 00Z','4/11, 12Z','5/11, 00Z','5/11, 12Z','6/11, 00Z', '6/11, 12Z','7/11, 00Z', '7/11, 12Z', '8/11, 00Z', '8/11, 12Z', '9/11, 00Z']
    
    filename = './haiyan_ibtracs.nc'
    dataset = Dataset(filename)
    print dataset
    act_lats =  dataset.variables['lat_for_mapping'][:]
    print act_lats
    act_lons = dataset.variables['lon_for_mapping'][:]
    Latitude=act_lats; Longitude=act_lons
    #exit()
    i=7
    
    for md in monthday:
        for TT in times:
            minlat = Latitude[i] - 5; maxlat = Latitude[i] + 5;
            minlon = Longitude[i] - 5; maxlon = Longitude[i] + 5;
            centre_lons = []; centre_lats = []; press_mins = []
            for vt in NN:
                filename = '/nfs/a37/scjea/suite-runs/Haiyan/data/slp/2013{0:4d}T{1:02d}00Z_Haiyan_4p4_L80_singv_2p1_pc{2:03d}.pp'.format(md,TT,vt)
                constraint = iris.AttributeConstraint(STASH='m01s16i222') #surface pressure
                slp = iris.load_cube(filename, constraint)
                slp = slp[0][:][:]  
                slp_reduced = reduce_slp(slp,minlat,maxlat,minlon,maxlon)
                cenlat, cenlon, minpress = find_centre(slp_reduced)
                minlat = cenlat - 5; maxlat = cenlat + 5; minlon = cenlon - 5; maxlon = cenlon + 5;
                if not centre_lons ==[]:
                    if centre_lons[-1]<112:
                        centre_lons = np.append(centre_lons,centre_lons[-1]); centre_lats = np.append(centre_lats,centre_lats[-1])
                        press_mins = np.append(press_mins,press_mins[-1])
                    else:
                        centre_lons = np.append(centre_lons,cenlon); centre_lats = np.append(centre_lats,cenlat)
                        press_mins = np.append(press_mins,minpress)
                else:
                    centre_lons = np.append(centre_lons,cenlon); centre_lats = np.append(centre_lats,cenlat)
                    press_mins = np.append(press_mins,minpress)
                
            if i==7:
                lats = centre_lats;
                lons = centre_lons;
                press  = press_mins;
            else:
                lats = np.vstack([lats,centre_lats]); 
                lons = np.vstack([lons,centre_lons]);
                press = np.vstack([press,press_mins])
            i = i+2
    
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    best_track = plt.plot(Longitude,Latitude,'k-',linewidth=2,label='IBTrACS Data')
    triangles = plt.plot(Longitude[3:-1:4],Latitude[3:-1:4],'gv',markersize=10,label='Position at 00Z')
    ax.set_xlim([100,170])
    ax.set_ylim([-4, 26])
    #daybesttrack = plt.plot(Longitude[2:len(Longitude):4],Latitude[2:len(Latitude):4],'gv',markersize=10,label='Position at 0Z')
    ax.annotate('Time-lagged track ensemble for Typhoon Haiyan',xy=(0.5,0.99), size='xx-large', xycoords='axes fraction', horizontalalignment='center', verticalalignment='top')
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
    colors = ['b','g','r','c','m','y','chocolate','b','g','r','c','m','y']
    linestyles = ['-','-','-','-','-','-','-','--','--','--','--','--','--']
    for i in np.arange(len(lons)):
        plt.plot(lons[i][:],lats[i][:],linewidth=1.5,linestyle=linestyles[i],color=colors[i],label=plt_labels[i])
    plt.legend(loc='lower left')
        
    for i,j,k in zip(Longitude[3:len(Longitude):4],Latitude[3:len(Latitude):4],np.arange(len(Latitude[3:len(Latitude):4]))):
        if k==5:
            day = 8
            month = 11
            ax.annotate('{0:02d}/{1:02d}'.format(day,month),xy=(i,j-1.5),xycoords='data', horizontalalignment='center',size='x-large')
        else:
            day = k+3
            month=11
            ax.annotate('{0:02d}/{1:02d}'.format(day,month),xy=(i,j-0.8),xycoords='data', horizontalalignment='center',size='x-large')
        
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1,1,1)
    for i in np.arange(len(press)):
        plt.plot(press[i][:])
    
    plt.show()
    

            
    
                
    
    
    
    
    
if __name__=='__main__':
    main()
