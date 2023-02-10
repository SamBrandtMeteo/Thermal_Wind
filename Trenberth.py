#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:22:50 2022

@author: sambrandt
"""
# MODULES #
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
import xarray as xr

# CONSTANTS #
g=9.8 # Gravitational acceleration
Omega=7.2921*10**-5 # Rotation rate of Earth in rad/s

# INPUTS #
# Pressure Level (mb)
# MUST be a multiple of 50 mb, otherwise there WILL be an error
plev=500
# Edges of Domain (in degrees)
north=44
south=22
east=-77
west=-104
# Time of Output (in UTC; code will find the latest GFS run to include it)
year=2023
month=2
day=11
hour=15
# Thermal Wind Barb Spacing (in degrees, must be an integer)
# 2 and larger better for zoomed out grids, 1 better for zoomed in grids
barbstep=1

# FUNCTIONS #
# Function to calculate gradients on a lat/lon grid
def partial(lat,lon,field,wrt):
    gradient=np.zeros(np.shape(field))
    if wrt=='x':
        upper=field[:,2::]
        lower=field[:,0:-2]
        dx=111000*np.cos(lat[:,2::]*(np.pi/180))*(lon[0,1]-lon[0,0])
        grad=(upper-lower)/(2*dx)
        gradient[:,1:-1]=grad
        gradient[:,0]=grad[:,0]
        gradient[:,-1]=grad[:,-1]
    if wrt=='y':
        upper=field[2::,:]
        lower=field[0:-2,:]
        dy=111000*(lat[1,0]-lat[0,0])
        grad=(upper-lower)/(2*dy)
        gradient[1:-1,:]=grad
        gradient[0,:]=grad[0,:]
        gradient[-1,:]=grad[-1,:] 
    return gradient

# DATA DOWNLOAD #
# Define location of the data in THREDDS
best_gfs=TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_onedeg/latest.xml')
best_ds=list(best_gfs.datasets.values())[0]
ncss=best_ds.subset()
# Create a datetime object to specify the output time that you want
valid=datetime(year,month,day,hour)
# Establish a query for the data
query = ncss.query()
# Trim data to location/time of interest
query.lonlat_box(north=north,south=south,east=east,west=west).time(valid)
# Specify that output needs to be in netcdf format
query.accept('netcdf4')
# Specify the variables that you want
query.variables('Geopotential_height_isobaric')
# Retrieve the data using the info from the query
data=ncss.get_data(query)
data=xr.open_dataset(NetCDF4DataStore(data))

# VARIABLE DEFINITIONS #
# Retrieve the geopotential height fields
plevs=np.array(list(map(float,ncss.metadata.axes['isobaric']['attributes'][2]['values'])))
upper=np.where(plevs==(plev+100)*100)[0][0]
lower=np.where(plevs==(plev-100)*100)[0][0]
gpht=np.array(data['Geopotential_height_isobaric'][0,lower:upper+1:2,:,:])
# Define the lat/lon grid with 1 degree spacing
lat=np.arange(south,north+1)
lon=np.arange(west,east+1)
lon,lat=np.meshgrid(lon,lat)

# CALCULATIONS #
# Planetary vertical vorticity
f=2*Omega*np.sin(lat*(np.pi/180))
# Lower pressure level geostrophic wind components
ugeo600=-(g/f)*partial(lat,lon,gpht[-1,:,:],'y')
vgeo600=(g/f)*partial(lat,lon,gpht[-1,:,:],'x')
# Middle pressure level geostrophic wind components
ugeo500=-(g/f)*partial(lat,lon,gpht[1,:,:],'y')
vgeo500=(g/f)*partial(lat,lon,gpht[1,:,:],'x')
# Upper pressure level geostrophic wind components
ugeo400=-(g/f)*partial(lat,lon,gpht[0,:,:],'y')
vgeo400=(g/f)*partial(lat,lon,gpht[0,:,:],'x')
# Middle pressure level thermal wind using centered difference
# Units of kt per mb
utwn=1.944*(ugeo400-ugeo600)/-200
vtwn=1.944*(vgeo400-vgeo600)/200
# Middle pressure level absolute vertical vorticity (geostrophic+planetary)
geov500=partial(lat,lon,vgeo500,'x')-partial(lat,lon,ugeo500,'y')+f

# FIGURE #
# Create cartopy axis
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(),'adjustable': 'box'},dpi=1000)
# Add geographic borders
ax.coastlines(lw=0.25)
ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=0.25)
ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='black',linewidth=0.25)
# Set aspect ratio, establish twin axis for seconday title (further down)
ax.set_box_aspect(len(lat[:,0])/len(lat[0,:]))
# Filled contours of absolute geostrophic vorticity
if np.mean(f)>0:
    levels=np.arange(10,110,10)
    cmap='plasma_r'
elif np.mean(f)<0:
    levels=np.arange(-100,0,10)
    cmap='plasma'
pcm=ax.contourf(lon,np.flip(lat,axis=0),geov500*10**5,levels,cmap=cmap)
# Create new, dynamically scaled axis for the colorbar
cax=fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
# Height contours
ct=ax.contour(lon,np.flip(lat,axis=0),gpht[1,:,:]/10,np.arange(int(np.min(gpht[1,:,:]/10)),int(np.max(gpht[1,:,:]/10))+3,3),colors='black',linewidths=0.25)
ax.clabel(ct, inline=True, fontsize=4)
# Thermal wind barbs
ax.barbs(lon[::barbstep,::barbstep],np.flip(lat[::barbstep,::barbstep],axis=0),utwn[::barbstep,::barbstep]*10**2,vtwn[::barbstep,::barbstep]*10**2,pivot='tip',length=4,linewidth=0.5,color='black')    
# Colorbar
cbar = plt.colorbar(pcm,cax=cax)
cbar.ax.set_ylabel('Vorticity (x$10^{-5}$ $s^{-1}$)',fontsize=6)
cbar.ax.tick_params(labelsize=8)  
# Upper title
ax.set_title(str(plev)+' mb Absolute Geostrophic Vorticity (x$10^{-5}$ $s^{-1}$)\n'+str(plev+100)+'-'+str(plev-100)+' mb Thermal Wind Barbs (x$10^{-2}$ $kt$ $mb^{-1}$)\nGFS Initialized '+ncss.metadata.time_span['begin'][0:10]+' '+ncss.metadata.time_span['begin'][11:13]+'z, Valid '+str(valid)[0:13]+'z',fontsize=8)







