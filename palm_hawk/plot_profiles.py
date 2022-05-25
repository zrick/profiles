#!/opt/local/bin/python

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import glob

files = glob.glob('ReD1600_*_pr.*.nc')
files = sorted(files, key=lambda x: x.split('.')[1])
files = sorted(files, key=lambda x: x.split('_')[1])


fig=plt.figure(figsize=(6,5))
ax=fig.add_axes([0.1,0.1,0.85,0.85])

for f in files:
    f_nc=Dataset(f,'r') 
    u = f_nc['u'][-1,:]
    z_u=f_nc['zu'][:]
    ax.plot(u[1:],z_u[1:],'o-',label=f)

    lg=plt.legend(loc='best') 
ax.set_yscale('log')
ax.set_ylim(0.01,2e2)
plt.show() 
    
