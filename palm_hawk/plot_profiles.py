#!/opt/local/bin/python

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import glob

files_pr = glob.glob('ReD1600_*_pr.all.nc')
files_pr = sorted(files_pr, key=lambda x: x.split('.')[1])
files_pr = sorted(files_pr, key=lambda x: x.split('_')[1])

fig=plt.figure(figsize=(6,5))
fig2=plt.figure(figsize=(6,5))
ax=fig.add_axes([0.1,0.1,0.85,0.85])
ax2=fig2.add_axes([0.1,0.1,0.85,0.85]) 

for f in files_pr:
    f_pr=Dataset(f,'r')
    res=int(f.split('_')[1] )
    red=int(f.split('D')[1].split('_')[0] )
    f_ts_name=('ReD{0}_{1:03d}_ts.all.nc'.format(red,res))
    print(f_ts_name)
    f_ts=Dataset(f_ts_name,'r') 

    nu = 2./(red*red)
    u = f_pr['u'][-1,:]
    z_u=f_pr['zu'][:]
    v = f_pr['v'][-1,:]
    z_v=f_pr['zv'][:]
    ww=f_pr['w*2'][-1,:] 
    us_palm=f_ts['us*'][-1] 
    us =np.sqrt(nu*np.sqrt(u[1]**2+v[1]**2)/z_u[1])
    zp = z_u*us/nu 
    G = np.sqrt(u[-1]**2 + v[-1]**2)

    ax.plot(zp[1:],u[1:]/us_palm,'-',label=f,c='black',alpha=np.sqrt(res/500))
    ax.plot(zp[1:],-v[1:]/us_palm,'-',c='red',alpha=np.sqrt(res/500))

    ax2.plot(zp[1:],ww[1:]/(us_palm**2),'-',label=f,c='black',alpha=np.sqrt(res/500)) 
    lg=plt.legend(loc='best')

    print('Geostrophic wind:', G,us,us_palm/G) 
    print('us_palm', us)
    us_norm = us/G

ax.plot(zp[1:50],4.1*np.log(z_u[1:50])+21.8,ls='--',lw=1,c='gray',label='MOST log-law')
ax.plot(zp[2:200],2.32*np.log(z_u[2:200])+17.2,ls=':',lw=1,c='gray',label='MOST log-law')

ax.set_xscale('log')
ax.set_xlabel('$z^+$')
ax.set_ylabel('$u^+,v^+$')
fig.savefig('les_profiles.pdf',format='pdf')
fig2.savefig('les_ww.pdf',format='pdf')

quit()

    
