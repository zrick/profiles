zaza#!/opt/local/bin/python 

import numpy as np
import my_pylib as mp
import matplotlib.pyplot as plt
from netCDF4 import Dataset 

D2R = np.pi/180. 

file='/Users/zrick/WORK/research_projects/SIM_NEUTRAL/netcdf/avg_flw_ri0.0_re1000.0_co0.0_1536x512x3072.nc'
f=Dataset(file,'r')

t=f['t'][:]
i0=mp.geti(t,t[-1]-2*np.pi) -1

u=f['rU'][i0:,:]
w=f['rW'][i0:,:]
al=f['FrictionAngle'][i0:]
us=f['FrictionVelocity'][i0:] 
t=f['t'][i0:]
it=f['it'][i0:] 
tke=f['Tke'][i0:,:]
y=f['y'][:] 

t_delta = t[-1]-t[0]- 2*np.pi

al_mean=np.mean(al) 
print('DIFFERENCE TO EXACT PREIODICITY:',t_delta/np.pi/2)
print('MEAN VEERING:', al_mean,'deg / ', al_mean*D2R, 'rad' ) 

tke_t=tke[10:,:]-tke[:-10,:]
u_t = u[8:,:] - u[:-8,:]
u_avg=np.mean(u,0)
w_avg=np.mean(w,0)

[u_rot,w_rot]= mp.rotate(u_avg,w_avg,0) #al_mean*D2R) 

print('w1-w0 ROTATED:', w_rot[1]-w_rot[0])
print('w1-w0 ORIGINAL:',w_avg[1]-w_avg[0])
print('===')
print('GEO rotation orig:',np.arctan(w_avg[-1]/u_avg[-1]))
print('GEO rotation rotd:',np.arctan(w_rot[-1]/u_rot[-1])) 

u_delta=u_rot-u[-1,:]
w_delta=w_rot-w[-1,:]

u_rotn=u_rot-u_avg
w_rotn=w_rot-w_avg 

#plt.plot(y,u_delta,label=r'$\Delta u$',ls='-',lw=1,c='red')
#plt.plot(y,u_rotn,label=r'$\Delta u (rotate)$', ls=':',lw=1,c='red')
#plt.plot(y,w_delta,label=r'$\Delta w$',ls='-',lw=1,c='blue')
#plt.plot(y,w_rotn,label=r'$\Delta u (rotate)$', ls=':',lw=1,c='blue')
plt.plot(t,u[:,-1]-u_avg[-1])
plt.plot(t,w[:,-1]-w_avg[-1]) 
usm=np.mean(us)
usv=np.var(us) 
#plt.plot(t,us*0+usm-np.pi/2*np.sqrt(usv)*np.sin(t-t[0]+0.3),ls='-',lw=1)
#plt.plot(t,us, ls=':',lw=1)
lg=plt.legend(loc='best')
lg.get_frame().set_linewidth(0.0) 
plt.show() 


for i in range(len(y)):
    print(i,u_avg[i],w_avg[i],u_delta[i],w_delta[i]) 
