#!/opt/local/bin/python

from netCDF4 import Dataset
import numpy as np 
import matplotlib.pyplot as plt 
import my_pylib as mp

f_coriolis=1.0    # Coriolis parameter = 1
U_G=1.0           # Geostrophic wind = 1 
                  # => nu=1/re

path='/Users/zrick/WORK/research_projects/SIM_NEUTRAL/netcdf/' 
cases=[400,500,750,1000,1300,1600]

files={ 400: 'avg_flw_ri0.0_re400.0_1024x192x1024.nc' , 
        500: 'avg_flw_ri0.0_re500.0_2048x192x2048.nc' , 
        750: 'avg_flw_ri0.0_re750.0_3072x384x3072.nc' ,
        1000:'avg_flw_ri0.0_re1000.0_co0_3072x512x6144.nc' ,
        1300:'avg_flw_ri0.0_re1300.0_co0_2560x640x5120.nc' ,
        1600:'avg_flw_ri0.0_re1600.0_co0_3840x960x7680.nc'} 

it_start= { 
    400:100,
    500:200,
    750:0,
    1000:100,
    1300:120,
    1600:90       
    }

angle={400:0.26177, 500:0., 750:0.35779, 1000:0.35779, 1300:0.35779, 1600:0.35779}
color={400:'grey', 500:'blue', 750: 'purple',  1000:'red', 1300:'orange', 1600:'green'}
ustar={} 
alpha={} 
retau={}
nu={} 
reg={}

# check tensor rotation
check_tensor=0
if check_tensor != 0 :
    a1=np.zeros([3,3])
    print(a1) 
    b1=mp.rotate_tensor(a1,1.0) 
    print(b1)
    print('=====')


    a2=np.zeros([3,3]) 
    a2[1,1]=1.0
    print(a2)
    b2=mp.rotate_tensor(a2,1.0) 
    print(b2)
    print('======')
    
    a3=np.zeros([3,3]) 
    a3[0,0]=1.0
    a3[1,1]=5.0 
    a3[2,2]=1.0
    a3[0,1]=0.1
    a3[0,2]=0.5
    a3[1,2]=0.0
    a3[2,0]=0.0
    print(a3)
    b3=mp.rotate_tensor(a3,np.pi/4)
    print(b3)
    print('======')

o=Dataset('data_hauke.nc','w',format='NETCDF4') 
for re in cases: 
    grp=o.createGroup('re{}_neutral'.format(re)) 
    i=Dataset('{}/{}'.format(path,files[re]),'r',format='NETCDF4')     
    ny=len(i.dimensions['y'])
    grp.createDimension('y',ny)  
    v_y=grp.createVariable('y','f4',('y',))    
    v_y.long_name='Distance from Wall normalized by Rossby Radius \Lambda=G/f' 

    reg[re]=re*re/2
    nu[re]=1./reg[re]
    ustar[re]=np.mean(i.variables['FrictionVelocity'][it_start[re]:])
    alpha[re]=np.mean(i.variables['FrictionAngle'][it_start[re]:] /180*np.pi)
    retau[re]=ustar[re]**2/nu[re]
    
    grp.setncattr('REg= G Lambda (nu-1)',reg[re]) 
    grp.setncattr('ustar',ustar[re])  
    grp.setncattr('Surface veering [alpha]',alpha[re]) 
    grp.setncattr('delta=ustar (f-1)',ustar[re]/f_coriolis)  
    grp.setncattr('RE_tau=delta_plus',retau[re])
    grp.setncattr('Viscosity nu',nu[re]) 

fig=plt.figure() 
ax=fig.add_axes([0.1,0.1,0.8,0.8])
print('  Re  ustar  alpha  Re_tau')

for re in cases: 
    f=Dataset('{}/{}'.format(path,files[re]),'r',format='NETCDF4')    
    grp=o['re{}_neutral'.format(re)]
    y=f.variables['y'][:] 
    us=ustar[re] 
    al=alpha[re]
    n=nu[re]
    y[0]=y[1]/2.
    delta_p=us*us/n
    # Info 
    print('{0:4d} {1:6.4f} {2:6.1f} {3:7.0f}'.format(re,us,(-al+angle[re])/np.pi*180,delta_p))

    # Vertical axes
    v_yp=grp.createVariable('y_p','f4',('y',)) 
    v_yp.long_name='Distance from Wall in Wall units'  
    v_yp[:]=y*us/n


    v_ym=grp.createVariable('y_m','f4',('y',))
    v_ym.long_name='Distance from Wall in Boundary-layer depth scales \delta=u_\star/\nu'
    v_ym[:]=y/(us/f_coriolis)   

    # Velocity profiles 
    u_m=np.mean(f.variables['rU'][it_start[re]:,:],0)
    u_p=u_m/us
    u_p[0]=u_p[1]/2. 

    w_m=np.mean(f.variables['rW'][it_start[re]:,:],0)
    w_p=w_m/us
    w_p[0]=w_p[1]/2.

    delta=us/f_coriolis
    delta_p=delta*us/n

    [u_m,w_m]=mp.rotate(u_m,w_m,angle[re])    # align outer scaling with geostrophic wind
    [u_p, w_p]=mp.rotate(u_p,w_p,al)          # align inner scaling with wall shear stress

    v_up=grp.createVariable('U_p','f4',('y',))  
    v_up.long_name='Streamwise velocity [aligned with surface shear], normalized by u*'
    v_up[:]=u_p

    v_wp=grp.createVariable('W_p','f4',('y',)) 
    v_wp.long_name='Spanwise velocity [in-plane orthogonal to surface shear], normalized by u*'
    v_wp[:]=w_p

    v_um=grp.createVariable('U_m','f4',('y',)) 
    v_um.long_name='Streamwise velocity [aligned with U_G], normalized by U_G'
    v_um[:]=u_m 

    v_wm=grp.createVariable('W_m','f4',('y',)) 
    v_wm.long_name='Spanwise velocity [in-plane orthogonal to U_G], normalized by U_G'
    v_wm[:]=w_m

    #plt.plot(y*us/n,u_p, label=r'w, Re={}'.format(re),c=color[re],ls='-',  lw=2)   
    #plt.plot(y*us/n,w_p, label=r'w, Re={}'.format(re),c=color[re],ls='--',  lw=2)   
    
    # Reynolds stress profiles  
    vid = {}
    ny=len(y) 
    v=np.zeros([3,3,ny])
    rs = np.array([['xx','xy','xz'], ['xy','yy','yz'], ['xz','yz','zz']]) 
    for i in range(3): 
        for j in range(i+1): 
            v[i,j]=np.mean(f.variables['R{}'.format(rs[i,j])][it_start[re]:,:],0)  
            if ( i!= j ): 
                v[j,i]=v[i,j]
    v_p=np.array([ mp.rotate_tensor(v[:,:,l],al)/us/us for l in range(ny) ])
    v_m=np.array([ mp.rotate_tensor(v[:,:,l],angle[re])for l in range(ny) ])


    for i in range(3):
        for j in range(i+1):
            v=grp.createVariable('R{}_p'.format(rs[i,j]),'f4',('y',)) 
            v.long_name='Reynolds stress R{} in stress aligned coordinates and normalized by u*^2' 
            v[:]=v_p[:,i,j] 

            v=grp.createVariable('R{}_m'.format(rs[i,j]),'f4',('y',)) 
            v.long_name='Reynolds stress R{} in outer [aligned with U_G] coordinates and normalized by U_G^2' 
            v[:]=v_p[:,i,j] 
            
    #plt.plot(y/us,v_m[:,1,2],label='Ryz, Re={}'.format(re),c=color[re],ls='-',lw=2)
    #plt.plot(y/us,v[1,2,:],label='Ryz, Re={}'.format(re),c=color[re],ls=':',lw=2)

lg=plt.legend()
ax.set_xscale('log') 
plt.show() 
