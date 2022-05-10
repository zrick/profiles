#!/opt/local/bin/python3 

import matplotlib.pyplot as plt
import numpy as np

t_f8=np.dtype('<f8') 


iv=1
nz=834
ny=5120
nx=5120 
nzy=nz*ny
nzx=nz*nx 

fig=plt.figure(figsize=(10,5))
ax=fig.add_axes([00.1,0.1,0.8,0.8])
grid_u300=[0,0.0338,0.1014,0.1691,0.2368]
u300=[0,0.0156,0.0210,0.0256,0.0283]
grid_u500=[0,0.020616,0.061848,0.10308,0.144312,0.185544,0.22678,0.26801]
u500=[0,0.010316,0.0205,0.0210,0.0235,0.02626,0.02767,0.02877]

u500_2=u500[:]
u500_2[2]=(u500_2[1]+u500_2[3])/2.

ax.plot(u300,grid_u300,ls='-',lw=3,c='black',label='ReD1600_300')
ax.plot(u500,grid_u500,'x-',lw=2,c='red',  label='ReD1600_500')
ax.plot(u500_2,grid_u500,'x:',lw=2,c='red',  label='ReD1600_500')

lg=plt.legend(loc='best')
plt.show()
plt.close('all')

for ivar in range(1,8): 
    
    zy=np.fromfile('var{}.zy'.format(ivar),t_f8,nzy).reshape([ny,nz])
    zx=np.fromfile('var{}.zx'.format(ivar),t_f8,nzx).reshape([nx,nz])
    avg={}
    for d in ['zy','zx']:
        print(ivar,d)
        if d == 'zy':
            v=zy
        elif d == 'zx':
            v=zx
        else:
            print('ERROR')
            quit() 
        fig=plt.figure(figsize=(10,5))
        ax=fig.add_axes([0,0,1,1])
        if ivar < 4:
            vplot = np.log(v)
        else :
            vplot = v

        avg[d] = np.mean(vplot,0)
        plt.imshow(np.transpose(vplot)[:20,:20],origin='lower',aspect=1)
        plt.savefig('var{}_{}.pdf'.format(ivar,d),format='pdf')
        plt.close('all')
    print('ZX:', avg['zx'][:5])
    print('ZY:', avg['zy'][:5]) 


