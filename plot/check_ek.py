#!/opt/local/bin/python
import matplotlib.pyplot as plt 
import numpy as np
from EkmanProfileClass import Complex, EkmanUniversalClass, build_grid


re=500
ny,yp,ym=build_grid(1500,2)

sc=EkmanUniversalClass()

up,wp=sc.profile_plus(yp,re)

fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.8]) 
ax.plot(yp,wp)
ax.plot(yp,up)
ax.set_xscale('log')
plt.show()
plt.close('all')

quit() 

