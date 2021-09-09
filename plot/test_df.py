#!/opt/local/bin/python
import numpy as np
import my_pylib as mp 
us_loc=0.0619
dp_loc=479

yp=np.arange(0,10.5,0.25) 

c_off=40/(dp_loc*us_loc)
c1=26/(dp_loc*us_loc)
dir_fit = c1*np.log(yp)**2+c_off

a_off=0 #2.84/(dp_loc*us_loc)
a_sqr=24.66/(dp_loc*us_loc)
a_lin=10.24/(dp_loc*us_loc) 
dir_fit2= a_off + a_sqr*np.sqrt(yp) + a_lin*yp


for i in range(11):
    ix=mp.geti(yp,i)
    print(yp[ix],dir_fit[ix],dir_fit2[ix]) 
