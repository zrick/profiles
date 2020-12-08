#!/opt/local/bin/python

import numpy as np 


def write_header(h,nx,ny,nz,it,t,re,fr,sc):
    t_i4=np.dtype('<i4')
    t_f8=np.dtype('<f8')
    ipar=[52,nx,ny,nz,it]
    fpar=[t,re,fr,sc] 
    np.tofile(h,ipar,t_i4,count=5)
    np.tofile(h,fpar,t_f8,count=4)
    np.tofile(h,[52],t_i4,count=1)

nx=10
ny=512
nz=10


ih=open('delta_velo.dat','r')
ou=open('delta_u.bin','ab')
ov=open('delta_v.bin','ab') 
ow=open('delta_w.bin','ab') 

arr=np.zeros([3,ny,nx]) 

i=0;
for line in ih:
    dat=line.split()[3:5]
    arr[0,i,:] = dat[0]
    arr[1,i,:] = 0.
    arr[2,i,:] = dat[1] 
    i=i+1 




write_header(ou,0 )
