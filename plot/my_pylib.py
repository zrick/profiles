#!/opt/local/bin/python 

from numpy import zeros

def read_fortran_record(f_h,t): 
   t_i4=np.dtype('<i4') 
   dum1,=np.fromfile(f_h,t_i4,count=1)  
   n=int(dum1/t.itemsize) 
   rec=np.fromfile(f_h,t,count=n) 
   dum2=np.fromfile(f_h,t_i4,count=1) 
   
   if dum1 == dum2 : 
      return [n,rec]
   else :  
      print('ERROR READING RECORD',dum1,dum2) 
      quit() 


class DnsGrid : 
   def __init__(self,fname): 
      f_h=open(fname,'rb') 
      t_i4=np.dtype('<i4') 
      t_f8=np.dtype('<f8') 

      [n_isize,rec_isize]=read_fortran_record(f_h,t_i4) 
      [n_fsize,rec_fsize]=read_fortran_record(f_h,t_f8) 

      self.nx=rec_isize[0];  self.lx=rec_fsize[0]
      self.ny=rec_isize[1];  self.ly=rec_fsize[1]
      self.nz=rec_isize[2];  self.lz=rec_fsize[2] 

      [nx_test,self.x]=read_fortran_record(f_h,t_f8)  
      [ny_test,self.y]=read_fortran_record(f_h,t_f8) 
      [nz_test,self.z]=read_fortran_record(f_h,t_f8) 

      if not ( nx_test == self.nx and ny_test == self.ny and nz_test == self.nz ): 
         print('Grid Dimensions do not match') 
         quit() 

def cumsum(x,y,n) : 
   res=np.zeros(n)
   for i in range(n-1):   
      res[i+1] = res[i] + y[i]*(x[i]-x[i-1]) 
      
   return res; 

def rotate(a,b,alpha) : 
   from math import sin,cos 
   c= cos(alpha)*a + sin(alpha)*b 
   d=-sin(alpha)*a + cos(alpha)*b

   return [c,d] 

def rotate_tensor(a,alpha) :
   from math import sin,cos  
   #check shape 
   if ( a.shape != (3,3) ) : 
      print ('my_pylib: ERROR Shape {} not supported'.format(a.shape)) 
      quit()
   c=cos(alpha) 
   s=sin(alpha) 
   [m,c2,s2] = [c*s, c*c, s*s] 
   b=np.zeros([3,3])  

   
   
   b[0,0] = c2*a[0,0] + s2*a[2,2] + 2*m*a[0,2]  
   b[0,1] = c* a[0,1] + s*a[1,2] 
   b[0,2] = m*(a[2,2]-a[0,0]) +a[0,2]*(c2-s2) 
   b[1,0] = b[0,1] 
   b[1,1] = a[1,1] 
   b[1,2] =-s*a[0,1]  + c*a[1,2] 
   b[2,0] = b[0,2] 
   b[2,1] = b[1,2]
   b[2,2] = s2*a[0,0] + c2*a[2,2] - 2*m*a[0,2] 
   
   return b
   
def geti(arr,val): 
    for i in range(len(arr)): 
        if ( arr[i] >= val ): 
            break
    return i 

def smooth(x,y,n,s):  
    xo =zeros(n-(2*s+1))
    smt=zeros(n-(2*s+1))
    for i in range(s,n-(s+1)): 
        for ss in range(-s,s+1): 
            smt[i-s] = smt[i-s] + y[i+ss]
            xo[i-s]  = xo[i-s]  + x[i+ss]
    smt=smt/(2*s+1)
    xo =xo/ (2*s+1)

    return [xo,smt]


import numpy as np
def smooth_new(a,ns):
    n = len(a) 
    a_s = np.zeros(n) 
    if ns > n-1: 
        ns = n-1
    # print 2*ns+1, range(-ns,ns+1) 
    for i in range(ns):  
        for j in range(i+ns+1): 
            a_s[i] += a_s[j]/float(i+ns+1)
    
    for i in range(ns,n-ns):  
        for j in range(-ns,ns+1):
            a_s[i] += a[i+j]/float(2*ns+1)

    for i in range(n-ns,n): 
        for j in range(i-ns,n): 
            a_s[i] += a[j]/float(n-i+ns) 

    return a_s 

def integrate (x,y,n):
    sum=0
    for i in range(1,n): 
       sum= sum+ (x[i]-x[i-1])*(y[i] + y[i-1])/2 
    return sum
def derivative1 ( x,y,n,m=1 ):
   import numpy 
   der=numpy.zeros([n,m]) 
   
   der[0] = (y[1]-y[0])/(x[1]-x[0]) 
   for idum in range(n-2): 
      i=idum+1
      der[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1]) 
   der[n-1]=(y[n-1]-y[n-2])/(x[n-1]-x[n-2])     
   return der;

def derivative2 ( x,y,n ):
   import numpy
   der=numpy.zeros(n) 
   
   der[0] = 4*(y[2]-2*y[1]-y[0])/(x[2]-x[0])**2 
   for idum in range(n-2): 
      i=idum+1
      der[i] = 4*(y[i+1]+y[i-1]-2*y[i])/(x[i+1]-x[i-1])**2 
   der[n-1]=4*(y[n-1]-2*y[n-2]+y[n-3])/(x[n-1]-x[n-3])**2     
   return der; 
