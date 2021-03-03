#!/opt/local/bin/python 
import my_pylib as mp 
import numpy as np
import scipy as sp 
from scipy.special import lambertw as prodlog

def build_grid(trgt,thick=1) :
    vert_stretch=1.05   #grid stretching
    dyloc=1             #grid resolution at wall
    ny=1
    yloc=dyloc
    while yloc < thick*trgt:
        dyloc*=vert_stretch 
        yloc+=dyloc
        ny+=1
        
    yp=np.zeros(ny)

    dyp=1
    yp[0]=0
    for i in range(ny-1):
        yp[i+1]=yp[i]+dyp
        dyp*=vert_stretch 
    ym=yp/trgt

    return ny,yp,ym


class Complex:
    def __init__(self, real, imag=0.0):
        self.r = real
        self.i = imag

    def __add__(self, other):
        return Complex(self.r + other.r,
                       self.i + other.i)

    def __sub__(self, other):
        return Complex(self.r - other.r,
                       self.i - other.i)

    def __mul__(self, other):
        return Complex(self.r*other.r - self.i*other.i,
                       self.i*other.r + self.r*other.i)

    def __div__(self, other):
        sr, si, ot, oi = self.r, self.i,other.r, other.i # short forms
        r = float(ot**2 + oi**2)
        return Complex((sr*ot+si*oi)/r, (si*ot-sr*oi)/r)

    def __abs__(self):
        return sqrt(self.r**2 + self.i**2)

    def __neg__(self):   # defines -c (c is Complex)
        return Complex(-self.r, -self.i)

    def __eq__(self, other):
        return self.r == other.r and self.i == other.i

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __str__(self):
        return '(%g + %g i)' % (self.r, self.i)

    def __repr__(self):
        return 'Complex' + str(self)
    
class EkmanUniversalClass:
    
    def __init__(self):
        #initialized based on hard-coded constants (no data needed)

        self.LIMIT_INNER=50
        self.LIMIT_OUTER=0.10
        
        self.KAPPA=0.416 # k
        self.C = 5.4605 # C
        self.A = Complex(4.79823,-5.56645)
        self.C5= Complex(-57.7728,19.5046)
        self.initialized=True
        return
    
    def estimate(self,yp,up,wp,deltap,us,al,plot=False):
        #re-estimate constants from given profile at fixed deltap
        
        ym=yp/deltap
        [um,wm] = [up*us,wp*us]
        
        i1=mp.geti(yp,self.LIMIT_INNER)-1
        i2=mp.geti(yp,self.LIMIT_OUTER*deltap)-1
        
        k = np.log(yp[i2]/yp[i1])/(up[i2]-up[i1])
        C = up[i1] - np.log(yp[i1])/k
        ny=len(yp)
        self.KAPPA = k
        self.C = C

        #log_law=np.zeros(len(yp))
        log_law = self.profile_log(yp,re) # np.log(yp[1:])/k + C
        log_dev_outer_u=up-log_law
        log_dev_outer_u[:i2]=0.
        log_dev_outer_w=wm
        
        ud_align = up - np.cos(al)/us
        wd_align = wp + np.sin(al)/us
        
        f1=Complex(ud_align,wd_align) 

        # self.outer_wp_itp=spip.interp1d(ym,wp*us)
        

        imax=mp.geti(yp,50)
        f5_x=np.zeros(len(up))
        f5_x[1:imax]=up[1:imax] - ( np.log(yp[1:imax])/k + C )
        # #let f5_x @ 0 thus that du*+/dz+=1 
        f5_x[imax:]=0.
        f5_z= wp
        f5_z[imax:]=0. 
        f5=Complex(f5_x,f5_z) 
        f1_high=Complex(f1.r-f5.r,f1.i) 

        #self.outer_ud_itp=spip.interp1d(ym,log_dev_outer_u,'cubic')
        #self.outer_wd_itp=spip.interp1d(ym,log_dev_outer_w,'cubic')
        # self.inner_u_itp=spip.interp1d(yp,up,'cubic')

        #estimate intercept of f1 - log-law
        # Real part from f1_high u component at first grid point
        #     ud_align0 - log(ym0)/KAPPA 
        A_r=f1_high.r[1]-1./self.KAPPA * np.log(yp[1]/deltap)
        # Imaginary part from f1 w component at LIMIT_INNER
        #     w_align @ LIMIT_INNER 
        A_i=np.average(f1.i[i1-2:i1+2])
        self.A=Complex(A_r,A_i) 
        self.yp=yp
        #print(self.A) 
        # integrate constants C1 and C5 for higher-order theory
        # C1 = int_0^{\infty} f_1(y^- )dy^-
        C1_r=mp.integrate(yp/deltap,f1.r,len(yp))
        C1_i=mp.integrate(yp/deltap,f1.i,len(yp)) 
        self.C1=Complex(C1_r,C1_i) 

        # C5 = int_0^{50} f_5(y+)dy^+
        imax=mp.geti(yp,50)
        self.C5=Complex(mp.integrate(yp,f5.r,imax),mp.integrate(yp,f5.i,imax))
        #print(self.C5) 

        self.initialized=True

        return 

    def ustar_alpha(self,re):
        self.qinit()

        Ai=-self.A.i
        C=self.C
        Ar=self.A.r 
        k=self.KAPPA 
        C5=self.C5.r
        k=0.415
        z=10

        #Ai=Ai+0.285
        Ai=Ai+0.23
        CD=C-Ar - 0.0

        # print('SPALART (1989) THEORY -- USING CONSTANTS: Ai=',Ai, 'C-Ar =',CD) 
        
        i=0
        while True and i < 10 :
            p=np.arcsin(Ai/z)
            z_old=z
            al=np.cos(p)
            bt=2./k
            gm=2./k*np.log(re)-1./k*np.log(2)+CD
            z=np.real(bt*prodlog(al*np.exp(gm/bt)/bt)/al)
            
            if np.abs(z_old -z) < 1e-15:
                break;
            i=i+1

        p=p-1.00*C5/(re**2) * (z**2)
            
        return(1./z,p) 

    def profile_log(self,yp,re):
        self.qinit()

        x=np.zeros(len(yp))
        x[1:] = np.log(yp[1:])/self.KAPPA  + self.C
        return x 

    def erf_transition(self,w,l,x):
        return 0.5*(sp.special.erf(w*np.log(x/l))+1)
    
    def profile_plus(self,yp,re,us=-1,al=-1):
        self.qinit()

        if us == -1 and al == -1:
            us_loc,al_loc=self.ustar_alpha(re)
        elif al == -1:
            us_dum,al_loc=self.ustar_alpha(re)
            us_loc=us
        elif us == -1:
            us_loc,al_dum=self.ustar_alpha(re)
            al_loc=al
        else:
            us_loc,al_loc=us,al

        u=np.zeros(len(yp))
        w=np.zeros(len(yp)) 
        
        z=1/us_loc 
        dp_loc=re**2/(2*z**2)
        ym=np.array(yp/dp_loc)
        trans_scale=2.0

        ################################################################################
        # 2-COMPONENT OUTER REFERENCE PROFILES
        argz=    0.66*(2.*np.pi*(ym+0.12)) 
        dampA =  8.4*us_loc       # - 150./re) #np.exp(zdum) * ( dU_mtc*np.cos(zdum) + dW_mtc*np.sin(zdum) )
        #dampB= 0. 
        outer_u= (1-(dampA*np.cos(argz))*np.exp(-argz))*z  #-dampB*np.cos(argz)*np.exp(-argz)*z
        outer_w=   ( dampA*np.sin(argz))*np.exp(-argz)*z   #-dampB*np.cos(argz)*np.exp(-argz)*z
        [ous,ows]=mp.rotate(outer_u,outer_w,al_loc)
        
        ################################################################################
        # STREAMWISE COMPONENT (shear-aligned) 
        log_law=self.profile_log(yp,re) # np.log(yp)/self.KAPPA + self.C   

        # inner-layer (empirical profile for viscous and buffer layer) 
        c4=-0.0003825; c6=6.32e-6; u_ref = 0.07825
        yp_mtc = 19 
        wgt_i = self.erf_transition(trans_scale,yp_mtc,yp[1:]) 
        u_visc=( yp + c4*yp**4+c6*yp**6) / (1 + u_ref*c6*yp**6)
        u[1:]=(1-wgt_i)*u_visc[1:] + wgt_i*log_law[1:]

        # outer-layer deficit
        ctr= 0.30 - 120/re 
        wgt_o=self.erf_transition(trans_scale,ctr,ym[1:]) #(sp.special.erf(trans_scale*np.log(ym[1:]/ctr))+1)/2 #starts at index 1! 
        u[1:] -= ( wgt_o*(log_law[1:]-ous[1:]) )

        ################################################################################
        # SPANWISE COMPONENT (in-plane orthogonal to shear)
        # inner region - empirical profile
        #   - based on direction below y+~10
        i1=mp.geti(yp,10)
        c_off=40/(dp_loc*us_loc) ##4.0e2*us_loc/np.sqrt(dp_loc)
        c1=26./(dp_loc*us_loc)
        dir_fit=c1*np.log(yp[1:i1+2])**2+c_off
        w[1:i1+2]=np.tan(dir_fit/180*np.pi)*u[1:i1+2]
        # inner-outer transition 
        #   - match gradient and value at y+~10 (index i0) 
        #   - match value at y-=0.27
        match_height=0.27
        i2=mp.geti(yp,match_height*dp_loc) 
        C1=(w[i1+1]-w[i1-1])/(yp[i1+1]-yp[i1-1])
        ap = ( -ows[i2] - w[i1] - C1 * (yp[i2]-yp[i1]) ) / ( np.log(yp[i2]/yp[i1]) - yp[i2]/yp[i1]) 
        w[i1:] = w[i1] + ap * np.log(yp[i1:]/yp[i1]) + (C1-ap/yp[i1])*(yp[i1:]-yp[i1])
        # Blend inner and outer profile
        #   - matching height same as above 
        wgt = ( 0.5*(sp.special.erf(trans_scale*(np.log(yp[1:]/yp[i2])))+1) )
        w[1:]= (1-wgt)*w[1:] + wgt*(-ows[1:]) 
        
        return u,w

    
    def profile_minus(self,ym,re,us=-1,al=-1): 
        self.qinit()

        if us == -1 and al == -1:
            us_loc,al_loc=self.ustar_alpha(re)
        elif al == -1:
            us_dum,al_loc=self.ustar_alpha(re)
            us_loc=us
        elif us == -1:
            us_loc,al_dum=self.ustar_alpha(re)
            al_loc=al 
        else: 
            us_loc,al_loc=us,al

        nu_loc=2./re**2
        dp_loc=us_loc**2/nu_loc
        u,w = self.profile_plus(ym*dp_loc,re,us=us_loc,al=al_loc)
        return u*us_loc,w*us_loc
        
    def profile_deficit(self,ym,re):
        self.qinit()

        us,al=self.ustar_alpha(re) 
        z=1/us

        deltap=re*re/(2*z*z)
        yp=ym*deltap

        # outer layer 
        ud=self.f1_r_itp(ym)# + ( self.f5_r_itp(yp) if yp <50 else 0 ) 
        wd=self.f1_i_itp(ym)

        # correction for viscous layer below yp~50 
        i_x=mp.geti(yp,50)
        ud[:i_x] = ud[:i_x] + self.f5_r_itp(yp[:i_x])

        # fix surface
        return yp,ud,wd

    def qinit(self):
        if self.initialized == False:
            raise Exception("Class not initialized")
        else:
            return 

