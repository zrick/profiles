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

        log_law=np.zeros(len(yp))
        log_law[1:] = np.log(yp[1:])/k + C
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

        p=p-1.20*C5/(re**2) * (z**2)
            
        return(1./z,p) 

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

        #print('PPLUS: us_loc, al_loc:',us_loc,al_loc, us,al)
        
        z=1/us_loc 
        dp_loc=re**2/(2*z**2)

        ym=np.array(yp/dp_loc)

        # OUTER REFERENCE PROFILES
        #
        argz=0.66*(2.*np.pi*(ym+0.12)) 
        dampA =  0.42*(us_loc/0.05- 0.07/re) #np.exp(zdum) * ( dU_mtc*np.cos(zdum) + dW_mtc*np.sin(zdum) )
        dampB =  0.0#np.exp(zdum) * ( dU_mtc*np.sin(zdum) + dW_mtc*np.cos(zdum) )

        outer_u= (1-(dampA*np.cos(argz)+dampB*np.sin(argz))*np.exp(-argz))/us_loc
        outer_w=   ( dampA*np.sin(argz)-dampB*np.cos(argz))*np.exp(-argz)/us_loc
        [ous,ows]=mp.rotate(outer_u,outer_w,al_loc)
        

        ################################################################################
        # STREAMWISE COMPONENT (SHEAR-ALGINED)
        ################################################################################

        u=np.zeros(len(yp))
        i_x=mp.geti(yp,self.LIMIT_INNER)

        # start with logarithmic law (everywhere)
        log_law=np.zeros(len(yp)) 
        log_law[1:]=np.log(yp[1:])/self.KAPPA + self.C 
        u[1:]=log_law[1:]
        u[0]=0

        # inner-layer correction [in inner units] 
        c2=-0.0000; c4=-0.0003825; c6=6.32e-6;
        scale=2.03
        u_ref = 1/0.07825
        ymatch = 19 
        wgt=np.zeros(len(yp)) 
        wgt[1:] = ( 0.5*(sp.special.erf(scale*(np.log(yp[1:]/ymatch)))+1) )
        u_visc=( yp + c4*yp**4+c6*yp**6) / (1 + c6/u_ref*yp**6)
        u[1:]=(1-wgt[1:])*u_visc[1:] + wgt[1:]*u[1:]
        u[0]=0 


        # outer-layer deficit [in outer units] 
        scale_trans=2 
        ctr= 0.29 - 70/re 
        arg=np.zeros(len(ym))
        arg[1:]=np.log(ym[1:]/ctr)
        delta_log=z * np.cos(al_loc)

        
        wgt=(sp.special.erf(scale_trans*arg)+1)/2

        fit2= ( wgt*(log_law-delta_log) ) # + 0*fit1 * (0.0497/us_loc)**2 )
        fit2= ( wgt*(log_law-ous) ) 
        u[1:]-= fit2[1:] #* diff_u/dev_u


        ################################################################################
        # NEW part for shear-spanwise velocity
        ################################################################################            
        k = self.KAPPA * 0.836 * (1+150/1600)# us_loc   the second part is a low-re correction that is dispensable for re_loc >~ 1000 
        w = -ows#wgt + (1-wgt)*log_w

        i0=mp.geti(yp,np.sqrt(dp_loc))
        i1=mp.geti(yp,1.5*np.sqrt(dp_loc))
        i2=mp.geti(yp,0.2*dp_loc)
        c_off=4.9e2*us_loc/np.sqrt(dp_loc)
        c1=2.7e3*us_loc/np.sqrt(dp_loc)/(np.log(yp[i0])**2.0)
        dir_fit=c1*np.log(yp[1:])**2+c_off
        slp=(dir_fit[i1+1]-dir_fit[i1])/(yp[i1+1]-yp[i1])
        off=dir_fit[i1]-slp*yp[i1]
        dir_fit[i1:]=slp*yp[i1:-1]+off
        
        i_03=mp.geti(yp,0.30*dp_loc)
        z_03=yp[i_03]
        s_03=(w[i_03+1]-w[i_03-1])/(yp[i_03+1]-yp[i_03-1])
        a_03=s_03*z_03
        o_03=w[i_03]-a_03*np.log(z_03)
        w[0]=0.
        w[1:i_03]=o_03+a_03*np.log(yp[1:i_03])

        #BLENDING (inner / outer)
        #blending height: geometric avg between yp[i_03] and yp[i0]
        ib=mp.geti(yp,0.12*dp_loc)
        scale=2.0
        wgt=np.zeros(len(yp)) 
        wgt[1:] = ( 0.5*(sp.special.erf(scale*(np.log(yp[1:]/yp[ib])))+1) )


        dir_fit[i_03:]=dir_fit[i_03]
        w_inner=np.zeros(len(yp))
        w_inner[0]=0. 
        w_inner[1:]=np.tan(dir_fit/180*np.pi)*u[1:]

        w=wgt*w + (1-wgt)*w_inner 

        
        return u,w

    def profile_log(self,yp,re):
        self.qinit()

        return 1./self.KAPPA * np.log(yp)  + self.C
        
    
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

        #print('PMINUS: us_loc, al_loc:',us_loc,al_loc, us,al)
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

