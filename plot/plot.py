#!/opt/local/bin/python
import my_pylib as mp 
import numpy as np
import scipy.interpolate as spip
import argparse 
import matplotlib.pyplot as plt
import sys
import warnings
from scipy.special import lambertw as prodlog
import scipy as sp
from netCDF4 import Dataset
from math import erf

from EkmanProfileClass import Complex, EkmanUniversalClass, build_grid

D2R = np.pi / 180. 

use_data=False   # if true -- the netcdf files from DNS are needed 
plot_ustar_alpha=True
plot_summary=True
plot_visc=True
print_table=True
plot_outer_log=True
plot_total_rot=True
plot_convergence_1000=False

colors = {400 : 'gray',
          500 : 'pink',
          750 : 'cyan',
          1000: 'blue',
          1300: 'red',
          1600: 'black',
          5000: 'black',
          10000:'gray',
          30000:'black',
          150000:'black',
          300000:'black'} 

base='/Users/zrick/WORK/research_projects/SIM_NEUTRAL/netcdf/' 

files = { 400: base+'avg_flw_ri0.0_re400.0_1024x384x1024.nc',
          500: base+'avg_flw_ri0.0_re500.0_2048x192x2048.nc',
          750: base+'avg_flw_ri0.0_re750.0_3072x384x3072.nc',
          1000:base+'avg_flw_ri0.0_re1000.0_co0.0_1536x512x3072.nc',
          1300:base+'avg_flw_ri0.0_re1300.0_co0_2560x640x5120.nc',
          1600:base+'avg_flw_ri0.0_re1600.0_co0_3840x960x7680.nc' } 

sc=EkmanUniversalClass() # yp_ref,up_ref,wp_ref,deltap_ref,us_ref,al_ref*D2R-geo_rotate,plot=False)

if use_data: 
    re_ref=1600
    nu_ref=2./(re_ref**2)

    ncf='/Users/zrick/WORK/research_projects/SIM_NEUTRAL/netcdf/avg_flw_ri0.0_re1600.0_co0_3840x960x7680.nc' 
    nch=Dataset(ncf,'r')
    
    t_start=9.7
    it_min=mp.geti(nch['t'][:],t_start)
    print('START TIME: ',nch['t'][it_min], 'END TIME:', nch['t'][-1] ) 
    
    us_ref=np.mean(nch['FrictionVelocity'][it_min:])
    al_ref=np.mean(nch['FrictionAngle'][it_min:])
    
    
    y_ref=nch['y'][:]
    yp_ref=y_ref*us_ref/nu_ref  
    ym_ref=y_ref/us_ref
    
    u_ref=np.mean(nch['rU'][it_min:,:],0)
    w_ref=np.mean(nch['rW'][it_min:,:],0)
    
    up_ref=u_ref/us_ref
    wp_ref=w_ref/us_ref
    deltap_ref=us_ref*us_ref/nu_ref
    geo_rotate= np.arctan(w_ref[-1]/u_ref[-1])
    
    [up_ref,wp_ref] = mp.rotate(up_ref,wp_ref,al_ref/180*np.pi)
    [um_ref,wm_ref] = mp.rotate(u_ref,w_ref,geo_rotate) 
    wm_ref=-wm_ref

    #print('GEO DIRECTION:        ', geo_rotate)
    #print('SHEAR ALIGNMENT ANGLE:', al_ref)

    sc.estimate(yp_ref,up_ref,wp_ref,deltap_ref,us_ref,al_ref*D2R-geo_rotate,plot=False)

if plot_visc:
    fig=plt.figure(figsize=(5,4))
    ax=fig.add_axes([0.1,0.1,0.87,0.8])

    fig2=plt.figure(figsize=(5,4))
    ax2=fig2.add_axes([0.13,0.1,0.85,0.87]) 
    
    for re in [1600,1300,1000,750,500]:
        c=colors[re]
        nu=2./re**2
        us,al=sc.ustar_alpha(re)
        dp=us**2/nu
        ny,yp,ym=build_grid(dp,5.5)

        up,wp=sc.profile_plus(yp,re)
        um,wm=sc.profile_minus(yp,re)
        f=Dataset(files[re],'r')
        t_use = -1
            
        u_dat=f['rU'][t_use,:]
        w_dat=f['rW'][t_use,:]
        y_dat=f['y'][:]
        us_dat=f['FrictionVelocity'][t_use] 
        al_geo = np.arctan(w_dat[-1]/u_dat[-1])
        al_sfc1 = f['FrictionAngle'][t_use]*D2R
        al_sfc2 = np.arctan(w_dat[1]/u_dat[1]) 
            
        [u_sfc,w_sfc] = mp.rotate(u_dat,w_dat,al_sfc1) 
        yp_dat = y_dat*us_dat/nu
        ym_dat = y_dat/us_dat

        ax.plot(yp_dat,u_sfc/us_dat, c=c,lw=3,ls='-',label=r'$Re={}$'.format(re),alpha=0.5)
        ax.plot(yp,up,               c=c,lw=3,ls=':')

        if  re ==1600:
            ax2.plot(ym_dat,(u_sfc-u_sfc[-1])/us_dat,c=c,lw=3,ls='-',label=r'$Re={}$ (data)'.format(re),alpha=0.5)
            ax2.plot(ym_dat,(w_sfc-w_sfc[-1])/us_dat,c=c,lw=3,ls=':',alpha=0.5)
            ax2.plot(ym,(up-up[-1]),c=c,lw=1,ls='--',label=r'$Re={}$ (model)'.format(re))
            ax2.plot(ym,(wp-wp[-1]),c=c,lw=1,ls=':',label=r'$Re={}$ (model)'.format(re))
        else:
            ax2.plot(ym_dat,(u_sfc-u_sfc[-1])/us_dat,c=c,lw=3,ls='-',label=r'$Re={}$'.format(re),alpha=0.5)
            ax2.plot(ym_dat,(w_sfc-w_sfc[-1])/us_dat,c=c,lw=3,ls=':',alpha=0.5)
            ax2.plot(ym,(up-up[-1]),c=c,lw=1,ls='--')
            ax2.plot(ym,(wp-wp[-1]),c=c,lw=1,ls=':')


            

    ulog=sc.profile_log(yp,re) 
    ax.plot(yp[5:],ulog[5:],ls='--',lw=1,c='black',alpha=1.0,label=r'$U^+=\kappa^{-1}log(z^+) + C$')
    ax.plot(yp[:20],yp[:20],  ls=':',lw=1,c='black',alpha=1.0,label=r'$U^+=z^+$')
    ax.plot(yp,  ( yp -0.0003825*(yp**4) + 0.00000632 * (yp**6) ) / (1 + 0.00000632 * 0.07825 * (yp **6) ),lw=1,ls=':',c='blue',label=r'$U_{visc}$')
    ax.set_xlim(0,50)
    ax.set_ylim(0,15)
    ax.set_title(r'$\kappa=0.416; C=5.4605$') 

    lg=plt.legend()
    lg.get_frame().set_linewidth(0.0)
    ax.set_xlabel(r'$z^+$')
    ax.set_ylabel(r'$U^+$') 
    fig.savefig('visc_layer.pdf',format='pdf' )
    plt.close('all')

    
    ax2.set_xlim(0,1.5)
    ax2.set_ylim(-0.5,3.2)
    ax2.set_xlabel(r'$z^-$')
    ax2.set_ylabel(r'$(U^\alpha - G^\alpha)/u_\star$')
    lg2=ax2.legend(loc='best')
    lg2.get_frame().set_linewidth(0.0) 
    fig2.savefig('outer_layer.pdf',format='pdf')
    
        

if print_table : 

    print('#  Re_D  f[1/s] nu[m2/s] G[m/s] |   re_tau u*/G[1]   u*[m/s] alpha[deg] delta[m] Lambda[km] +unit[mm]') 
    for re_loc in [500,1000,2000,5000,1e4,2e4,4e4,8e4,9.5e4,1.5e5,3e5,1e6]:
        relambda_loc=re_loc**2/2
        f_dim=1e-4     #1/s
        nu_dim=1.5e-05 #m2/s
        
        us_loc,al_loc=sc.ustar_alpha(re_loc)
        G_dim=np.sqrt(relambda_loc*f_dim*nu_dim) #m/s
        us_dim=us_loc*G_dim                      #m/s
        delta_dim=us_dim/f_dim                   #m
        wall_dim=nu_dim/us_dim                   #m
        print('{0:7.0f} {1:7.2g} {2:8.3g} {3:6.3f} | {4:8.3g} {5:7.4f} {6:9.4f}  {7:9.2f} {8:8.1f}  {9:9.1f}   {10:7.3g}'.format(re_loc,f_dim,nu_dim,G_dim,delta_dim/wall_dim,us_loc,us_dim,al_loc/D2R,us_dim/f_dim,G_dim/f_dim/1e3,wall_dim*1e3))

if plot_outer_log : 

    fig1=plt.figure(figsize=(12,5))
    ax1=fig1.add_axes([0.05,0.1,0.4,0.8])
    ax2=fig1.add_axes([0.55,0.1,0.4,0.8])
    ax5=fig1.add_axes([0.09,0.455,0.18,0.43]) 

    fig2=plt.figure(figsize=(12,5))
    ax3=fig2.add_axes([0.05,0.1,0.4,0.8])
    ax4=fig2.add_axes([0.55,0.1,0.4,0.8])
    ax6=fig2.add_axes([0.26,0.185,0.18,0.37]) 

    for re in [500,750,1000,1300,1600,10000,150000]:
        c=colors[re]
        nu=2./re**2
        us,al=sc.ustar_alpha(re)
        dp= us**2 / nu
        ny,yp,ym=build_grid(dp,5)

        print(re,nu,us,al,dp,ny)
        
        um,wm=sc.profile_minus(ym,re)
        up,wp=sc.profile_plus(yp,re)
        
        ax1.plot(yp, wp,        c=c,ls='-',lw=2,label=r'$Re_D={}$'.format(re)) 
        ax2.plot(ym,(wp-wp[-1]),c=c,ls='-',lw=2)
        ax5.plot(yp, wp,        c=c,ls='-',lw=2) 

        
        ax3.plot(yp,up,         c=c,ls='-',lw=2,label=r'$Re_D={}$'.format(re))
        ax4.plot(ym,up-up[-1],  c=c,ls='-',lw=2,label=r'$Re_D={}$'.format(re))
        ax6.plot(yp,up,         c=c,ls='-',lw=1) 
        
        i1=mp.geti(ym,0.001)
        ax1.scatter(yp[i1],wp[i1],marker='D',c=c,s=20)

        if  re > 1600 :
            continue; 
        
        f=Dataset(files[re],'r')
        t_use = -1
            
        u_dat=f['rU'][t_use,:]
        w_dat=f['rW'][t_use,:]
        y_dat=f['y'][:]
        us_dat=f['FrictionVelocity'][t_use] 
        al_geo = np.arctan(w_dat[-1]/u_dat[-1])
        al_sfc1 = f['FrictionAngle'][t_use]*D2R
        al_sfc2 = np.arctan(w_dat[1]/u_dat[1]) 
            
        [u_geo,w_geo] = mp.rotate(u_dat,w_dat,al_geo)
        [u_sfc,w_sfc] = mp.rotate(u_dat,w_dat,al_sfc1) 
        yp_dat = y_dat*us_dat/nu
        ym_dat = y_dat/us_dat

        ax1.plot(yp_dat,w_sfc/us_dat,c=c,ls='--',lw=1) 
        ax2.plot(ym_dat,(w_sfc-w_sfc[-1])/us_dat,c=c,ls='--',lw=1)
        ax5.plot(yp_dat,w_sfc/us_dat,c=c,ls='--',lw=1)

        ax3.plot(yp_dat,u_sfc/us_dat,c=c,ls='--',lw=1)
        ax4.plot(ym_dat,(u_sfc-u_sfc[-1])/us_dat,c=c,ls='--',lw=1)
        ax6.plot(yp_dat,u_sfc/us_dat,c=c,ls='--',lw=1) 
        
    ax2.plot([1e-3,1e2],[0,0],lw=0.5,ls='-',c='black')

    ax1.set_xscale('log')
    ax1.set_xlim(1,3e2)
    ax1.set_ylim(0,5)
    ax1.set_xlabel(r'$z^+$')
    ax1.set_ylabel(r'$\left(V^{\alpha}\right)^+$')
    
    ax2.set_xscale('log')
    ax2.set_xlim(1e-2,2e0)
    ax2.set_xlabel(r'$z^-$')
    ax2.set_ylabel(r'$V^{\alpha+}-V^{\alpha+}_G$')

    ax5.set_xscale('linear')
    ax5.set_xlim(0,10)
    ax5.set_ylim(-0.1,0.5)
    ax5.set_xlabel(r'$z^+$')
    ax5.set_ylabel(r'$V^{\alpha+}$') 

    ax3.set_xscale('log')
    ax3.set_xlim(1,1e3)
    ax3.set_ylim(0,22)
    ax3.set_xlabel(r'$z^+$')
    ax3.set_ylabel(r'$U^{\alpha +}$') 

    ax4.plot([0,2],[0,0],ls=':',lw=1,c='black') 
    ax4.set_xlim(0,2)
    ax4.set_ylim(-15,2) 
    ax4.set_xlabel(r'$z^-$')
    ax4.set_ylabel(r'$(U^{\alpha+}-G^{\alpha+})/u_\star$')

    ax6.set_xscale('linear')
    ax6.set_xlim(0,10)
    ax6.set_ylim(0,10)
    ax6.set_xlabel(r'$z^+$')
    ax6.set_ylabel(r'$U^{\alpha+}$')
    
    lg1=ax1.legend(loc='best')
    lg1.get_frame().set_linewidth(0.0) 

    lg2=ax4.legend(loc='best')
    lg2.get_frame().set_linewidth(0.0)


    fig1.savefig('w_profile.pdf',format='pdf')
    fig2.savefig('u_profile.pdf',format='pdf')


    fig1.clf()
    fig2.clf() 
        
if plot_summary : 

    fig=plt.figure(figsize=(13,5))
    ax1=fig.add_axes([0.06,0.09,0.27,0.8])
    ax2=fig.add_axes([0.39,0.09,0.27,0.8])
    ax3=fig.add_axes([0.72,0.09,0.27,0.8])


    for re_loc in [500,750,1000,1300,1600,5000,30000,150000,300000]:#,300000,6e6]:#,5000,10000,20000,40000,80000,160000]:
        c_loc=colors[re_loc]
        nu_loc=2./re_loc**2
        us_loc,al_loc=sc.ustar_alpha(re_loc)
        deltap_loc = us_loc**2/nu_loc
        ny_loc,yp_loc,ym_loc=build_grid(deltap_loc,5)
        
        um_loc,wm_loc=sc.profile_minus(ym_loc,re_loc)
        up_loc,wp_loc=sc.profile_plus(yp_loc,re_loc)
        
        sqdp = 1.#/(deltap_loc)
        
        alpha_loc= re_loc/1601 % 1 
        
        match_i=0.10
        match_o=0.40

        if re_loc <= 1600 and use_data:
            # read data from file
            f=Dataset(files[re_loc],'r')
            t_use = -1

            u_dat=f['rU'][t_use,:]
            w_dat=f['rW'][t_use,:]
            y_dat=f['y'][:]
            us_dat=f['FrictionVelocity'][t_use] 
            al_geo = np.arctan(w_dat[-1]/u_dat[-1])
            al_sfc1 = f['FrictionAngle'][t_use]*D2R
            al_sfc2 = np.arctan(w_dat[1]/u_dat[1]) 
            
            [u_geo,w_geo] = mp.rotate(u_dat,w_dat,al_geo)
            [u_sfc,w_sfc] = mp.rotate(u_dat,w_dat,al_sfc1) 
            yp_dat = y_dat*us_dat/nu_loc
            ym_dat = y_dat/us_dat 

            i_o=mp.geti(yp_dat,match_o*deltap_loc)
            ax2.scatter(yp_dat[i_o]*sqdp,(u_sfc[i_o])/us_dat,c='orange')

            i_i=mp.geti(yp_dat,match_i*deltap_loc)
            ax2.scatter(yp_dat[i_i]*sqdp,(u_sfc[i_i])/us_dat,c='black')


        ax2.text(yp_loc[-1]*sqdp, up_loc[-1], 'Re='+str(re_loc),c=c_loc)

        ax1.plot(ym_loc[1:], (um_loc[1:]-um_loc[-1])/us_loc, c=c_loc, lw=2,ls='-',alpha=alpha_loc)
        ax1.plot(ym_loc[1:], (wm_loc[1:]-wm_loc[-1])/us_loc, c=c_loc, lw=2,ls='-',alpha=alpha_loc)

        if use_data and re_loc < 1600 : 
            ax1.plot(ym_dat[1:], (u_sfc[1:]-u_sfc[-1])/us_dat,   c=c_loc, lw=0.5,ls='--',alpha=alpha_loc)
            ax1.plot(ym_dat[1:], (w_sfc[1:]-w_sfc[-1])/us_dat,   c=c_loc, lw=0.5,ls='--',alpha=alpha_loc)
            veer = -np.arctan(w_sfc/u_sfc)/D2R
            ax1.plot(ym_dat, -10*veer/veer[-1],c='black',lw=1,ls='-',alpha=alpha_loc)
            il=mp.geti(yp_dat,18)-1

            ax3.plot(u_geo,-w_geo,ls='--',c=c_loc,lw=0.5,alpha=alpha_loc)
        
        umr_loc,wmr_loc=mp.rotate(um_loc,-wm_loc,-al_loc) 

        ax3.plot(umr_loc,wmr_loc,alpha=alpha_loc,c=c_loc)
    
        i_loc = mp.geti(wmr_loc,np.amax(wmr_loc))
        ax3.text(umr_loc[i_loc],wmr_loc[i_loc],'Re='+str(re_loc),c=c_loc)
        yp01 = 0.08 *us_loc**2 / nu_loc

        ax2.plot(yp_loc[1:]*sqdp, up_loc[1:], c='blue',lw=2,ls='-',alpha=alpha_loc)
        ax2.plot(yp_loc[1:]*sqdp, wp_loc[1:], c='red',lw=2,ls='-',alpha=alpha_loc)

        if use_data and re_loc < 1600: 
            ax2.plot(yp_dat[1:]*sqdp, u_sfc[1:]/us_dat,ls='--',alpha=alpha_loc,lw=1,c=c_loc)
            ax2.plot(yp_dat[1:]*sqdp, w_sfc[1:]/us_dat,ls='--',alpha=alpha_loc,lw=1,c=c_loc)



    ax2.plot([0.01,10000],[0,0],c='black',lw=0.5,ls='-')
    ax1.plot([0,2],[0,0],ls=':',c='black',lw=1)


    ax1.set_xscale('log')
    ax1.set_xlim(2e-4,2e0 )
    ax1.set_ylim(-20,1.5)
    ax1.set_xlabel(r'$y^-=y/\delta$')
    ax1.set_ylabel(r'$(U_i^-G_i)/u_\star$')
    ax1.set_title('Shear-aligned velocity deficit')
    
    ax2.set_xscale('log')
    ax2.set_ylim(0,2.7e1)
    ax2.set_xlim(1,3e4)
    ax2.set_xlabel(r'$y^+=y u_\star/\nu$')
    ax2.set_ylabel(r'$U_i^+=U_i/u_\star$')
    ax2.set_title('Shear-aligned profiles in inner scaling') 
    
    ax3.set_xlabel(r'$U^-=U/G$')
    ax3.set_ylabel(r'$W^-=W/G$')
    ax3.set_xlim(0,1.1)
    ax3.set_ylim(-0.02,0.27) 
    ax3.set_title('Geostrophy-aligned Hodographs')
    
    plt.savefig('ekman_profiles.pdf',format='pdf') 
    plt.close('all') 

if plot_ustar_alpha: 
    re_sim = np.array([   400,    500,    750,   1000,   1300,   1600])
    #   OLD VALUES
    us_sim = np.array([0.0647, 0.0618, 0.0561, 0.0520, 0.0502, 0.0479])
    al_sim = np.array([ 29.97,  25.47,  21.00,  18.64,  17.93,  17.48])
    #   NEW VALUES
    us_sim = np.array([0.0647, 0.0618, 0.0561, 0.0530, 0.0502, 0.0480])
    al_sim = np.array([ 28.21,  25.47,  21.00,  18.94,  17.93,  17.49])
    re_arr=10**np.arange(2.5,7,0.25)
    us_arr=np.zeros(len(re_arr))
    al_arr=np.zeros(len(re_arr))
    err_us=0.
    err_al=0.
    for i in range(len(re_arr)):
        us_arr[i],al_arr[i]=sc.ustar_alpha(re_arr[i])

    for i in range(len(re_sim)):
        i_loc=i
        nc = Dataset(files[re_sim[i_loc]],'r')
        al_mean=np.mean(nc['FrictionAngle'][:])
        us_mean=np.mean(nc['FrictionVelocity'][:]) 
        u_top=nc['rU'][:,-1]
        w_top=nc['rW'][:,-1]
        w_1=nc['rW'][:,1]
        u_1=nc['rU'][:,1] 
        dir_top=np.arctan(w_top/u_top)/D2R
        dir_sfc=np.arctan(w_1/u_1)/D2R
        veer=np.mean(dir_top-al_mean)
        times=nc['t'][:]
        t_skip=10
        print(re_sim[i_loc], veer,us_mean,np.mean(nc['FrictionAngle'][:]),np.mean(dir_sfc),np.mean(dir_top) )

        
    for i in range(len(re_sim)):
        u,a=sc.ustar_alpha(re_sim[i])
        err_us += (u-us_sim[i])**2
        err_al += (a/D2R-al_sim[i])**2

    print('TOTAL ERROR (us,alpha): {},{}'.format(err_us,err_al))
    fig=plt.figure(figsize=(10,5))
    ax=fig.add_axes([0.1,0.1,0.85,0.85])
    ax.plot(re_arr,1./us_arr,label=r'$G/u_\star$')
    ax.scatter(re_sim,1./us_sim) 
    ax.plot(re_arr,al_arr/D2R,label=r'$\alpha$')
    us_est=3.9*np.log(re_arr)-8
    ax.plot(re_arr,us_est,ls=':',lw=1,c='black',label=r'$u_\star=4\log(Re)-8$') 
    ax.scatter(re_sim,al_sim)
    ax.set_xscale('log')
    ax.set_xlabel(r'$Re_D$')
    ax.set_ylabel(r'$(G/u_{\star}), \alpha$') 
    lg=plt.legend(loc='lower left')
    lg.get_frame().set_linewidth(0.0)
    plt.xlim(3e2,1e4)
    plt.ylim(10,30)  
    plt.savefig('ustar_alpha.pdf',format='pdf')

# Mean wind speed profile according to Emeis et al. (2007) MetZet vol 16, p. 393-406
def emeis07_profile(re,dp,Z,al0):
    # estimate zPr from 3.10

    # us = kappa G ( - sin(al) + cos(al) )  /  ( ln(zPr+) + C ) 
    # => ln (zPr+) = kappa Z ( -sin(al) + cos(al)) - C
    # => zPr+ = exp ( kappa Z ( -sin(al) + cos(al)) -C ) 
    C_OFF = 5.4605
    KAPPA=0.415
    arg = KAPPA*Z* ( np.cos(al0)-np.sin(al0) ) - C_OFF 
    zPr_plus = np.exp ( KAPPA * Z * ( np.cos(al0) - np.sin(al0) ) - C_OFF )
    print('SIN/COS TEST:',np.sin(np.pi),np.cos(np.pi))
    print('{0:8.4g} {1:9.3g} {2:8.3g} {3:8.3g} {4:8.3g} {5:5.3G}'.format(re,dp,arg,al0,zPr_plus,zPr_plus/dp) )
    
    # estimate gamma from 3.11 
    
    
if plot_total_rot:
    print('Plotting rotation and velocity magnitude')
    t_use=-1

    fig=plt.figure(figsize=(10,5))
    ax1=fig.add_axes([0.07,0.11,0.42,0.85])
    ax2=fig.add_axes([0.57,0.11,0.42,0.85]) 
    
    for re in [500,750,1000,1300,1600,5000,10000,30000]:
        nu=2./(re*re)

        if re <= 1600: 
            f=Dataset(files[re],'r',format='NETCDF4')
            us=f['FrictionVelocity'][t_use]
            al=f['FrictionAngle'][t_use]
            u_dat=f['rU'][t_use,:]
            w_dat=f['rW'][t_use,:]
            y_dat=f['y'][:]

            al_geo = np.arctan(w_dat[-1]/u_dat[-1])
            al_sfc1 = f['FrictionAngle'][t_use]*D2R
            al_sfc2 = np.arctan(w_dat[1]/u_dat[1]) 

            [u_geo,w_geo] = mp.rotate(u_dat,w_dat,al_geo)
            [u_sfc,w_sfc] = mp.rotate(u_dat,w_dat,al_sfc1) 
            yp = y_dat*us/nu
            ym = y_dat/us 

            vm = np.sqrt(u_geo**2+w_geo**2)
            vp = vm/us
            d = -(np.arctan(u_sfc/w_sfc) - np.pi/2) / np.pi * 180
            d[0]=0

            ax1.plot(yp[1:],vp[1:],     ls='-', lw=1,c=colors[re],label=r'$Re={}$'.format(re))
            ax1.plot(yp[1:],d[1:],      ls='--',lw=1,c=colors[re])
            ax2.plot(ym[1:],(vm[1:]-vm[-1])/us,        ls='-', lw=1,c=colors[re])
            ax2.plot(ym[1:],d[1:]-d[-1],               ls='--',lw=1,c=colors[re])
        else:
            us,al=sc.ustar_alpha(re)
            deltap=us*us/nu 
            ny,yp,ym=build_grid(deltap,2.5) 

        up_mod,wp_mod=sc.profile_plus(yp,re)
        um_mod,wm_mod=sc.profile_minus(ym,re)
        vp_mod = np.sqrt(up_mod**2+wp_mod**2)
        vm_mod = np.sqrt(um_mod**2+wm_mod**2) 
        d_mod = -( np.arctan(up_mod/wp_mod) - np.pi/2) / np.pi*180 

        for i in range(len(d_mod)):
            if wp_mod[i] < 1e-6:
                d_mod[i]=0. 

        ax1.plot(yp[1:], vp_mod[1:],ls='-', lw=3,c=colors[re],alpha=0.5 ) 
        ax1.plot(yp[1:],d_mod[1:],  ls='--',lw=3,c=colors[re],alpha=0.5) 
        ax2.plot(ym[1:],(vm_mod[1:]-vm_mod[-1])/us,ls='-', lw=3,c=colors[re],alpha=0.5) 
        ax2.plot(ym[1:],d_mod[1:]-d_mod[-1],       ls='--',lw=3,c=colors[re],alpha=0.5)

        deltap=us*us/nu
        i0=mp.geti(yp,np.sqrt(deltap))
        i1=mp.geti(yp,1.5*np.sqrt(deltap))
        i2=mp.geti(yp,0.2*deltap)
        
        c_off=4.9e2*us/np.sqrt(deltap)
        c1=2.7e3*us/np.sqrt(deltap)/(np.log(yp[i0])**2.0)
        dir_fit=c1*np.log(yp[1:])**2.0+c_off
        ax1.plot(yp[1:],dir_fit,c=colors[re])

        slp=(dir_fit[i1+1]-dir_fit[i1])/(yp[i1+1]-yp[i1])
        off=dir_fit[i1]-slp*yp[i1]
        y_use=yp[i1:] 
        ax1.plot(y_use,slp*y_use+off,ls=':',lw=0.5,c=colors[re]) 
        
        i_03=mp.geti(yp,0.30*deltap)
        z_03=yp[i_03] 
        s_03=(wp_mod[i_03+1]-wp_mod[i_03-1])/(yp[i_03+1]-yp[i_03-1]) 
        a_03=s_03*z_03
        o_03=wp_mod[i_03]-a_03*np.log(z_03)
        ax1.scatter([yp[i0+1],yp[i_03]],[dir_fit[i0],d_mod[i_03]],c=colors[re])
        ylen=30
        y_use=yp[i_03-ylen:i_03]
        dir_plt=- ( np.arctan( up_mod/ (o_03+a_03*np.log(yp))) - np.pi/2)/np.pi*180.
        dir_plt = [d if d<180 else d-360 for d in dir_plt]
        print('LEN:',len(dir_plt),len(y_use))
        ax1.plot(y_use,dir_plt[i_03-ylen:i_03],c=colors[re],ls=':',lw=2) 



        

    ax1.plot([-1,-1],[1,1],ls='-', c='black',label=r'$U_{mag}$')
    ax1.plot([-1,-1],[1,1],ls='--',c='black',label=r'DIR') 

    ax2.plot([1e-4,2],[0,0],ls=':',c='black',lw=1) 
    
    ax1.set_xscale('log')
    ax1.set_xlim(1,2e3)
    ax1.set_ylim(0,25)
    ax1.set_xlabel(r'$z^+$')
    ax1.set_ylabel(r'$U_{mag}^+,  \alpha_{sfc}$')

    ax2.set_xscale('log')
    ax2.set_xlim(5e-3,2)
    ax2.set_xlabel(r'$z^-$')
    ax2.set_ylabel(r'$(U_{mag}-G)^+, \alpha_{G}$') 
    
    lg=ax1.legend(loc='best')
    lg.get_frame().set_linewidth(0.0)
    plt.savefig('mag_dir.pdf',format='pdf') 
    plt.close('all') 


    for re in [1000,2000,5000,10000,20000,50000,100000,200000,500000,1e6]:
        us,al=sc.ustar_alpha(re)
        deltap = us*us*re*re/2
        emeis07_profile(re,deltap,1/us,al)  
        


    # fig=plt.figure(figsize=(6,5))
    # ax=fig.add_axes([0.1,0.1,0.8,0.8]) 
    # for re in [1000,2000,10000,20000,50000,100000,150000,200000]:
    #     ym=np.arange(1e5)/3e4
    #     nu=2./(re*re) 
    #     us,al=sc.ustar_alpha(re)
    #     deltap = us*us/nu
    #     yp=ym*deltap
    #     up,wp = sc.profile_plus(yp,re)
    #     #ax.plot(ym,up,ls='-',  lw=1
    #     i0=mp.geti(yp,1) 
    #     print(re,deltap,yp[i0]) 
    #     ax.plot(ym[i0:],wp[i0:],ls='--', lw=1,label=r'$Re={}$'.format(re)) 

    # ax.set_xscale('log')
    # ax.plot([0,10],[0,0],ls=':',lw=1,c='black')
    # lg=plt.legend(loc='best')
    # lg.get_frame().set_linewidth(0.0) 
    # plt.show() 
    # quit() 

if plot_convergence_1000:
    re=1000
    nu=2./(re*re) 
    f=Dataset(files[re],'r',format='NETCDF4')

    t_all=f['t'][:]
    i0=mp.geti(t_all,f['t'][-1]-2.*np.pi) -1

    print(i0,f['it'][i0],f['t'][-1]-f['t'][i0]) 

    us=f['FrictionVelocity'][i0:]
    al=f['FrictionAngle'][i0:]
    u_dat=f['rU'][i0:,:]
    w_dat=f['rW'][i0:,:]
    y_dat=f['y'][:]
    t=f['t'][i0:] 
    us_avg=np.mean(us)
    al_avg=np.mean(al) 
    
    al_geo = np.arctan(w_dat[-1,-1]/u_dat[-1,-1])
    al_sfc1 = al_avg*D2R
    al_sfc2 = np.arctan(w_dat[-1,1]/u_dat[-1,1]) 

    [u_geo,w_geo] = mp.rotate(u_dat,w_dat,al_geo)
    [u_sfc,w_sfc] = mp.rotate(u_dat,w_dat,al_sfc1) 
    yp = y_dat*us_avg/nu
    ym = y_dat/us_avg 

    al =np.arctan(w_dat/u_dat) / D2R
    al_0 = np.arctan(w_dat[0,:]/u_dat[0,:]) 
    al_1 = np.arctan(w_dat[-1,:]/u_dat[-1,:]) 

    print(t[-1]-t[0])
    fig=plt.figure(figsize=(10,5))
    ax=fig.add_axes([0.1,0.1,0.8,0.8])


    fangle=f['FrictionAngle'][i0:]
    delta_al=al[0,1]-fangle[0]
    al_avg=np.mean(fangle) 
    #ax.plot(t,al[:,1],label=r'$\alpha(u,w,)$')
    #ax.plot(t,fangle,label=r'$\alpha(DNS)$')
    #ax.plot(t,t*0+al_avg,ls=':',lw=0.5,c='black')
    print(al_avg)
    ax.plot(yp,np.mean(u_sfc,0)/us_avg,label=r'$u^\alpha$',ls='-',c='blue')
    ax.plot(yp,np.mean(w_sfc,0)/us_avg,label=r'$w^\alpha$',ls='-',c='red')
    ax.plot(yp,np.mean(u_dat,0)/us_avg,label=r'$u^{comp}$',ls=':',c='blue')
    ax.plot(yp,np.mean(w_dat,0)/us_avg,label=r'$w^{comp}$',ls=':',c='red')
    ax.plot(yp,np.mean(u_geo,0)/us_avg,label=r'$u^{geo}$',ls='--',c='blue')
    ax.plot(yp,np.mean(-w_geo,0)/us_avg,label=r'$w^{geo}$',ls='--',c='red')
    #ax.set_xlim(1,1e2)
    #ax.set_ylim(0,5)
    ax.set_xscale('log')
    lg=plt.legend(loc='best')
    lg.get_frame().set_linewidth(0.0) 
    plt.show()
    plt.close('all')



    fig=plt.figure()
    ax=fig.add_axes([0.1,0.1,0.8,0.8]) 
    #ax.plot(ym,(u_sfc[-1,:]-u_sfc[0,:])/us_avg,c='blue', ls='-',label=r'$\Delta U^{\alpha+}$')
    ###ax.plot(ym,(u_geo[-1,:]-u_geo[0,:])/us_avg,c='blue', ls=':',label=r'$\Delta U^{G+}$')
    #ax.plot(ym,(w_sfc[-1,:]-w_sfc[0,:])/us_avg,c='red',  ls='-',label=r'$\Delta W^{\alpha+}$')
    #ax.plot(ym,(w_geo[-1,:]-w_geo[0,:])/us_avg,c='red',  ls=':',label=r'$\Delta W^{G+}$')
    #ax.plot(ym,(al_1-al_0)/D2R,                c='black',ls='-',label=r'$\Delta \alpha$')
    mag=np.sqrt(u_sfc*u_sfc+w_sfc*w_sfc)

    usmin=np.amin(u_sfc[:,:],0); ugmin=np.amin(u_geo[:,:],0) 
    usmax=np.amax(u_sfc[:,:],0); ugmax=np.amax(u_geo[:,:],0) 
    wsmin=np.amin(w_sfc[:,:],0); wgmin=np.amin(w_geo[:,:],0)
    wsmax=np.amax(w_sfc[:,:],0); wgmax=np.amax(w_geo[:,:],0)
    usavg=np.mean(u_sfc[:,:],0); ugavg=np.mean(u_geo[:,:],0)
    wsavg=np.mean(w_sfc[:,:],0); wgavg=np.mean(w_geo[:,:],0)
    mgmin=np.amin(mag[:,:],0);
    mgmax=np.amax(mag[:,:],0);
    mgavg=np.mean(mag[:,:],0);

    #
    ax.fill_between(ym,(usmin-usmin[-1])/us_avg,(usmax-usmax[-1])/us_avg,alpha=0.5,color='blue')
    ax.fill_between(ym,(wsmin-wsmin[-1])/us_avg,(wsmax-wsmax[-1])/us_avg,alpha=0.5,color='red')
    ax.fill_between(ym,(mgmin-mgmin[-1])/us_avg,(mgmax-mgmax[-1])/us_avg,alpha=0.5,color='black') 
    #ax.plot(ym,(u_geo[0,:]-u_geo[0,-1])/us_avg,c='blue',ls='--', alpha=0.5)
    #ax.plot(ym,(w_geo[0,:]-w_geo[0,-1])/us_avg,c='red',ls='--',  alpha=0.5)

    ax.plot(ym,(usavg-usavg[-1])/us_avg,c='blue',ls='-',  lw=1,label=r'$U^\alpha$')
    ax.plot(ym,(wsavg-wsavg[-1])/us_avg,c='red', ls='-',  lw=1,label=r'$W^\alpha$')
    ax.plot(ym,(mgavg-mgavg[-1])/us_avg,c='black',   label=r'$T$',lw=1)
    #ax.plot(ym,(u_geo[0,:]-u_geo[0,-1])/us_avg,c='blue',ls='--', alpha=0.5)
    #ax.plot(ym,(w_geo[0,:]-w_geo[0,-1])/us_avg,c='red',ls='--',  alpha=0.5)
    ax.set_ylim(-1,1)
    ax.set_xlim(0,2)

    plt.show()
    plt.close('all') 

    plt.imshow(u_sfc[:,:] - u_sfc[0,:],vmin=-0.02,vmax=0.02)
    plt.colorbar()
    plt.show() 
