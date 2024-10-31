#!/usr/bin/python3
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
plot_ustar_alpha=False
plot_ekman_ideal=False 
plot_summary=False
plot_evisc=False
plot_profiles=False
plot_visc_outer=False
print_table=False
plot_outer_log=False
plot_convergence=False
plot_re1600=False
plot_les_comp=False
plot_applications=False
test_inner_streamwise=False
plot_profile_comparison=False
plot_vandriest=False
plot_total_rot=False
plot_shear_vs_rotation=True

colors = {400 : 'gray',
          500 : 'pink',
          750 : 'cyan',
          1000: 'blue',
          1300: 'red',
          1301: 'red',
          1600: 'black',
          5000: 'black',
          10000:'gray',
          30000:'black',
          150000:'black',
          300000:'black',
          1000000:'black'} 

base='/home/cedria87/WORK/RESEARCH/SIM_NEUTRAL/netcdf/' 

files = { 400: base+'avg_flw_ri0.0_re400.0_1024x384x1024.nc',
          500: base+'avg_flw_ri0.0_re500.0_2048x192x2048.nc',
          750: base+'avg_flw_ri0.0_re750.0_3072x384x3072.nc',
          1000:base+'avg_flw_ri0.0_re1000.0_co0.0_1536x512x3072.nc',
          1300:base+'avg_flw_ri0.0_re1300.0_co0_2560x640x5120aligned.nc',
          1301:base+'avg_flw_ri0.0_re1300.0_co0_2560x640x5120.nc',
          1600:base+'avg_flw_ri0.0_re1600.0_co0_3840x960x7680.nc' } 


def etling_us(z0,zp,al):
    # universal parameter lz = lambda_z * 1/sqrt(2)
    lz=np.sqrt(2)*sc.KAPPA / np.log(zp/z0)
    us = lz*np.sin(np.pi/4.-al)
    
    # checking solution
    #print('SOLUTION: {1:8.6f}, {2:8.6f} {3:8.6f}'.format(0,us,cos_al,al/np.pi*180) )
    #print(us/kappa*np.log(zp/z0)*np.cos(al_0))
    #print(G*(1-np.sqrt(2)*np.sin(al_0)*np.cos(pi4-al_0)))
    #print(us/kappa*np.log(zp/z0)*np.sin(al_0))
    #print(G*np.sqrt(2)*np.sin(al_0)*np.sin(pi4-al_0))
    #print(us/G,al/np.pi*180) 
    return us 


def etling_profile(yp,ret,et_al,gamma_p=0.1):
    kappa=sc.KAPPA
    off_A=sc.C 

    et_al = et_al *D2R 
    
    ny,yp,ym=build_grid(ret,thick=3)
    i_01 =mp.geti(yp,gamma_p*ret)
    yprandtl = gamma_p*ret
    surface_factor=np.exp(kappa*off_A) 
    z0 = yprandtl / (surface_factor*gamma_p*ret)
    et_us = etling_us(z0,yprandtl,et_al)
    iprandtl = mp.geti(yp,yprandtl)
    print(iprandtl)

    etling_pr_up=np.zeros(ny) 
    etling_pr_vp=np.zeros(ny)
    etling_ek_up=np.zeros(ny)
    etling_ek_vp=np.zeros(ny)

    print('ETLING PROFILE FOR RE_tau={}: us={}, al={}'.format(ret,et_us,et_al*D2R))

    d_ek_m = np.sqrt(2*kappa*gamma_p)
    d_ek_p = d_ek_m * ret

    # profiles
    z_tilde = (yp - gamma_p*ret ) / d_ek_p 
    z_arg=z_tilde[1:]+np.pi/4-et_al
    damp = np.exp(-z_tilde[1:])
    etling_pr_up[1:]=1./sc.KAPPA*np.log(yp[1:]*surface_factor)*np.cos(et_al) 
    etling_pr_vp[1:]=1./sc.KAPPA*np.log(yp[1:]*surface_factor)*np.sin(et_al)
    etling_ek_up[1:]=(1-damp*np.sqrt(2)*np.sin(et_al)*np.cos(z_arg))/et_us
    etling_ek_vp[1:]=damp*np.sqrt(2)*np.sin(et_al)*np.sin(z_arg)/et_us

    # fig=plt.figure(figsize=(5,3))
    # ax=fig.add_axes([0.1,0.1,0.8,0.8])
    # ax.plot(yp,etling_pr_up[:],label='prandtl',lw=3)
    # ax.plot(yp[1:],etling_ek_up[1:],label='Ekman',  lw=3)

    etling_ek_up[:iprandtl] = etling_pr_up[:iprandtl] 
    etling_ek_vp[:iprandtl] = etling_pr_vp[:iprandtl]
    # ax.plot(yp,etling_ek_up,label='combined u',ls='--',lw=2)
    # ax.plot(yp,etling_ek_vp,label='combined v',ls='--',lw=2)
    # lg=plt.legend(loc='best')
    # ax.set_xscale('log')
    # plt.show() 
    # plt.close('all')
    return etling_ek_up,etling_ek_vp 

sc=EkmanUniversalClass() # yp_ref,up_ref,wp_ref,deltap_ref,us_ref,al_ref*D2R-geo_rotate,plot=False)

if use_data: 
    re_ref=1600
    nu_ref=2./(re_ref**2)

    ncf='/home/cedria87/WORK/RESEARCH/SIM_NEUTRAL/netcdf/avg_flw_ri0.0_re1600.0_co0_3840x960x7680.nc' 
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

if plot_re1600:
    re=1600;
    nu=2./(re*re) 


    f=Dataset(files[re],'r')
    u_datC=f['rU'][:,:]
    w_datC=f['rW'][:,:]
    y_dat=f['y'][:]
    t_dat=f['t'][:] 
    us=np.average(f['FrictionVelocity'][:])
    al=np.average(f['FrictionAngle'][:])
    
    print(t_dat[-1]-t_dat[0]) 

    file_name = '/home/cedria87/WORK/RESEARCH/SIM_NEUTRAL/netcdf/avg_flw_ri0.0_re1600.0_co0_3840x960x7680.nc'
    f=Dataset(file_name,'r')
    y=f['y'][:]
    ny=len(y)
    fig=plt.figure(figsize=(10,5))
    ax1=fig.add_axes([0.06,0.1,0.4,0.85])
    ax2=fig.add_axes([0.50,0.1,0.47,0.85])
    NT=10
    U=np.zeros([NT,ny])
    W=np.zeros([NT,ny])
    us=np.zeros([NT])
    al=np.zeros([NT]) 


    U = u_datC[28:,:]
    W = w_datC[28:,:]

    al = f['FrictionAngle'][:]
    us = f['FrictionVelocity'][:]
    Uavg=np.average(U,0)
    Wavg=np.average(W,0)
    Udsh=(U-Uavg)/np.average(us) 
    Wdsh=(W-Wavg)/np.average(us)

    
    im1=ax1.contourf(f['t'][28:]-f['t'][28],f['y'][:]*np.average(us)*1600*1600/2,np.transpose(Udsh),levels=64,vmin=-0.04,vmax=0.04)
    im2=ax2.contourf(f['t'][28:]-f['t'][28],f['y'][:]*np.average(us)*1600*1600/2,np.transpose(Wdsh),levels=64,vmin=-0.04,vmax=0.04)

    ax1.set_yscale('log')
    ax1.set_ylim(1,3e3)
    ax1.set_xlabel(r'time $(tf) [1]$')
    ax1.set_ylabel(r'height $(y^+) [1]$') 
    ax1.set_title(r'$U-\overline{U}$')
    
    ax2.set_yscale('log')
    ax2.set_ylim(1,3e3)
    ax2.set_xlabel(r'time $(tf) [1]$')
    ax2.set_title(r'$W-\overline{W}$' )
    # ax2.set_ylabel(r'height $(y^+) [1]$') 

    fig.colorbar(im1,fraction=0.1) 
    plt.savefig('re1600_uw_variation.jpg',format='jpg',dpi=900) 
    plt.close('all')
    

    fig=plt.figure(figsize=(10,5))
    ax=fig.add_axes([0.1,0.1,0.8,0.8]) 
    
    l = ['modified', 'orig','50','100','150','200','250','300','350','400','450','500']
    
    for it in range(NT): 
        U_loc=f['rU'][it,:]
        W_loc=f['rW'][it,:]
        us[it]=f['FrictionVelocity'][it] 
        al[it]=f['FrictionAngle'][it]
        al_geo=np.arctan(W_loc[-1]/U_loc[-1]) 
        [U[it],W[it]]  = mp.rotate(U_loc,W_loc,al[it]/180*np.pi)

        yp=y*us[it]/nu
        
        print('I',it,us[it],al[it])

        if it==0: 
            ax.plot(yp,U[it]/us[it],label='{}'.format(l[it]),c='red',ls='-',lw=1)
            ax.plot(yp,W[it]/us[it],label='{}'.format(l[it]),c='blue',ls='--',lw=1)
        else:
            print(U[it,10]) 
            ax.plot(yp,U[it]/us[it],c='red', ls='-',lw=1)
            ax.plot(yp,W[it]/us[it],c='blue',ls='--',lw=1)

    IT=1
    #print(al) 
    al_mod = -al[IT]/180*np.pi+al_geo
    print('ANGLE:', al_mod, al_mod*180/np.pi,al_geo)
    
    u_mod,w_mod= sc.profile_plus(yp,re,us=us[IT],al=al_mod)
    u_modC,w_modC=mp.rotate(u_mod,w_mod,-al[IT])
    ax.plot(y*us[IT]/nu,10*(U[1]-U[0])/us[1],c='red', ls=':',lw=0.5)
    ax.plot(y*us[IT]/nu,10*(W[1]-W[0])/us[1],c='blue',ls=':',lw=0.5)
    ax.plot(y*us[IT]/nu,u_mod,c='black',ls=':',label='model',lw=0.5)
    ax.plot(y*us[IT]/nu,w_mod,c='black',ls=':', lw=0.5) 
    ax.plot(y[1:]*us[IT]/nu,y[1:]*0,c='black',ls='-',lw=0.5) 
    ax.set_xscale('log') 
    ax.set_xlim(1,1e4)
    lg=ax.legend(loc='best')

    fig1=plt.figure(figsize=(5,4))
    ax1=fig1.add_axes([0.1,0.1,0.8,0.8])

    fig2=plt.figure(figsize=(5,4))
    ax2=fig2.add_axes([0.1,0.1,0.8,0.8])
    
    al_geo = np.arctan(w_datC[0,-1]/u_datC[0,-1]) /np.pi*180
    #print('AVERAGE:', us,    al_geo-al,     al,     al_geo)
    us_last=f['FrictionVelocity'][-1]
    al_last=f['FrictionAngle'][-1] 
    print('FINAL:  ',us_last,al_geo-al_last,al_last,al_geo) 

    # USING FINAL VALUES FOR U* and ALPHA 
    us=us_last
    al=al_last 
    
    ym=y_dat/us
    yp=y_dat*us/nu
    [u_datR,w_datR] = mp.rotate(u_datC,w_datC,al/180*np.pi) 
    

    print('DELTA PLUS (FRICTION-RE):', us*us/nu)
    print('HIGHEST GRID POINT: (N=', len(ym), '):',ym[-1]) 
    print(mp.geti(ym,0.66))
    al_mod=(al_geo-al_last)/180*np.pi 
    up_modR,wp_modR = sc.profile_plus(yp,re,us=us,al=al_mod)

    print('PROFILE VALUES AT SFC:', up_modR[0],wp_modR[0])
    print('DATA VALUES AT SFC:',    u_datC[-1,0],w_datC[-1,0])
    up_modC,wp_modC=mp.rotate(up_modR,wp_modR,-al/180*np.pi) 

    u_deltaR = up_modR-u_datR[-1,:]/us
    w_deltaR = wp_modR-w_datR[-1,:]/us

    u_deltaC = up_modC-u_datC[-1,:]/us   # => u_dat+ u_deltaC = up_modC
    w_deltaC = wp_modC-w_datC[-1,:]/us
    print('CORRECTION AT SFC:', u_deltaC[0],w_deltaC[0]) 

    [utest,wtest] = mp.rotate(u_datC[-1]+u_deltaC,w_datC[-1]+w_deltaC,al/180*np.pi)
    print('CHECK FOR DIFFERENCES IN PROFILES :',np.amin(utest-up_modR),np.amax(utest-up_modR),np.amin(wtest-wp_modR),np.amax(wtest-wp_modR))
    
    ax1.plot(yp,up_modR,c='red',ls='-',label='Model')
    ax1.plot(yp,u_datR[-1]/us,c='red',ls=':',label='Data')
    ax1.plot(yp,u_deltaR*10/us,ls='--',c='red',label='10 x (Model-Data)') 

    ax1.plot(yp,wp_modR,c='blue',ls='-')
    ax1.plot(yp,w_datR[-1]/us,c='blue',ls=':')
    ax1.plot(yp,w_deltaR*10/us,c='blue',ls='--')
    ax1.set_xscale('log')
    lg1=ax1.legend(loc='best')

    ax2.plot(yp,up_modC,c='red',ls='-',label='Model')
    ax2.plot(yp,u_datC[-1]/us,c='red',ls=':',label='Data')
    ax2.plot(yp,u_deltaC*10,ls='--',c='red',label='10 x (Model-Data)') 

    ax2.plot(yp,wp_modC,c='blue',ls='-')
    ax2.plot(yp,w_datC[-1]/us,c='blue',ls=':')
    ax2.plot(yp,w_deltaC*10 ,c='blue',ls='--')
    ax2.plot(yp,yp*0,c='black',ls='-',lw=0.5) 
    ax2.set_xscale('log')
    ax2.set_ylim(-1,5) 
    lg2=ax2.legend(loc='best')

    dat = np.array([y_dat,u_deltaC*us,w_deltaC*us])
    #for i in range(len(dat[0])):
    #    print(i,dat[0,i],dat[1,i],dat[2,i]) 
    dat.T.tofile('velocity_correction.dat')
    

    ax1.set_xlim(1,1e4)
    ax2.set_xlim(1,1e4)

if plot_ekman_ideal:
    fig=plt.figure(figsize=(10,3.75))
    ax1=fig.add_axes([0.06,0.11,0.41,0.87])
    ax2=fig.add_axes([0.55,0.11,0.44,0.87])

    fig2=plt.figure(figsize=(5,3.25))
    ax3=fig2.add_axes([0.18,0.13,0.8,0.84]) 
    
    z=np.arange(0,2*np.pi,0.02)
    
    sa_arr=[[+0.00, 0.4,'black', '-', 1.0],
            [+0.12, 0.4,'red',   '-', 4.0],
            [-0.12, 0.4,'blue',  '-', 1.0],
            [+0.00, 1.0,'black', ':', 4.0], 
            [+0.12, 1.0,'red',  '--', 1.0],
            [-0.12, 1.0,'blue', '--', 1.0]
            ]

    for sa in sa_arr:
        sek=sa[0]
        aek=sa[1]
        zek = z-sek
        Uek=1-aek*np.exp(-zek)*np.cos(zek)
        Vek=aek*np.exp(-zek)*np.sin(zek)
        c=sa[2]
        s=sa[3] 
        w=sa[4]

        i03=mp.geti(z,0.3*np.pi*2)
        
        ax1.plot(z[i03:],Uek[i03:],  c=c,ls=s,lw=w, label='a={1:4.1f},s={0:5.2f}'.format(sek,aek))
        ax1.plot(z[:i03],Uek[:i03],  c=c,ls=s,lw=w, alpha=0.5)
       
        ax2.plot(Uek[i03:],Vek[i03:], c=c,ls=s,lw=w,label='a={1:4.1f} s={0:5.2f}'.format(sek,aek))
        ax2.plot(Uek[:i03],Vek[:i03], c=c,ls=s,lw=w,alpha=0.5)
        ax2.scatter(Uek[i03],Vek[i03],c=c)

        ax3.plot(Uek[i03:],Vek[i03:], c=c,ls=s,lw=w,label=r'A={1:4.1f} $z_r$={0:5.2f}'.format(sek,aek))
        ax3.plot(Uek[:i03],Vek[:i03], c=c,ls=s,lw=w,alpha=0.5)
        ax3.scatter(Uek[i03],Vek[i03],c=c)
        
    lg=ax1.legend(loc='best')
    lg2=ax3.legend(loc='best') 

    ax1.plot([0.3*np.pi*2,0.3*np.pi*2],[-0.5,1.1],c='black',ls='--',lw=0.5)
    ax1.plot([0,6.3],[1,1],c='black',ls='--',lw=0.5) 
    
    ax2.plot([0,1.1],[0,0],lw=0.5,ls='--',c='black')
    ax2.plot([1,1],[-0.1,0.35],lw=0.5,ls='--',c='black')

    ax3.plot([0,1.1],[0,0],lw=0.5,ls='--',c='black')
    ax3.plot([1,1],[-0.1,0.35],lw=0.5,ls='--',c='black')
    
    ax1.set_ylim(-0.15,1.1)
    ax1.set_xlim(0,5)
    ax1.set_ylabel(r'$U_\mathrm{ek}/G$')
    ax1.set_xlabel(r'$z/\delta_{ek}$')
    
    ax2.set_xlim(0,1.1)
    ax2.set_ylim(-0.08,0.35)
    ax2.set_xlabel(r'$U_\mathrm{ek}/G$')
    ax2.set_ylabel(r'$V_\mathrm{ek}/G$') 

    ax3.set_xlim(0,1.1)
    ax3.set_ylim(-0.08,0.35)
    ax3.set_xlabel(r'$U_\mathrm{ek}/G$')
    ax3.set_ylabel(r'$V_\mathrm{ek}/G$') 

    fig.savefig('ekman_ideal.pdf',format='pdf')
    fig2.savefig('ekman_ideal_hod.pdf',format='pdf') 
    plt.close('all')
    
if plot_evisc:
    fig=plt.figure(figsize=(12,4.5))
    ax2=fig.add_axes([0.06,0.11,0.43,0.87])
    ax1=fig.add_axes([0.55,0.11,0.43,0.87])
    fig2=plt.figure(figsize=(12,4.5)) 
    ax3=fig2.add_axes([0.56,0.11,0.43,0.87])
    ax4=fig2.add_axes([0.06,0.11,0.43,0.87])

    for re in [500,750,1000,1300,1600]:
        nu=2./(re*re)
        f=Dataset(files[re],'r',format='NETCDF4')
        y=f['y'][:]
        us_arr=f['FrictionVelocity'][:]
        us=np.average(us_arr)
        yp=y*us/nu
        ym=y/us
        deltap=us*us/nu  # = delta * (us/nu) =  (us/f) * us /  nu [with f=1]
        ny=len(y)
        nt=len(us_arr)
        u=f['rU'][:,:]
        w=f['rW'][:,:]
        Rxy=f['Rxy'][:,:]
        Ryz=f['Ryz'][:,:]
        # Get Gradient of velocity components
        if  'U_y1XXX' in f.variables.keys() : # New outpout (contains derivatives
            dUdy=f['U_y1'][:,:]   # calculated using compct scheme)
            dWdy=f['W_y1'][:,:]
        else :                                  # Old file format (does not contain
            print("YP:",yp.shape)
            print("U: ",u.shape)
            dUdy = mp.derivative1(y,np.transpose(u),ny,nt).transpose() # derivatives...) 
            dWdy = mp.derivative1(y,np.transpose(w),ny,nt).transpose()


        print('RE=',re,np.average(dUdy[:,0]),np.average(dWdy[:,0]),us)
        Txy=nu*dUdy               # Viscous Stress 
        Tyz=nu*dWdy

        us_mean=np.average(us)

        print("Txy:",Txy.shape)
        print("Rxy:",Rxy.shape)
        
        stress_x = -np.average(-Txy + Rxy,0)
        stress_z = -np.average(-Tyz + Ryz,0)
        stress_t = np.sqrt(stress_x**2 + stress_z**2)
        tstress_t  = np.sqrt(np.average(Rxy,0)**2+np.average(Ryz,0)**2) 

        dTdy = np.average(np.sqrt(dUdy**2+dWdy**2),0)
        tstress_v = nu*dTdy
        tstress_s = tstress_t+tstress_v
        # evisc = stress_t / dTdy  / (deltap * us * us)
        evisc = tstress_t / dTdy # / (deltap*us*us) 
        evisc2= 0.*evisc[:]
        print(evisc.shape,evisc2.shape,dTdy.shape)
        evisc[0]=0
        for i in range(1,len(dTdy)-1):
            if dTdy[i] < 1e-6:
                evisc[i]=np.nan
            else :
                evisc2[i] = (evisc[i-1]+ 2*evisc[i]+evisc[i+1])/4.
        evisc=evisc2

        print(evisc[10],evisc2[10])
        
        norm = 1./(us*us)
        norm_v_outer=nu/(nu*us*us)#*us)
        norm_v_inner=1./nu

        ax3.plot(ym, evisc2*norm_v_outer,lw=2,ls='-',label='Re={}'.format(re),c=colors[re])
        ax4.plot(yp, evisc2*norm_v_inner,lw=2,ls='-',label='Re={}'.format(re),c=colors[re])
        if re == 1600:
            ax1.plot([1,1],[1,1],lw=2,ls='-',label=r'$Re={}$'.format(re),c=colors[re]) 
            ax1.plot([1,1],[1,1],lw=0,ls=' ',label=' ',c='white') 
            ax1.plot(ym, tstress_t*norm, ls='--', lw=1, label=r'$S_\mathrm{turb}^+$',c=colors[re])
            ax1.plot(ym, tstress_v*norm, ls=':',  lw=1, label=r'$S_\mathrm{visc}^+$',c=colors[re])
            ax1.plot(ym, tstress_s*norm, ls='-',  lw=2, label=r'$S^+$',c=colors[re])

            ax2.plot([1,1],[1,1],lw=2,ls='-',label=r'$Re={}$'.format(re),c=colors[re])
            ax2.plot([1,1],[1,1],lw=0,ls=' ',label=' ',c='white') 
            ax2.plot(yp, tstress_t*norm, ls='--', lw=1, label=r'$S_\mathrm{turb}^+$',c=colors[re])
            ax2.plot(yp, tstress_v*norm, ls=':',  lw=1, label=r'$S_\mathrm{visc}^+$',c=colors[re])
            ax2.plot(yp, tstress_s*norm, ls='-',  lw=2, label=r'$S^+$',c=colors[re])

            ax3.plot(ym, evisc2*norm_v_outer,lw=2,ls='-',label=''.format(re),c=colors[re])
            ax4.plot(yp, evisc2*norm_v_inner,lw=2,ls='-',label=''.format(re),c=colors[re])

        else : 
            ax1.plot(ym, tstress_t*norm, ls='--', lw=1, c=colors[re])
            ax1.plot(ym, tstress_v*norm, ls=':',  lw=1, c=colors[re])
            ax1.plot(ym, tstress_s*norm, ls='-',  lw=2, c=colors[re],label=r'$Re={}$'.format(re))

            ax2.plot(yp, tstress_t*norm, ls='--', lw=1, c=colors[re])
            ax2.plot(yp, tstress_v*norm, ls=':',  lw=1, c=colors[re])
            ax2.plot(yp, tstress_s*norm, ls='-',  lw=2, c=colors[re],label=r'$Re={}$'.format(re))

        i01 = mp.geti(ym,0.15)
        i50 = mp.geti(yp,50)
        ax1.scatter(ym[i50],tstress_s[i50]*norm,c=colors[re],marker='d')
        ax2.scatter(yp[i01],tstress_s[i01]*norm,c=colors[re],marker='o') 
        ax3.scatter(ym[i50],evisc2[i50]*norm_v_outer,c=colors[re],marker='d') 
        ax4.scatter(yp[i01],evisc2[i01]*norm_v_inner,c=colors[re],marker='o') 

            
    ax4.plot(yp, 23*np.log(yp)-72,ls=':',lw=1,label=r'$ 24 log z^+ -76$',c='black') 

    #lg=ax1.legend(loc=(0.11,0.5))
    #lg.get_frame().set_linewidth(0.0 )
    ax1.set_xscale('log')
    # ax.set_yscale('log')

    ax1.set_xlabel(r'$z^-$')
    # ax1.set_ylabel(r'$S^+, S_\mathrm{turb}^+, S_\mathrm{visc}^+$')
    ax1.set_xlim(1e-3,2)
    ax1.set_ylim(0,1)

    lg2=ax2.legend(loc='upper right')
    lg2.get_frame().set_linewidth(0.0)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$z^+$')
    ax2.set_ylabel(r'$S^+, S_\mathrm{turb}^+, S_\mathrm{visc}^+$')
    ax2.set_xlim(1,2e3)
    ax2.set_ylim(0,1) 

    lg4=ax4.legend(loc='upper left')
    lg4.get_frame().set_linewidth(0.0)
    ax4.set_xlabel(r'$z^+$')
    ax4.set_xscale('log')
    ax4.set_xlim(1,2e3)
    ax4.set_ylim(0,1e2)
    ax4.set_ylabel(r'$\dfrac{\nu_E}{\nu}$')


    ax3.set_xlabel(r'$z^-$')
    ax3.set_xscale('log')
    ax3.set_xlim(1e-3,2)
    ax3.set_ylim(0,0.1)
    ax3.set_ylabel(r'$\dfrac{\nu_E}{\nu} \dfrac{1}{\delta^+}$')
    
    fig.savefig('stresses.pdf',format='pdf')
    fig2.savefig('eddy_viscosity.pdf',format='pdf')

if plot_profiles:
    # calculate and plot a_log

    c_log = []
    c1=[]
    c2=[]
    re_arr=np.arange(2.5,10,0.05)
    re_arr=np.power(10,re_arr) 
    for re in re_arr:
        us,al=sc.ustar_alpha(re)
        dp=us*us*re*re/2 
        ym=np.array([0.3])
        [e_u,e_w]=sc.profile_ekman(ym,us)
        [ous,ows]=mp.rotate(e_u,e_w,al)
        w0=27
        d0=4.01
        w1=-ows[0]*us*dp
        deltav=w1-w0
        deltaz=0.3*dp-10
        c_log.append((deltav - 10*d0*np.log(0.03*dp)) / deltaz)
        c1.append(deltav/deltaz)
        c2.append(10*d0*np.log(0.03*dp)/deltaz)
        #blog=(d0-clog)*10
        #a_log.append(w0-blog*np.log(10) - clog*10)
        print(re,us,al,c_log[-1],c1[-1],c2[-1])
        
    c_log=np.array(c_log).flatten()
    c1=np.array(c1).flatten()
    c2=np.array(c2).flatten() 
    fig=plt.figure(figsize=(5,4))

    ax=fig.add_axes([0.13,0.13,0.85,0.85])
    i0=mp.geti(re_arr,4e4)
    i1=mp.geti(re_arr,4e6)
    ax.plot(re_arr[i0:i1],c_log[i0:i1],ls='-',lw=4,c='black',alpha=0.5)
    ax.plot(re_arr,c_log,ls='-',lw=2,label=r'$c_\mathrm{log}$',c='black')
    ax.plot(re_arr,c1,ls='--',lw=1,label=r'$\Delta v/ \Delta $') 
    ax.plot(re_arr,c2,ls='--',lw=1,label=r'$z_0 d_0 log(\delta^+/z_0) / \Delta z$')
    ax.plot(re_arr[[0,i1]],c_log[[i1,i1]],ls='--',lw=0.5,c='black')
    ax.plot(re_arr[[0,i0]],c_log[[i0,i0]],ls='--',lw=0.5,c='black')
    ax.plot(re_arr[[i0,i0]],c_log[[0,i0]],ls='--',lw=0.5,c='black')
    ax.plot(re_arr[[i1,i1]],c_log[[0,i1]],ls='--',lw=0.5,c='black')
    ax.set_xlim(re_arr[0],re_arr[-1])
    ax.set_ylim(0,max(c_log)+0.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$Re_D$')
    ax.set_ylabel(r'$a_\mathrm{log}$') 
    lg=ax.legend(loc='best')
    lg.get_frame().set_linewidth(0.0)

    fig.savefig('c_log.pdf',format='pdf') 
    
    fig=plt.figure(figsize=(5,4))
    ax=fig.add_axes([0.1,0.1,0.85,0.85])

    
    for re in [500,750,1000,1300,1600,2000,5000,10000,100000]:
        us,al=sc.ustar_alpha(re)
        dp=us*us*re*re/2
        yp=np.arange(dp) 
        ym=np.array([0.2,0.3,5])
        [e_u,e_w]=sc.profile_ekman(ym,us)
        [ous,ows]=mp.rotate(e_u,e_w,al)
        w0=27.
        yp[0]=0.1
        yp[1]=0.2
        yp[2]=0.4
        yp[3]=0.8
        yp[4]=1.6
        yp[5]=3.2
        yp[6]=6 
        d0=4.01
        w1=-ows[1]*us*dp
        deltav=w1-w0 
        deltaz=0.3*dp-10 
        clog=(deltav - 10*d0*np.log(0.03*dp)) / deltaz
        blog=(d0-clog)*10
        alog=w0-blog*np.log(10) - clog*10 
        print('.... ',re,alog,blog,clog)

        wlog=alog + blog*np.log(yp) + clog*yp
        i1=mp.geti(yp,10000)
        i0=mp.geti(yp,5) 
        ax.plot(yp[i0:],wlog[i0:],ls='-',lw=1)
    wsfc=18.85*(0.2353*yp/100 -1 + np.exp(-0.2353*yp/100))
    ax.plot(yp[:i1]/100,wsfc[:i1],ls='--',c='black')
                
    ax.set_xscale('log')
    ax.set_xlim(1e-2,1e5) 
    ax.set_yscale('log')
    plt.show()
    plt.close('all') 

    fig1=plt.figure(figsize=(9,3.25))
    ax1=fig1.add_axes([0.08,0.13,0.41,0.85])
    ax2=fig1.add_axes([0.56,0.13,0.43,0.85])

    fig2=plt.figure(figsize=(5,4))
    ax3=fig2.add_axes([0.1,0.1,0.85,0.85]) 

    fig3=plt.figure(figsize=(5,4))
    ax4=fig3.add_axes([0.1,0.1,0.85,0.85]) 

    
    ylog=np.arange(1e1,1e3)
    vlog=sc.profile_log(ylog) 
    ax1.plot(ylog,sc.profile_log(ylog),lw=3,ls='-',c='black',alpha=0.5,
             label=r'$\kappa^{-1} log z^+ + C$')
    # ax1.plot([50,50],[0,21],ls='--',lw=0.5,c='black') 
    for re in [500,750,1000,1300,1600]:
        c=colors[re]
        nu=2./re**2
        f=Dataset(files[re],'r')
        y=f['y'][:]
        u_dat=np.average(f['rU'][:,:],0)
        v_dat=np.average(f['rW'][:,:],0) 
        us_tim=f['FrictionVelocity'][:]
        al_tim=f['FrictionAngle'][:]
        us=np.average(us_tim)
        al=np.average(al_tim)

        [u,v]=mp.rotate(u_dat,v_dat,al/180*np.pi)
        yp=y*us/nu
        ym=y/us
        up,vp=u/us,v/us
        um,vm=u/us,v/us
        ax1.plot(yp[1:],up[1:],ls='-', c=c,lw=2,label=r'Re={}'.format(re))
        ax1.plot(yp[1:],vp[1:],ls='--',c=c,lw=2)

        ax2.plot(ym[1:],um[1:]-um[-1],ls='-',c=c,lw=2)
        ax2.plot(ym[1:],vm[1:]-vm[-1],ls='--',c=c,lw=2)
        ax2.plot([0,1.5],[0,0],ls='--',lw=0.5,c='black')
        #ax2.plot([0.3,0.3],[-7,1.1],ls='--',lw=0.5,c='black')
        #ax2.plot([0.15,0.15],[-7,1.1],ls='--',lw=0.5,c='black')

        i01 = mp.geti(ym,0.15)
        i50 = mp.geti(yp,50)
        ax1.scatter(yp[i01],up[i01],c=c,marker='o') 
        ax2.scatter(ym[i50],um[i50]-um[-1],c=c,marker='d') 

        uu=np.average(f['Rxx'][:,:],0)
        vv=np.average(f['Ryy'][:,:],0)
        ww=np.average(f['Rzz'][:,:],0)
        us2=us*us
        ax3.plot(yp,uu/us2,ls='-', label='RE={}'.format(re),c=c)
        ax3.plot(yp,vv/us2,ls='--',c=c)
        ax3.plot(yp,ww/us2,ls=':', c=c)

        ax4.plot(ym,uu/us,ls='-', label='RE={}'.format(re),c=c)
        ax4.plot(ym,vv/us,ls='--',c=c)
        ax4.plot(ym,ww/us,ls=':', c=c) 
        
        print(re,us,al)
    #ax1.plot([1,1],[1,1],ls=':', c='white',lw=0,label=' ')
    #ax1.plot([1,1],[1,1],ls='-', lw=2,c='gray',label=r'$U^\alpha$')
    #ax1.plot([1,1],[1,1],ls='--',lw=2,c='gray',label=r'$V^\alpha$')
        
    ax1.set_xscale('log')
    ax1.set_xlim(1,2e3) 
    ax1.set_xlabel(r'$z^+$')
    ax1.set_ylabel(r'$U^{\alpha+}, V^{\alpha+}$')
    ax1.set_ylim(0,21)

    lg1=ax1.legend(loc='best')
    lg1.get_frame().set_linewidth(0.0)

    ax2.set_xscale('log') 
    ax2.set_xlim(1e-3,2.e0)
    ax2.set_ylim(-7,1.2)
    ax2.set_xlabel(r'$z^-$')
    ax2.set_ylabel(r'$(U^\alpha-G_{1}^\alpha)/u_\star, (V^\alpha-G_{2}^{\alpha}/u_\star$') 

    lg2=ax2.legend(loc='best')
    lg2.get_frame().set_linewidth(0.0) 
    fig1.show()
    fig1.savefig('uv_innerouter.pdf',format='pdf')

    ax3.set_xlabel('yp')
    ax3.set_ylabel('Stress')
    ax3.set_xscale('log')
    ax3.set_xlim(1,2e3)
    l2=ax3.legend(loc='best')
    l2.get_frame().set_linewidth(0.0) 
    fig2.savefig('stress_re_inner.pdf',format='pdf') 

    ax4.set_xlabel('ym')
    ax4.set_ylabel('Stress' )
    ax4.set_xscale('log')
    ax4.set_xlim(1e-3,2)
    l3=ax4.legend(loc='best')
    l3.get_frame().set_linewidth(0.0) 
    
    fig3.savefig('stress_re_outer.pdf',format='pdf')

    plt.close('all')
    
    
if plot_visc_outer:
    fig=plt.figure(figsize=(5,4))
    ax=fig.add_axes([0.1,0.1,0.87,0.8])

    fig2=plt.figure(figsize=(6,4))
    ax2=fig2.add_axes([0.13,0.1,0.85,0.87]) 

    fig3=plt.figure(figsize=(5,3.5))
    ax3=fig3.add_axes([0.15,0.15,0.83,0.82]) 

    for re in [1600,1000,750]:
        c=colors[re]
        nu=2./re**2
        us,al=sc.ustar_alpha(re)
        dp=us**2/nu
        ny,yp,ym=build_grid(dp,5.5)

        up,wp=sc.profile_plus(yp,re)
        um,wm=sc.profile_minus(yp,re)
        f=Dataset(files[re],'r')
        ifinal=np.size(f['t'][:])-1
        t_final=f['t'][ifinal]
        i0=mp.geti(f['t'],t_final-2*np.pi)
        t_0=f['t'][i0] 
        print(re,'TIMES:',i0,ifinal,t_0,t_final)
        

        
            
        u_dat=np.average(f['rU'][i0:ifinal,:],0)
        w_dat=np.average(f['rW'][i0:ifinal,:],0) 
        y_dat=f['y'][:]
        us_dat=np.average(f['FrictionVelocity'][i0:ifinal])
        al_geo = np.arctan(w_dat[-1]/u_dat[-1])
        al_sfc1 = np.average(f['FrictionAngle'][i0:ifinal])*D2R
        al_sfc2 = np.arctan(w_dat[1]/u_dat[1]) 
            
        [u_sfc,w_sfc] = mp.rotate(u_dat,w_dat,al_sfc1) 
        yp_dat = y_dat*us_dat/nu
        ym_dat = y_dat/us_dat

        c_argz = 0.66*2.*np.pi
        argz=    c_argz*(ym+0.12)
        dampA =  8.4*us       - 0./re #np.exp(zdum) * ( dU_mtc*np.cos(zdum) + dW_mtc*np.sin(zdum) )
        z=1/us
        
        dampB= 0. 
        outer_u= (1-(dampA*np.cos(argz))*np.exp(-argz))*z  #-dampB*np.cos(argz)*np.exp(-argz)*z
        outer_w=   ( dampA*np.sin(argz))*np.exp(-argz)*z   #-dampB*np.cos(argz)*np.exp(-argz)*z
        [ous,ows]=mp.rotate(outer_u,outer_w,al)
        print(ous[-1],ows[-1])


        ax.plot(yp_dat,u_sfc/us_dat, c=c,lw=2,ls='-',label=r'$Re={}$'.format(re),alpha=0.5)
        ax.plot(yp,up,               c=c,lw=2,ls=':')

        if  re==1600 or re == 1300 :
            ax2.plot(ym_dat,(u_sfc-u_sfc[-1])/us_dat,c=c,lw=2,ls='-',label=r'$Re={}$ (DNS)'.format(re),alpha=0.5)
            ax3.plot(ym_dat,(w_sfc-w_sfc[-1])/us_dat,c=c,lw=2,ls='-',label=r'$Re={}$ (DNS)'.format(re),alpha=0.5)
            #ax2.plot(ym,(up-up[-1]),c=c,lw=1,ls='--',label=r'$Re={}$ (model)'.format(re))
            #ax3.plot(ym,(wp-wp[-1]),c=c,lw=1,ls=':',label=r'$Re={}$ (model)'.format(re))
        else:
            ax2.plot(ym_dat,(u_sfc-u_sfc[-1])/us_dat,c=c,lw=1,ls='-',label=r'$Re={}$'.format(re),alpha=0.5)
            ax3.plot(ym_dat,(w_sfc-w_sfc[-1])/us_dat,c=c,lw=1,ls='-',alpha=0.5,label=r'$Re={}$'.format(re))
            #ax2.plot(ym,(up-up[-1]),c=c,lw=1,ls='--')
            #ax3.plot(ym,(wp-wp[-1]),c=c,lw=1,ls=':')
        i03=mp.geti(ym,0.3)
        ax2.plot(ym[i03:],ous[i03:]-ous[-1],ls='--',lw=1, c=c, label=r'  (ekman)')
        ax2.plot(ym[:i03],ous[:i03]-ous[-1],ls='--',lw=1, c=c,alpha=0.5)
        ax2.scatter(ym[i03],ous[i03]-ous[-1],c=c)

        ax3.plot(ym[i03:],-ows[i03:]+ows[-1],ls='--',lw=1,c=c)
        ax3.plot(ym[:i03],-ows[:i03]+ows[-1],ls='--',lw=1,c=c,alpha=0.5)
        ax3.scatter(ym[i03],-ows[i03]+ows[-1],c=c)
        

    ulog=sc.profile_log(yp) 
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

    
    ax2.plot([0,1.5],[0,0],lw=0.5,c='black',ls='-') 
    ax2.set_xlim(0,1.5)
    ax2.set_ylim(-1,1.2)
    ax2.set_xlabel(r'$z^-$')
    ax2.set_ylabel(r'$(U^\alpha - G_{1}^\alpha)/u_\star$')
    lg2=ax2.legend(loc='best')
    lg2.get_frame().set_linewidth(0.0) 
    fig2.savefig('outer_layer_u.pdf',format='pdf')


    ax3.plot([0,1.5],[0,0],lw=0.5,c='black',ls='-') 
    ax3.set_xlim(0.2,1.5)
    ax3.set_ylim(-1.5,0.2)
    ax3.set_xlabel(r'$z^-$')
    ax3.set_ylabel(r'$(V^\alpha - G_{2}^\alpha)/u_\star$')
    lg3=ax3.legend(loc='best')
    lg3.get_frame().set_linewidth(0.0) 
    fig3.savefig('outer_layer_w.pdf',format='pdf')

        

if print_table : 

    print('DATA FROM SC MODEL') 
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

    print(' ')
    print('DATA FROM DNS')
    print('=======================================================================')
    print('  RE Delta+  u*     alpha    d95/d ')
    for re in [500,750,1000,1300,1600]:
        f=Dataset(files[re],'r',format='NETCDF4')
        t=f['t'][:]
        y=f['y'][:]
        us=np.mean(f['FrictionVelocity'][:])
        al=f['FrictionAngle'][:]
        nu=2./(re*re)
        yp=y*us/nu
        deltap=us*us/nu 
        wtop = np.average(f['rW'][:,-1])
        utop = np.average(f['rU'][:,-1])
        al_geo=np.arctan(wtop/utop)/np.pi*180
        al_tot=al_geo-np.mean(al)
        sx=np.mean(f['Rxy'][:,:],0)
        sz=np.mean(f['Ryz'][:,:],0) 
        st=np.sqrt(sx*sx+sz*sz)
        ny=len(yp)
        i95=ny-mp.geti(np.flip(st),0.05*us*us)

        y=f['y']

        dy=y[1:]-y[:-1]
        tke = np.average(f['Tke'][:,:],0) 
        eps = np.average(f['Eps'][:,:],0)
        tke_int = (tke[1:] + tke[:-1])*dy/2
        eps_int = (eps[1:] + eps[:-1])*dy/2

        idelta=len(y)#mp.geti(y,us) 
        tke_int = np.sum(tke_int[:idelta])
        eps_int = np.sum(eps_int[:idelta]) 
        print(' {0:4d} {1:5.0f} {2:7.4f} {3:7.4f} {4:6.3f} {5:6.3g} {6:8.2e} {7:8.2e} {8:5.2f} {9:5.2f}'.format(re,deltap,us,al_tot,yp[i95]/deltap,tke_int/us**3,tke_int,eps_int,eps_int/us**3,yp[1]))
        
if plot_outer_log : 

    fig1=plt.figure(figsize=(12,5))
    ax1=fig1.add_axes([0.05,0.1,0.4,0.8])
    ax2=fig1.add_axes([0.55,0.1,0.4,0.8])
    ax5=fig1.add_axes([0.11,0.455,0.18,0.43]) 

    fig2=plt.figure(figsize=(12,5))
    ax3=fig2.add_axes([0.055,0.1,0.4,0.8])
    ax4=fig2.add_axes([0.55,0.1,0.4,0.8])
    ax6=fig2.add_axes([0.265,0.18,0.18,0.35]) 

    fig3=plt.figure(figsize=(6,5))
    ax7=fig3.add_axes([0.1,0.1,0.8,0.8]) 

    fig4=plt.figure(figsize=(6,5))
    ax8=fig4.add_axes([0.13,0.1,0.8,0.8])
    ylog=10**np.arange(0.7,3.5,0.01)
    alog=-46.5
    blog=29.
    clog=0.68 
    mlog=alog+blog*np.log(ylog) + clog*ylog
    mlog[1]=(mlog[0]+mlog[2])/2.

    yvisc=np.arange(0,30,1) 
    y10 = 9. 
    f10 = alog + blog*np.log(y10) + clog*y10
    d10 = blog/y10 + clog
    print('FITTING BASED ON:', f10,d10)

    # first guess
    bvisc = -0.2353
    avisc = f10 / (1 -np.exp(bvisc*y10) + bvisc*y10) #est'd from f10 

    #CHECK
    mvisc = avisc + avisc*bvisc*yvisc - avisc*np.exp(bvisc*yvisc)
    print('F0: ',mvisc[0]  )
    print('F10:',f10, avisc+avisc*bvisc*y10 - avisc*np.exp(bvisc*y10))
    print('D0: ',avisc*bvisc-avisc*bvisc,(mvisc[1]-mvisc[0])/(yvisc[1]-yvisc[0]))
    print('D10:',avisc*bvisc - avisc*bvisc*np.exp(bvisc*y10), d10)
    print(avisc,bvisc)
    ax8.plot(yvisc,   mvisc,   lw=2,c='blue', alpha=0.5,label=r'$a_\mathrm{visc}(b_\mathrm{visc}z^+ + 1-e^{b_\mathrm{visc} z^+})$')
    ax8.plot(ylog[1:],mlog[1:],lw=2,c='black',alpha=0.5,label=r'$a_\mathrm{log} + b_\mathrm{log} log(z^+) + c_\mathrm{log} z^+ $') 
    ax8.plot([y10,y10],[0,100],c='black',lw=0.5,ls=':')
    ax8.plot([0,30],[f10,f10], c='black',lw=0.5,ls=':') 
    print(yvisc,mvisc)

    colors[10000]='gray'
    
    re_use=[500,750,1000,1300,1600,10000]#,10000,150000,1000000]
    for re in re_use:
        c=colors[re]
        nu=2./re**2
        if  re <= 1600 :
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

            us=us_dat
            al=al_sfc1+al_geo
        else :
            us,al=sc.ustar_alpha(re)
            
        dp= us**2 / nu
        ny,yp,ym=build_grid(dp,5)
        
        um,wm=sc.profile_minus(ym,re)
        up,wp=sc.profile_plus(yp,re)
        
        ax1.plot(yp[1:], wp[1:],        c=c,ls='-',lw=2)
        if re==500: 
            ax2.plot(yp[1:], wp[1:]*us*dp,c=c,ls='-',lw=2,label=r'$Re_D={}$ (model)'.format(re))
        else:
            ax2.plot(yp[1:], wp[1:]*us*dp,c=c,ls='-',lw=2,label=r'$Re_D={}$'.format(re))
        i03=mp.geti(yp,0.3*dp)
        ax2.scatter(yp[i03],wp[i03]*us*dp,c=c)
        ax5.plot(yp[1:], wp[1:],        c=c,ls='-',lw=2) 

        if re == 500: 
            ax3.plot(yp[1:],up[1:],         c=c,ls='-',lw=2,label=r'$Re_D={}$ (model)'.format(re))
        else:
            ax3.plot(yp[1:],up[1:],         c=c,ls='-',lw=2,label=r'$Re_D={}$'.format(re))
        ax4.plot(ym[1:],up[1:]-up[-1],  c=c,ls='-',lw=2,label=r'$Re_D={}$'.format(re))
        ax6.plot(yp[1:],up[1:],         c=c,ls='-',lw=1)

        i1=mp.geti(ym,0.001)
        ax1.scatter(yp[i1],wp[i1],marker='D',c=c,s=20)

        if  re > 1600 :
            continue; 
        

        ax1.plot(yp_dat[1:],w_sfc[1:]/us_dat,       c=c,ls='--',lw=1)
        if re == 500:
            ax2.plot(yp_dat[0:],w_sfc[0:]*dp,           c=c,ls='--',lw=1,label='    (DNS)')
        else: 
            ax2.plot(yp_dat[0:],w_sfc[0:]*dp,           c=c,ls='--',lw=1)
        ax5.plot(yp_dat[:],w_sfc[:]/us_dat,         c=c,ls='--',lw=1)

        if re == 500: 
            ax3.plot(yp_dat[1:],u_sfc[1:]/us_dat,            c=c,ls='--',lw=2,label='    (DNS)')
        else :
            ax3.plot(yp_dat[1:],u_sfc[1:]/us_dat,            c=c,ls='--',lw=2)
        ax4.plot(ym_dat[1:],(u_sfc[1:]-u_sfc[-1])/us_dat,c=c,ls='--',lw=2)
        ax6.plot(yp_dat[:],u_sfc[:]/us_dat,              c=c,ls='--',lw=2)

        dp_dat = us_dat*us_dat/nu
        ax7.plot(yp_dat[:20],w_sfc[:20]*dp_dat,    c=c,ls='--',lw=1,label=r'$DNS: Re_D={}$'.format(re))
        ax8.plot(yp_dat,     w_sfc*dp_dat,         c=c,ls='--',lw=1,label=r'$DNS: Re_D={}$'.format(re))
        ctr=0.28-2.25*np.sqrt(1./re)
        ictr=mp.geti(yp_dat,dp*ctr) 
        print('CENTER_LOC:', ctr,yp_dat[ictr],yp_dat[ictr]/dp,ictr) 
        ax8.scatter(yp_dat[ictr],w_sfc[ictr]*dp_dat,     c=c) 
        
    ax7.plot(yp[:20],wp[:20]*us*dp,     c=c,ls='-',lw=2,label=r'$Model: Re_D={}$'.format(re),alpha=0.5)
    ax6.plot(yp[:15],yp[:15],ls=':',lw=2,c='black',label='$z^+$')
    ax2.plot([1e-3,1e2],[0,0],lw=0.5,ls='-',c='black')

    c4=-3.6e-4
    c6=4.6e-6
    ur=14.5# 12.78
    ui= ( yp + c4*yp**4+c6*yp**6 ) / (1+ c6/ur*yp**6)
    ax3.plot(yp,ui,ls='--',lw=1,c='black',label=r'$U_\mathrm{inner}^{\alpha_\star+}$')
    ax3.plot(yp[10:],sc.profile_log(yp[10:]),ls=':',c='black',label=r'$\kappa^{-1} \log z^+$') 
    lg3=ax3.legend(loc='upper left')
    lg3.get_frame().set_linewidth(0.0) 

    av=18.8524729
    bv=0.2353
    ax2.plot(yp,av* (bv*yp - 1 + np.exp(-bv*yp) ),ls='--',lw=4,c='black',label=r'$V_\mathrm{inner}^{\alpha}/G \delta^+$',alpha=0.5) 
    print('AT 10:',bv*10,np.exp(-bv*10),av*( (bv*10) -1 + np.exp(-bv*10)))
    lg6=ax6.legend(loc='best')
    lg6.get_frame().set_linewidth(0.0)
    
    ax1.set_xscale('log')
    ax1.set_xlim(1,1e3)
    ax1.set_ylim(0,5)
    ax1.set_xlabel(r'$z^+$')
    ax1.set_ylabel(r'$\left(V^{\alpha}\right)^+$')
    
    ax2.set_xlim(1e0,2e3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(3e-1,2e3)
    ax2.set_xlabel(r'$z^-$')
    ax2.set_ylabel(r'$V^{\alpha+} \dfrac{\delta^+}{Z_\star}$') 

    ax5.set_xscale('linear')
    ax5.set_xlim(0,15)
    ax5.set_ylim(-0.1,0.5)
    ax5.set_xlabel(r'$z^+$')
    ax5.set_ylabel(r'$V^{\alpha+}$') 

    ax3.set_xscale('log')
    ax3.set_xlim(1,1e3)
    ax3.set_ylim(0,22)
    ax3.set_xlabel(r'$z^+$')
    ax3.set_ylabel(r'$U^{\alpha +}$') 

    ax4.plot([0,2],[0,0],ls=':',lw=1,c='black')
    ax4.set_xscale('log')
    ax4.set_xlim(1e-2,3)
    ax4.set_ylim(-10,2) 
    ax4.set_xlabel(r'$z^-$')
    ax4.set_ylabel(r'$U^{\alpha+}-G^{\alpha+}$')

    ax6.set_xscale('linear')
    ax6.set_xlim(0,14.8)
    ax6.set_ylim(0,9.8)
    ax6.set_xlabel(r'$z^+$')
    ax6.set_ylabel(r'$U^{\alpha+}$')

    ax7.set_xscale('linear')
    ax7.set_xlim(0,12)
    ax7.set_ylim(0,35)
    ax7.set_xlabel(r'$z^+$')
    ax7.set_ylabel(r'$V^+ \dfrac{u_\star}{G} \mathrm{Re}_\tau$') 
    
    ax8.set_xscale('log')
    ax8.set_yscale('log')
    ax8.set_xlim(1,1e3)
    ax8.set_ylim(2e-1,1e3)
    ax8.set_xlabel(r'$z^+$')
    ax8.set_ylabel(r'$V^+ \dfrac{u_\star}{G} \mathrm{Re}_\tau$') 
    
    lg1=ax2.legend(loc='best')
    lg1.get_frame().set_linewidth(0.0) 

    lg2=ax4.legend(loc='best')
    lg2.get_frame().set_linewidth(0.0)

    lg3=ax7.legend(loc='best')
    lg3.get_frame().set_linewidth(0.0) 

    lg4=ax8.legend(loc='best')
    lg4.get_frame().set_linewidth(0.0)
    
    fig1.savefig('w_profile.pdf',format='pdf')
    fig2.savefig('u_profile.pdf',format='pdf')
    plt.show()
    fig3.savefig('w_viscous.pdf',format='pdf')
    fig4.savefig('w_log.pdf',format='pdf') 

    fig1.clf()
    fig2.clf() 
        
if plot_summary : 

    fig=plt.figure(figsize=(13,5))
    ax1=fig.add_axes([0.06,0.09,0.27,0.8])
    ax2=fig.add_axes([0.39,0.09,0.27,0.8])
    ax3=fig.add_axes([0.72,0.09,0.27,0.8])


    for re_loc in [500,750,1000,1300,1600,5000,30000,150000,1000000]:#,300000,6e6]:#,5000,10000,20000,40000,80000,160000]:
        c_loc=colors[re_loc]
        nu_loc=2./re_loc**2
        us_loc,al_loc=sc.ustar_alpha(re_loc)
        deltap_loc = us_loc**2/nu_loc
        ny_loc,yp_loc,ym_loc=build_grid(deltap_loc,5)
        
        um_loc,wm_loc=sc.profile_minus(ym_loc,re_loc)
        up_loc,wp_loc=sc.profile_plus(yp_loc,re_loc)
        
        sqdp = 1.#/(deltap_loc)

        if re_loc < 1601: 
            alpha_loc= re_loc/1601 % 1
        else :
            alpha_loc = 1. 
        
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

        print('RE:',  re_loc, 'u*:', us_loc, 'transparency:',alpha_loc) 
        print('UP:', np.amin(up_loc),np.amax(up_loc),um_loc[-1]/us_loc)
        print('WP:', np.amin(wp_loc),np.amax(wp_loc),wm_loc[-1]/us_loc)

        if re_loc < 1601: 
            ax2.text(yp_loc[-1]*0.5, (1.+(7./re_loc))*up_loc[-1], 'Re='+str(re_loc),c='black')#c_loc)

        ax1.plot(ym_loc[1:], (um_loc[1:]-um_loc[-1])/us_loc, c='blue', lw=2,ls='-',alpha=alpha_loc)
        ax1.plot(ym_loc[1:], (wm_loc[1:]-wm_loc[-1])/us_loc, c='red',  lw=2,ls='-',alpha=alpha_loc)

        if use_data and re_loc < 1600 : 
            ax1.plot(ym_dat[1:], (u_sfc[1:]-u_sfc[-1])/us_dat,   c=c_loc, lw=0.5,ls='--',alpha=alpha_loc)
            ax1.plot(ym_dat[1:], (w_sfc[1:]-w_sfc[-1])/us_dat,   c=c_loc, lw=0.5,ls='--',alpha=alpha_loc)
            veer = -np.arctan(w_sfc/u_sfc)/D2R
            ax1.plot(ym_dat, -10*veer/veer[-1],c='black',lw=1,ls='-',alpha=alpha_loc)
            il=mp.geti(yp_dat,18)-1

            ax3.plot(u_geo,-w_geo,ls='--',c=c_loc,lw=0.5,alpha=alpha_loc)
        
        umr_loc,wmr_loc=mp.rotate(um_loc,-wm_loc,-al_loc) 

        ax3.plot(umr_loc,wmr_loc,alpha=alpha_loc,c='black') #c=c_loc)
    
        i_loc = mp.geti(wmr_loc,np.amax(wmr_loc))
        if re_loc in [500, 1600,5000,30000,1000000]:
            ax3.text(umr_loc[i_loc]*0+0.8,wmr_loc[i_loc],'Re='+str(re_loc),c='black',alpha=alpha_loc)#c_loc)
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

    re_les = np.array([1e3,15e4,1e6])
    us_les = np.array([0.0532,0.0257,0.0211]) 
    al_les = np.array([20.17,8.59,7.03])
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
    fig=plt.figure(figsize=(6,3))
    ax=fig.add_axes([0.12,0.15,0.85,0.83])
    ax.plot(re_arr,1./us_arr,label=r'$G/u_\star$',c='blue')
    us_est=3.9*np.log(re_arr)-8
    ax.plot(re_arr,us_est,ls=':',lw=1,c='black',label=r'$u_\star=4\log(Re)-8$') 
    ax.plot(re_arr,al_arr/D2R,label=r'$\alpha$',c='red')
    lg=plt.legend(loc='center right')
    ax.set_xlim(4e2,1.5e6)
    ax.set_ylim(0,49)
    ax.set_xscale('log')
    ax.set_xlabel(r'$Re_D$')
    ax.set_ylabel(r'$(G/u_{\star}), \alpha$') 
    lg.get_frame().set_linewidth(0.0)
    plt.savefig('ustar_alpha_1.pdf')
    
    ax.scatter(re_sim,1./us_sim,c='blue')
    ax.scatter(re_sim,al_sim,c='red')
    lg=plt.legend(loc='center right')
    lg.get_frame().set_linewidth(0.0)
    plt.savefig('ustar_alpha_2.pdf')
    ax.scatter(re_les,al_les,   marker='x',c='red',label='PALM-LES')
    ax.scatter(re_les,1./us_les,marker='x',c='blue')
    lg=plt.legend(loc='center right') 
    lg.get_frame().set_linewidth(0.0)
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
    
    for re in [500,750,1000,1300,1600,10000,1000000]:
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
            al= - al/180*np.pi + al_geo
        else:
            us,al=sc.ustar_alpha(re)
            deltap=us*us/nu 
            ny,yp,ym=build_grid(deltap,2.5) 
            
        up_mod,wp_mod=sc.profile_plus(yp,re)
        um_mod,wm_mod=sc.profile_minus(ym,re)
        vp_mod = np.sqrt(up_mod**2+wp_mod**2)
        vm_mod = np.sqrt(um_mod**2+wm_mod**2)
        d_mod = np.zeros(len(yp))
        d_mod[1:] = -( np.arctan(up_mod[1:]/wp_mod[1:]) - np.pi/2) / np.pi*180 

        for i in range(len(d_mod)):
            if wp_mod[i] < 1e-6:
                d_mod[i]=0. 

        ax1.plot(yp[1:], vp_mod[1:],ls='-', lw=3,c=colors[re],alpha=0.5 ) 
        ax1.plot(yp[1:],d_mod[1:],  ls='--',lw=3,c=colors[re],alpha=0.5) 
        ax2.plot(ym[1:],(vm_mod[1:]-vm_mod[-1])/us,ls='-', lw=3,c=colors[re],alpha=0.5) 
        ax2.plot(ym[1:],d_mod[1:]-d_mod[-1],       ls='--',lw=3,c=colors[re],alpha=0.5)

        deltap=us*us/nu
        i0=mp.geti(yp,10) 
        i1=mp.geti(yp,1.5*np.sqrt(deltap))
        i2=mp.geti(yp,0.27*deltap)
        
        c_off=35./(deltap*us)+0.04
        c1=26/(deltap*us) 
        print('RE:',re,'ALPHA: ', al,c1,c_off)
        dir_fit=np.zeros(len(yp))
        dir_fit[1:]=c1*np.log(yp[1:])**2.0+c_off
        w_inner=np.tan(dir_fit/180*np.pi)*up_mod
        
        ax1.plot(yp[:50],dir_fit[:50],c=colors[re],ls='-',lw=1)

        i_03=mp.geti(yp,0.30*deltap)
        z_03=yp[i_03] 
        s_03=(wp_mod[i_03+1]-wp_mod[i_03-1])/(yp[i_03+1]-yp[i_03-1]) 
        a_03=s_03*z_03
        o_03=wp_mod[i_03]-a_03*np.log(z_03)

        C1=(w_inner[i0+1]-w_inner[i0-1])/(yp[i0+1]-yp[i0-1])
        ap = ( wp_mod[i2] - w_inner[i0] - C1 * (yp[i2]-yp[i0]) ) / ( np.log(yp[i2]/yp[i0]) - yp[i2]/yp[i0]) 
        bt = C1 - ap/yp[i0]
        w_inner[i0:] = w_inner[i0] + ap * np.log(yp[i0:]/yp[i0]) + bt*(yp[i0:]-yp[i0])
        dir2=np.arctan(w_inner[1:]/up_mod[1:])*180/np.pi
        ax1.scatter([yp[i0],yp[i_03]],[dir2[i0],d_mod[i_03]],c=colors[re])
        ax1.plot(yp[i0+1:],dir2[i0:],c=colors[re],ls=':',lw=0.5)
        
        ylen=mp.geti(ym,0.1)
        y_use=yp[i_03-ylen:i_03+ylen]

        ym=yp/deltap 
        argz=0.66*(2.*np.pi*(ym+0.12)) 
        dampA =  0.42*(us/0.05- 0.07/re) #np.exp(zdum) * ( dU_mtc*np.cos(zdum) + dW_mtc*np.sin(zdum) )
        dampB =  0.0#np.exp(zdum) * ( dU_mtc*np.sin(zdum) + dW_mtc*np.cos(zdum) )
        outer_u= (1-(dampA*np.cos(argz)+dampB*np.sin(argz))*np.exp(-argz))/us
        outer_w=   ( dampA*np.sin(argz)-dampB*np.cos(argz))*np.exp(-argz)/us
        print('ROT:', re,al)
        [ous,ows]=mp.rotate(outer_u,outer_w,al)
        odir = np.arctan(-ows/ous)*180/np.pi
        ax1.plot(y_use,odir[i_03-ylen:i_03+ylen],c=colors[re],ls=':',lw=2) 
        

    ax1.plot([-1,-1],[1,1],ls='-', c='black',label=r'$U_{mag}$')
    ax1.plot([-1,-1],[1,1],ls='--',c='black',label=r'DIR') 

    ax2.plot([1e-4,2],[0,0],ls=':',c='black',lw=1) 
    
    ax1.set_xscale('log')
    ax1.set_xlim(1,1e4)
    ax1.set_ylim(0,25)
    ax1.set_xlabel(r'$z^+$')
    ax1.set_ylabel(r'$U_{mag}^+,  \alpha_{sfc}$')

    ax2.set_xscale('log')
    ax2.set_xlim(1e-3,2)
    ax2.set_ylim(-25,2)
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

if plot_convergence:
    re_arr=[500,750,1000,1300,1301,1600] 
    plt.close('all') 
    fig1=plt.figure(figsize=(8,6))
    ax1=fig1.add_axes([0.1,0.1,0.8,0.8]) 
    fig2=plt.figure(figsize=(8,6))
    ax2=fig2.add_axes([0.1,0.1,0.8,0.8])

    re_col= {500:'gray',750:'blue',1000:'pink',1300:'green',1301:'green',1600:'black'}

    for re in re_arr: 
        f=Dataset(files[re],'r',format='NETCDF4')
        nu=2./(re*re)

        t_all=f['t'][:]
        i0=mp.geti(t_all,f['t'][-1]-2*np.pi)

        print(i0,len(t_all),f['t'][-1]-f['t'][i0]) 

        us=f['FrictionVelocity'][i0:]
        al=f['FrictionAngle'][i0:]
        u_dat=f['rU'][i0:,:]
        w_dat=f['rW'][i0:,:]
        y_dat=f['y'][:]
        t=f['t'][i0:] 
        us_avg=np.average(us)
        al_avg=np.average(al)
    
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


        fangle=f['FrictionAngle'][i0:]
        delta_al=al[0,1]-fangle[0]
        al_avg=np.mean(fangle) 

        print('====', re)
        print(t[0],t[-1],us_avg,al_avg)
        us_anom=1/us-1/us_avg
        al_anom=al[:,1]-np.average(al[:,1])
        ax1.plot((t-t[0])/(2*np.pi),us_anom,label=r'Re={} $Z-<Z>$'.format(re),  c=re_col[re],ls='-')
        ax1.plot((t-t[0])/(2*np.pi),al_anom*D2R,label=r'$\alpha-<\alpha>$',c=re_col[re],ls=':')

    lg=ax1.legend(loc='best')
    lg.get_frame().set_linewidth(0.0) 
    fig1.savefig('convergence_usalpha.pdf',format='pdf') 


if plot_les_comp == True:
    les_path='../data/'
    f_cor=1.03e-04
    les_ReD1000_10cm={'f':'N_ReD1000_10cm_big_pr.001.nc',
                      'z0m':0.05,
                      'z0':0.00208,
                      'c':'red',
                      're':1e3}
    les_ReD15E4_10m={'f':'N_ReD15E4_10m_big_pr.000.nc',
                     'z0m':5.00,
                     'z0':0.0000287,
                     'c':'blue',
                     're':1.5e5} 
    les_ReD1e6_25m= {'f':'N_ReD1e6_25m_pr.000.nc',
                     'z0m':12.5,
                     'z0':0.00000519,
                     'c':'green',
                     're':1e6} 

    les_cases = [les_ReD1000_10cm, les_ReD15E4_10m, les_ReD1e6_25m] 
    
    fig=plt.figure(figsize=(12,4))
    ax1=fig.add_axes([0.05,0.11,0.27,0.8])
    ax2=fig.add_axes([0.38,0.11,0.27,0.8])
    ax3=fig.add_axes([0.71,0.11,0.27,0.8])
    
    for c in les_cases:
        f_nc='{}{}'.format(les_path,c['f'])
        f_h=Dataset(f_nc,'r')

        t=f_h['time'][:] 
        t_last=t[-1]
        i_srt=mp.geti(t,t_last-1.03e4*np.pi*2)
        t=t[i_srt:]
        print('')
        print('Plotting LES Data for Re=',c['re']) 
        print('  STARTING FROM INDEX', i_srt, '(nt=', len(t), ') time-RANGE: ', (t[-1]-t[0])/1.e4/np.pi/2,'inertial periods' )
        u=f_h['u'][i_srt:,:]
        v=-f_h['v'][i_srt:,:]
        w=f_h['w'][i_srt:,:]

        z=f_h['zu'][:]

        
        std=np.array([np.sqrt(np.var(xi,axis=0)) for xi in [u,v,w]])
        avg=np.array([np.average(xi,axis=0) for xi in [u,v,w]])

        c0=-25./12.; c1=4; c2=-3; c3=4./3.; c4=-1./4. 
        d4_sfc= 4.*(c0*avg[:,0] +c1*avg[:,1]+c2*avg[:,2]+c3*avg[:,3]+c4*avg[:,4])/(z[4]-z[0])
        d1_sfc=(avg[:,1]-avg[:,0])/(z[1]-z[0])
        al4_sfc=np.arctan(d4_sfc[1]/d4_sfc[0])/np.pi*180
        al1_sfc=np.arctan(d1_sfc[1]/d1_sfc[0])/np.pi*180
        [ur,vr] = mp.rotate(u,v,al4_sfc*np.pi/180)
        avg[0]=np.average(ur,axis=0)
        avg[1]=np.average(vr,axis=0)

        u_geo=avg[0][-1]
        v_geo=avg[1][-1] 
        t_geo=np.sqrt(u_geo**2+v_geo**2) 

        
        # zp   = z*us/nu
        # ulog = us/kappa log(z/z0)
        z0 = c['z0']
        KAPPA=0.40
        log_level=1
        nu_est=2*(t_geo)**2 / (c['re']**2 * f_cor) 
        us = avg[0][log_level] * KAPPA / np.log(z[log_level]/z0)
        zp=z*us/nu_est
        zm=z/us*f_cor
        
        print('  - roughness length                   [m]:',c['z0']) 
        print('  - RE (CALCULATED)                    [1]:',t_geo*np.sqrt(2./(f_cor*nu_est)))
        print('  - nu (estimated from exact Re)   [m^2/s]:',nu_est )
        print('  - outer velocity (u,v,magnitude)   [m/s]:',u_geo,v_geo,t_geo)
        print('  - domain rotation w.r.t. sfc-stress[deg]:',al4_sfc)
        print('  - total surface veering            [deg]:',np.arctan(v_geo/u_geo)/np.pi*180+al4_sfc)
        print('  - geostrophic drag u*/G              [1]:',us/t_geo) 

        
        z_ana=np.arange(0,8,0.1)
        z_ana=10**z_ana
        us_mod,al_mod=sc.ustar_alpha(c['re'])
        dp_mod=us_mod*us_mod*(c['re']**2)/2.
        print('CHECK:',c['re'],us,dp_mod)
        ny_mod,zp_mod,zm_mod=build_grid(dp_mod)
        up_mod,wp_mod=sc.profile_plus(zp_mod,c['re']) 
        
        ax1.fill_between(zp[1:], (avg[0,1:]-2*std[0,1:])/us,(avg[0,1:]+2*std[0,1:])/us,alpha=0.5,color='gray',edgecolor=None)
        ax1.plot(zp[1:], avg[0,1:]/us,ls='-',c=c['c'],lw=2,label='{}'.format(c['f'].split('_')[1]),alpha=0.5)
        ax1.plot(1.15*zp_mod[1:], up_mod[1:],  ls=':',c=c['c'],lw=0.5)

        ax1.fill_between(zp[1:],(avg[1,1:]-2*std[1,1:])/us,(avg[1,1:]+2*std[1,1:])/us,alpha=0.5,color='gray',edgecolor=None)
        ax1.plot(zp[1:],avg[1,1:]/us, ls='-',c=c['c'],lw=2,alpha=0.5)
        ax1.plot(1.15*zp_mod[1:], wp_mod[1:],  ls=':',c=c['c'],lw=0.5)

        ax2.fill_between(zm[1:],(avg[0,1:]-avg[0,-1]-std[0,1:])/us,(avg[0,1:]-avg[0,-1]+std[0,1:])/us,alpha=0.3,color=c['c'],edgecolor=c['c'],hatch='x') 
        ax2.plot(zm[1:],(avg[0,1:]-u_geo)/us,   lw=2.0,ls='-',c=c['c'],alpha=0.5)
        ax2.plot(zm_mod[1:],(up_mod[1:]-up_mod[-1]),lw=0.5,ls=':',c=c['c'])
        
        ax2.fill_between(zm[1:],(avg[1,1:]-avg[1,-1]-std[1,1:])/us,(avg[1,1:]-avg[1,-1]+std[1,1:])/us,alpha=0.3,color=c['c'],edgecolor=c['c'],hatch='x')
        ax2.plot(zm[1:],(avg[1,1:]-v_geo)/us,   lw=2.0,ls='-',c=c['c'],alpha=0.5)
        ax2.plot(zm_mod[1:],(wp_mod[1:]-wp_mod[-1]),lw=0.5,ls=':',c=c['c'])


        al=np.arctan(avg[1,-1]/avg[0,-1])
        
        [uo,vo] = mp.rotate(avg[0],avg[1],al)
        [uo_mod,vo_mod] = mp.rotate(up_mod,wp_mod,al_mod)
        t_mod=np.sqrt(wp_mod[-1]**2+up_mod[-1]**2) 
        
        ax3.plot(uo/t_geo,-vo/t_geo,c=c['c'],ls='-',lw=2,alpha=0.5,label='Re={0:6.2g}'.format(c['re']) )
        ax3.scatter(uo/t_geo,-vo/t_geo,s=1,c=c['c'],marker='x')
        ax3.plot(uo_mod/t_mod,-vo_mod/t_mod,c=c['c'],ls=':',lw=0.5) 

        
    ax1.plot(z_ana, np.log(5*z_ana)/KAPPA,ls=':',lw=0.5,c='black',label='log-law (PALM-est\'d)')
    ax1.plot(z_ana, np.log(z_ana)/0.416+5.416,ls='-',lw=0.5,c='black',label='log-law (DNS-based)')
    ax1.axhline(c='black',lw=0.5,ls='-') 
    ax1.set_xscale('log')
    ax1.set_ylim(-5,50)
    ax1.set_xlim(5,5e8)
    ax1.set_xlabel(r'$z^+$')
    ax1.set_ylabel(r'$U^\alpha/u_\star,\ V^\alpha/u_\star$')
    ax1.set_title('inner scaling (surface-layer profiles)') 

    lg1=ax1.legend(loc='upper left')
    lg1.get_frame().set_linewidth(0.0)

    ax2.axhline(c='black',lw=0.5,ls='-')
    ax2.set_ylabel(r'$(U^\alpha-U_G^\alpha)/u_\star,\ (V^\alpha-V_G^\alpha)/u_\star$')
    ax2.set_xlabel(r'$z^-$') 
    ax2.set_xlim(0,1.2)
    ax2.set_ylim(-5,1.5) 
    ax2.set_title('outer scaling (velocity deficit)')

    x=np.arange(0,12,0.1) 
    u_lam=1-np.cos(x)*np.exp(-x)
    v_lam=np.sin(x)*np.exp(-x)
    ax3.plot(u_lam,v_lam,c='black',ls='-',lw=1,label='laminar') 
    ax3.axhline(c='black',lw=0.5,ls='-')
    ax3.axvline(1,c='black',lw=0.5,ls='-')
    ax3.set_title('Hodograph') 

    lg3=ax3.legend(loc='best')
    lg3.get_frame().set_linewidth(0.0) 
    plt.savefig('les_comparison.pdf',format='pdf') 
    plt.close('all') 
        


if plot_applications == True:
    print('plotting applications')
    re_arr=np.arange(3.5,10.1,0.5)
    re_arr=10**re_arr
    cf_arr=np.zeros(len(re_arr))
    df_arr=np.zeros(len(re_arr))
    tn_arr1=np.zeros(len(re_arr))
    tn_arr2=np.zeros(len(re_arr))
    tn_arr3=np.zeros(len(re_arr)) 
    i=0
    for re in re_arr:
        print('Calculating for re=',re) 
        nu=2./re**2
        us,al=sc.ustar_alpha(re)
        dp=us**2/nu
        ny,yp,ym=build_grid(dp,10.5)
        [um,wm]=sc.profile_minus(ym,re)

        [um,wm]=mp.rotate(um,wm,al)
        print(um[-1],wm[-1])
        wm=-wm
        cf_arr[i]=mp.integrate(ym,wm,len(ym))
        df_arr[i]=mp.integrate(ym,um,len(ym))
        df_arr[i]-=ym[-1]

        idx=[ mp.geti(ym,zs) for zs in [0.05,0.15,0.25,0.5] ]
        d = [ np.arctan(wm[i]/um[i]) for i in idx ]
        tn_arr1[i] = (d[0]-d[1] )/np.pi*180
        tn_arr2[i] = (d[1]-d[2] )/np.pi*180
        tn_arr3[i] = (d[2]-d[3] )/np.pi*180 

        i+=1 
    
    print(df_arr)
    fig=plt.figure(figsize=(4,3))
    ax=fig.add_axes([0.1,0.15,0.8,0.8])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Re_D$') 
    ax.plot(re_arr,cf_arr,label=r'$\frac{1}{G\delta}\int_0^{\infty} V^G dz$',c='black')
    ax.plot(re_arr,0.32/np.log(re_arr),ls=':',label=r'$0.32/log(Re_D)$',c='black')
    ax.plot(re_arr,-df_arr,label=r'$\frac{1}{G\delta}\int_0^{\infty} (G-U^G)dz$',ls='--',c='blue')
    ax.plot(re_arr, df_arr,label=r'$\frac{1}{G\delta}\int_0^{\infty} (U^G-G)dz$',ls='--',c='red')
    ax.set_ylim(2e-4,5e-2)
    lg=plt.legend(loc='best')
    lg.get_frame().set_linewidth(0.0) 
    plt.savefig('mass_transport.pdf',format='pdf')

    
    fig=plt.figure(figsize=(4,3))
    ax=fig.add_axes([0.15,0.15,0.8,0.8])
    ax.set_xscale('log')
    ax.set_xlabel(r'$Re_D$')
    ax.set_ylabel(r'$\Delta_\alpha [deg]$') 
    ax.plot(re_arr,tn_arr1,label=r'Turning $z^-\simeq 0.05 - 0.15$')
    ax.plot(re_arr,tn_arr2,label=r'Turning $z^-\simeq 0.15 - 0.25$')
    ax.plot(re_arr,tn_arr3,label=r'Turning $z^-\simeq 0.25 - 0.50$')

    lg=plt.legend(loc='best')
    lg.get_frame().set_linewidth(0.0) 
    plt.savefig('turning.pdf',format='pdf')
    plt.close('all') 

if test_inner_streamwise:
    re=2000
    nu=2./(re*re) 
    us,al=sc.ustar_alpha(re)
    dp=us**2/nu
    yp=np.arange(2000)/4.
    [up,wp]=sc.profile_plus(yp,re)
    ulog=sc.profile_log(yp)

    c2=-.00; o2=0;
    c3=-0.00; o3=0;
    c4=3.1e-5; o4=0;
    c5=-2e-7
    a_corr=1.0
    yp_mtc = 19 
    uvsc=yp/(1+0.00185*yp**2)
    corr=(0.195*yp-3.569861)*(1.+np.tanh(0.2*(yp-22)))/2. + 0.40*np.exp(-0.035*(yp-22)**2)
    i40=mp.geti(yp,40)
    print(uvsc[i40]+corr[i40],ulog[i40])
    fig=plt.figure(figsize=(4,3))
    ax=fig.add_axes([0.1,0.1,0.85,0.85])

    ax.plot(yp,up,  c='red',ls='-',lw=1)
    ax.plot(yp[i40:],ulog[i40:],c='black',ls=':',lw=1)
    ax.plot(yp[:i40],uvsc[:i40]+corr[:i40],c='blue',ls='--',lw=1)
    ax.set_xlim(0,8e1)
    ax.set_ylim(-4,3.5e1) 

    plt.savefig('innertest.pdf',format='pdf')
    plt.close('all')

ReD_col={500: 'blue',
         1000:'orange',
         1600:'red',
         10000:'gray',
         40000:'black' }

if plot_shear_vs_rotation: 

    ReD_arr = [ 500, 1000, 1600 , 40000]
    fig=plt.figure(figsize=(5,3))
    fig2=plt.figure(figsize=(5,3))
    ax=fig.add_axes([0.15,0.15,0.83,0.80]) 
    ax2=fig2.add_axes([0.15,0.15,0.83,0.80]) 


    for re in ReD_arr: 

        us,al=sc.ustar_alpha(re)
        al=al/np.pi*180
        reTau = (re*us)**2/2. 
        ny,yp,ym=build_grid(reTau,thick=3)

        up,vp=sc.profile_plus(yp,re)
        um,vm=up/us,vp/us
        up_z=np.gradient(up,yp,edge_order=2)
        vp_z=np.gradient(vp,yp,edge_order=2)

        alpha = np.arctan2(vp,up)
        alpha_shear = np.arctan2(-vp_z,-up_z)
        if re == 500:
            ax.plot(ym,alpha,label='Re={} (DD)'.format(re),c=ReD_col[re],ls='--',alpha=0.5,lw=3)
            ax.plot(ym,alpha_shear-alpha,label='stress vs. DD',c=ReD_col[re],ls='-')
        else:
            ax.plot(ym,alpha,label='Re={}'.format(re),c=ReD_col[re],ls='--',alpha=0.5,lw=3)
            ax.plot(ym,alpha_shear-alpha,c=ReD_col[re],ls='-')

        if re in files.keys():
            f_nc = Dataset(files[re],'r')
            u_geo=f_nc.variables['rU'][-1,-1]
            w_geo=f_nc.variables['rW'][-1,-1]
            al_geo = np.arctan2(w_geo,u_geo)
            print(al_geo,al/180*np.pi)
            u_flx = np.mean(f_nc.variables['Rxy'][:,:],axis=0)
            w_flx = np.mean(f_nc.variables['Ryz'][:,:],axis=0) 
            y_dat=f_nc.variables['y']
            us_dat=np.mean(f_nc.variables['FrictionVelocity'][:])
            ym_dat =y_dat/us_dat 
            print(w_flx.shape,u_flx.shape)
            print(f_nc.variables)
            u_y=np.gradient(np.mean(f_nc.variables['rU'], axis=0),y_dat,edge_order=2)
            w_y=np.gradient(np.mean(f_nc.variables['rW'], axis=0),y_dat,edge_order=2)

            alpha_flux_dns = np.arctan2(w_flx,u_flx)
            alpha_flux = alpha_flux_dns -al_geo+al/180*np.pi
            alpha_flux =( (alpha_flux+np.pi) % (2*np.pi) ) - np.pi
            alpha_shear_dns =  ( np.arctan2(w_y,u_y)-np.pi ) 
            alpha_diff= ( alpha_shear_dns - alpha_flux_dns + 3*np.pi ) % (2*np.pi) - np.pi
            if re == 500: 
                ax.plot(ym_dat,alpha_flux,c=ReD_col[re],ls=':',lw=3,label='- (Flux Vector)')
                ax2.plot(ym_dat[1:],alpha_diff[1:],c=ReD_col[re],label='Re={}'.format(re),ls='-')
                #ax2.plot(ym_dat[1:],alpha_flux[1:],     c=ReD_col[re],label='Re={}'.format(re),ls='--')
                #ax2.plot(ym_dat[1:],alpha_shear_dns[1:],     c=ReD_col[re],label='Re={}'.format(re),ls=':')
            else:
                ax.plot(ym_dat,alpha_flux,c=ReD_col[re],ls=':',lw=3)
                ax2.plot(ym_dat[1:],alpha_diff[1:],label='Re_{}'.format(re),c=ReD_col[re],ls='-')
                #ax2.plot(ym_dat[1:],alpha_flux[1:],     c=ReD_col[re],ls='--')
                #ax2.plot(ym_dat[1:],alpha_shear_dns[1:],     c=ReD_col[re],ls=':')

    ax.plot(ym,ym*0,c='black',lw=0.5,ls='-')
    lg=fig.legend(loc='lower right',bbox_to_anchor=(0.63,0.15,0.35,0.6))
    ax.set_xlim(0,1.2)
    ax.set_xlabel(r'$z^-$')
    ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_yticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'])
    ax.set_ylim(-np.pi,np.pi/2)
    ax.set_ylabel(r'Direction $[rad]$')
    fig.savefig('stress_rotation.pdf',format='pdf')

    ax2.plot(ym_dat,0*ym_dat,c='black',ls='-',lw=0.5)
    ax2.set_xlim(0,1.5)
    ax2.set_yticks([-np.pi/2,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3/8*np.pi,np.pi/2]) 
    ax2.set_yticklabels([r'$-\frac{\pi}{2}$',
                         r'$-\frac{\pi}{4}$',
                         r'$-\frac{\pi}{8}$',
                         r'$0$',
                         r'$\frac{\pi}{8}$',
                         r'$\frac{\pi}{4}$',
                         r'$\frac{3\pi}{8}$',
                         r'$\frac{\pi}{2}$'])
    ax2.set_ylim(-np.pi/4,np.pi/4)
    ax2.set_xlabel(r'$z^-$')
    ax2.set_ylabel(r'Misalignment $[$rad$]$')
    lg2=fig2.legend(loc='lower right',bbox_to_anchor=(0.63,0.15,0.35,0.35))
    fig2.savefig('misalignment.pdf',format='pdf')
    plt.close('all')

if plot_profile_comparison:
    # comparison for etling profile
    
    ReD_arr=[500, 1000, 1600,10000,40000] 
  
    

    
    fig=plt.figure(figsize=(5,3))
    figm=plt.figure(figsize=(5,3)) 
    ax=fig.add_axes([0.12,0.15,.85,0.78]) 
    axm=figm.add_axes([0.12,0.15,.85,0.78]) 
    gamma_p=0.1
    
    for re in ReD_arr:
        us,al=sc.ustar_alpha(re)
        reTau = (re*us)**2/2.
        print(re,reTau,us,al/np.pi*180)
        ny,yp,ym=build_grid(reTau,thick=3)
        
        al=al/np.pi*180
        eek_u,eek_v = etling_profile(yp,reTau,al,gamma_p=gamma_p)
        up,vp=sc.profile_plus(yp,re)
        sc_dct=np.arctan2(up,vp)/np.pi*180-90 
        et_dct=np.arctan2(eek_u,eek_v)/np.pi*180 -90
        et_dct[0] = et_dct[1]
        sc_dct[0] = 0. 
        ax.plot(ym[1:],et_dct[1:]/et_dct[1] ,ls='-',label=r'$Re_D={},\alpha_0={}^\circ$'.format(re,int(al*10)/10.),c=ReD_col[re]) 
        ax.plot(ym[1:],(-sc_dct[1:]+sc_dct[-1])/sc_dct[-1] ,ls='--',c=ReD_col[re])
        i30=mp.geti(yp,30)
        print('I30:',i30)
        axm.plot(ym[i30:],np.sqrt(up*up+vp*vp)[i30:]*us,ls='--',c=ReD_col[re])
        axm.plot(ym[:i30],np.sqrt(up*up+vp*vp)[:i30]*us,ls='--',c=ReD_col[re],alpha=0.5)
        m=np.sqrt(eek_u**2+eek_v**2)
        print(ym.shape,yp.shape,m.shape)
        axm.plot(ym[1:],m[1:]/m[-1],ls='-',c=ReD_col[re],label=r'$Re_D={}; u_\star={}G$'.format(re,int(us*1000)/1000.))
    ax.set_xlabel(r'$z^-$')
    ax.set_ylabel(r'relative veer $\alpha/\alpha_0 [1]$')
    ax.set_xscale('log')
    ax.set_xlim(1e-3,2)
    ax.arrow(0.2,0.35,0,0.45,lw=1.0,head_width=0.03,color='gray')
    ax.arrow(0.2,0.8,0,-0.42,lw=1.0,head_width=0.03,color='gray')
    ax.annotate('Error',(0.21,0.5),color='gray',rotation=90 )
    ax.annotate('DNS data',(0.040,0.36),  rotation=-30 )
    ax.annotate('2-layer model',(0.3,0.35),rotation=-70 ) 
    lg=ax.legend()
    lg.get_frame().set_linewidth(0.0)
    fig.suptitle('Matching at $z_{prandtl}=0.10\delta$',fontsize=10)
    fig.savefig('etling_direction_0.png',format='png')

    axm.set_xscale('log')
    axm.set_xlim(1e-3,2)
    axm.set_xlabel(r'$z^-=y/\delta$')
    axm.set_ylabel(r'$|\mathbf{U}(z)|$')
    axm.set_ylim(0,1.1) 
    lgm=axm.legend()
    lgm.get_frame().set_linewidth(0.0)
    figm.suptitle('Matching at $z_{prandtl}=0.10\delta$',fontsize=10)
    figm.savefig('etling_magnitude_0.png',format='png') 
        
    plt.close('all')


    fig= plt.figure(figsize=(5,3))
    figm=plt.figure(figsize=(5,3))
    ax= fig.add_axes([0.12,0.15,.85,0.78])
    axm=figm.add_axes([0.12,0.15,.85,0.78]) 
    gamma_p=0.05
    
    for re in ReD_arr:
        us,al=sc.ustar_alpha(re)
        reTau = (re*us)**2/2.
        print(re,reTau,us,al/np.pi*180)
        ny,yp,ym=build_grid(reTau,thick=3)

        
        al=al/np.pi*180
        eek_u,eek_v = etling_profile(yp,reTau,al,gamma_p=gamma_p)
        up,vp=sc.profile_plus(yp,re)
        sc_dct=np.arctan2(up,vp)/np.pi*180-90 
        et_dct=np.arctan2(eek_u,eek_v)/np.pi*180 -90
        et_dct[0] = et_dct[1]
        sc_dct[0] = 0. 
        ax.plot(ym[1:],et_dct[1:]/et_dct[0] ,ls='-',label=r'$Re_D={},\alpha_0={}^\circ$'.format(re,int(al*10)/10.),c=ReD_col[re]) 
        ax.plot(ym[1:],(-sc_dct[1:]+sc_dct[-1])/sc_dct[-1] ,ls='--',c=ReD_col[re])
        i30=mp.geti(yp,i30)
        print('I30:',i30,ym[i30])
        axm.plot(ym[i30:],np.sqrt(up*up+vp*vp)[i30:]*us  ,ls='--',c=ReD_col[re])
        axm.plot(ym[:i30],np.sqrt(up*up+vp*vp)[:i30]*us  ,ls='--',c=ReD_col[re],alpha=0.5)
        m=np.sqrt(eek_u**2+eek_v**2)
        m=m/m[-1]
        axm.plot(ym[1:],m[1:],ls='-',c=ReD_col[re])
    lg=plt.legend()
    lg.get_frame().set_linewidth(0.0)
    ax.set_xlabel(r'$z^-$')
    ax.set_ylabel(r'relative veer $\alpha/\alpha_0 [1]$')
    ax.set_xscale('log')
    ax.set_xlim(1e-3,2)
    ax.arrow(0.060,0.57,0,0.41,lw=1.0,head_width=0.005,color='gray',overhang=0.3)
    ax.arrow(0.060,0.98,0,-0.41,lw=1.0,head_width=0.005,color='gray',overhang=0.3)
    ax.annotate('Error',(0.062,0.80),color='gray',rotation=90 )
    ax.annotate('DNS data',(0.040,0.36),  rotation=-30 )
    ax.annotate('2-layer model',(0.19,0.35),rotation=-70 )

    axm.set_xscale('log')
    axm.set_xlim(1e-4,2)
    axm.set_xlabel(r'$y^-=z/\delta$')
    axm.set_ylabel(r'$|\mathbf{U}(z)|$')
    axm.set_ylim(0,1.10) 
    lg=plt.legend()  
    fig.suptitle('Matching @ $z_{prandtl}=0.05\delta$',fontsize=9) 
    fig.savefig('etling_direction_1.pdf',format='pdf')

    
    figm.suptitle('Matching @ $z_{prandtl}=0.05\delta$',fontsize=9) 
    figm.savefig('etling_magnitude_1.pdf',format='pdf') 
        
    fig=plt.figure(figsize=(5,3))
    ax=fig.add_axes([0.12,0.15,0.85,0.84]) 

    re=1600
    us1,al1=sc.ustar_alpha(re) 
    reTau1=(re*us1)**2/2.
    ny,yp,ym = build_grid(reTau1,thick=3)
    u1,v1=sc.profile_minus(ym,re)    

    re=np.amin(ReD_arr) 
    us0,al0=sc.ustar_alpha(re) 
    reTau0=(re*us0)**2/2.
    u0,v0=sc.profile_minus(ym,re)

    m0=np.sqrt(u0*u0+v0*v0)
    m1=np.sqrt(u1*u1+v1*v1) 
    ax.fill_between(ym[1:],m0[1:],m1[1:],color='gray',alpha=0.25)

    for re in ReD_arr[0:3]:
        us,al=sc.ustar_alpha(re) 
        reTau=(re*us)**2/2.
        ny,yp,ym = build_grid(reTau,thick=3)
        up,vp=sc.profile_plus(yp,re)
        print('ReD:',re, 'ReTau:',reTau,ny,yp[0],yp[1],yp[-1]/reTau) 

        mag = np.sqrt(up*up + vp*vp)
        ax.plot(yp[1:]/reTau,mag[1:]*us,label=r'$Re_D={}$'.format(re),c=ReD_col[re])


    
        
    ax.set_xscale('log')
    ax.set_xlim(1e-5,2e0)
    lg=ax.legend()
    lg.get_frame().set_linewidth(0.0) 
    ax.set_xlabel(r'$z/\delta$')
    ax.set_ylabel(r'$|\mathbf{U}(z)|/G$')
    ax.arrow(0.05,.5,-0.02,+0.45,color='gray')

    ax.annotate(r'$Re$',(0.048,0.55),color='gray')
    ax.annotate(r'DNS',(0.008,0.35),color='gray',rotation=45)

    plt.savefig('magnitude_0.pdf',format='pdf') 

    
    for re in ReD_arr[3:]:
        us,al=sc.ustar_alpha(re) 
        reTau=(re*us)**2/2.
        ny,yp,ym = build_grid(reTau,thick=3)
        up,vp=sc.profile_plus(yp,re)  
        print('ReD:',re, 'ReTau:',reTau,ny,yp[0],yp[1],yp[-1]/reTau) 

        mag = np.sqrt(up*up + vp*vp)
        ax.plot(yp[1:]/reTau,mag[1:]*us,label=r'$Re_D={}$'.format(re),c=ReD_col[re])

        
    re=np.amax(ReD_arr)
    us0,al0=sc.ustar_alpha(re) 
    reTau0=(re*us0)**2/2.
    ny,yp,ym = build_grid(reTau0,thick=3)
    u0,v0=sc.profile_minus(ym,re)

    re=1e4
    us1,al1=sc.ustar_alpha(re) 
    reTau1=(re*us1)**2/2.
    u1,v1=sc.profile_minus(ym,re)    


    m0=np.sqrt(u0*u0+v0*v0)
    m1=np.sqrt(u1*u1+v1*v1) 
    ax.fill_between(ym[1:],m0[1:],m1[1:],color='red',alpha=0.25)
    

    ax.annotate(r'atmosphere',(2e-5,0.18),color='red',rotation=30) 
    plt.savefig('magnitude_1.pdf',format='pdf')

    ax.annotate(r'$\frac{dU_{log}}{d(log z)} =\frac{u_\star}{\kappa} \propto \frac{1}{log(Re_D)}$',(2e-5,0.65) )
    plt.savefig('magnitude_2.pdf',format='pdf')
    plt.close('all') 

    fig=plt.figure(figsize=(5,3))
    ax=fig.add_axes([0.12,0.16,0.85,0.82])

    for re in ReD_arr:
        us,al=sc.ustar_alpha(re) 
        reTau=(re*us)**2/2.
        ny,yp,ym = build_grid(reTau,thick=3)
        up,vp=sc.profile_plus(yp,re)  
        print('ReD:',re, 'ReTau:',reTau,ny,yp[0],yp[1],yp[-1]/reTau) 
        
        mag = np.sqrt(up*up + vp*vp)
        ax.plot(yp[1:],mag[1:],label=r'$Re_D={}$'.format(re),c=ReD_col[re])
        i_01=mp.geti(yp,0.1*reTau)
        ax.scatter(yp[i_01],mag[i_01],c='black')
    uv_log=sc.profile_log(yp)
    print(uv_log.shape) 
    ax.plot(yp[1:],uv_log[1:],ls='--',lw=1.0,c='gray',label=r'$\kappa^{-1}log(z^+)$') 
    ax.set_xscale('log')
    ax.set_xlabel('$z^+$')
    ax.set_ylabel('$|\mathbf{U}^+(z^+)|$')
    ax.set_xlim(1e0,1e6)
    lg=ax.legend() 
    plt.savefig('magnitude_inner.pdf',format='pdf')
    plt.close('all')

  
 
    fig=plt.figure(figsize=(5,3))
    ax=fig.add_axes([0.12,0.16,0.86,0.82]) 

    for re in ReD_arr:
        us,al=sc.ustar_alpha(re) 
        reTau=(re*us)**2/2.
        ny,yp,ym = build_grid(reTau,thick=3)
        up,vp=sc.profile_plus(yp,re)  
        print('ReD:',re, 'ReTau:',reTau,ny,yp[0],yp[1],yp[-1]/reTau) 
        
        dct= (np.arctan2(up,-vp) - np.pi/2)/D2R
        ax.plot(yp[1:],dct[1:]-dct[0]-90,label=r'$Re_D={}$'.format(re),c=ReD_col[re])
        i_01=mp.geti(yp,0.1*reTau)
        ax.scatter(yp[i_01],dct[i_01],c='black')
    uv_log=sc.profile_log(yp)
    u_ek = 1. - np.cos(yp/100)*np.exp(-yp/100)
    v_ek =      np.sin(yp/100)*np.exp(-yp/100) 
    dct_ek =(np.arctan2(u_ek,v_ek)+np.pi/2)/D2R 
    ax.plot(yp[1:],dct_ek[1:]-135,ls='--',lw=1,c='gray',label='laminar Ekman')
    ax.set_xscale('log')
    ax.set_xlabel(r'$z^+$')
    ax.set_ylabel(r'$\alpha(z^+)$') 
    ax.set_xlim(1e0,1e6)
    ax.arrow(5,15,2e5,-10,lw=1.5,head_width=4,head_length=16000,color='gray',overhang=0)
    ax.annotate(r'$Re$', (5.,11.5),color='gray' )
    lg=ax.legend() 
    plt.savefig('direction_inner.pdf',format='pdf')
    plt.close('all') 
 

    fig=plt.figure(figsize=(5,3))
    ax=fig.add_axes([0.12,0.16,0.86,0.82]) 

    for re in ReD_arr:
        us,al=sc.ustar_alpha(re) 
        reTau=(re*us)**2/2.
        ny,yp,ym = build_grid(reTau,thick=3)
        up,vp=sc.profile_plus(yp,re)  
        print('ReD:',re, 'ReTau:',reTau,ny,yp[0],yp[1],yp[-1]/reTau) 
        
        dct= (np.arctan2(up,-vp) - np.pi/2)/D2R
        ax.plot(ym[1:],dct[1:]-dct[-1],label=r'$Re_D={}$'.format(re),c=ReD_col[re])
        i_01=mp.geti(yp,0.1*reTau)
        ax.scatter(ym[i_01],dct[i_01]-dct[-1],c='black')
    uv_log=sc.profile_log(yp)
    d_ek=2*np.pi
    u_ek = 1. - np.cos(d_ek*ym)*np.exp(-d_ek*ym)
    v_ek =      np.sin(d_ek*ym)*np.exp(-d_ek*ym) 
    dct_ek =(np.arctan2(u_ek,v_ek)+np.pi/2)/D2R 
    ax.plot(ym[1:],dct_ek[1:]-dct_ek[-1],ls='--',lw=1,c='gray',label='laminar Ekman')
    ax.plot([1e-4,2e0],[0,0],ls='-',c='black',lw=0.5) 
    ax.set_xscale('log')
    ax.set_xlabel(r'$z^-=z/\delta$')
    ax.set_ylabel(r'$\alpha(z^-)-\alpha_{GEO}$') 
    ax.set_xlim(1e-4,2e0)
    ax.set_ylim(-20,1)
    ax.arrow(0.05,-18,-.01,12,lw=.60,head_width=.01,head_length=0.4,color='gray')
    ax.annotate(r'$Re$', (0.025,-8),color='gray' )
    lg=ax.legend() 
    plt.savefig('direction_outer.pdf',format='pdf')
    plt.close('all')



if plot_vandriest:
    KAPPA = sc.KAPPA
    ASTAR=26.8
    fig=plt.figure(figsize=(5,3))
    ax=fig.add_axes([0.12,0.16,0.8,0.82])


    zp = np.arange(0,1.0001e3,0.1)
    lam = zp*0+1
    turb = 1. / (KAPPA * zp )
    turb_corrected = 2./(1+np.sqrt(1+ (4*KAPPA**2 * zp**2 )* (1.0-np.exp(-zp/ASTAR) )**2 ))
    ax.plot(zp,turb_corrected,label='van Driest 1956')
    ax.plot(zp,lam,label='laminar')
    ax.plot(zp[15:],turb[15:],label='turbulent')
    leg=plt.legend(loc='best')
    ax.set_xlim(0,1e2)
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$\partial u^+ / \partial y^+$')

    plt.savefig('vanDriest_gradients.png',format='png')
    plt.close('all')

    fig=plt.figure(figsize=(5,3))
    ax=fig.add_axes([0.12,0.16,0.8,0.82])

    C0=5.42

    u =np.zeros(len(zp)) 
    for i in range(len(zp)-1):
        u[i+1] = u[i] + 0.5*(turb_corrected[i] + turb_corrected[i+1]) * (zp[i+1]-zp[i])
    
    lam = zp
    log=1./sc.KAPPA*np.log(zp)+sc.C
    print(log)

    yp=np.arange(2e3)/2.
    u_mod1,w_mod1= sc.profile_plus(yp,1e6)
    u_mod2,w_mod2= sc.profile_plus(yp,1e3)
    ax.plot(zp,u,label='van Driest 1956') 
    ax.plot(zp[:200],lam[:200],'--',lw=1,label='laminar')
    ax.plot(zp[100:],log[100:],lw=1,label='log')
    ax.plot(yp,u_mod1,lw=1,ls='--',label='model (re=1e6)' )
    #ax.plot(yp,u_mod2,lw=1,ls='-',label='model (re=1e3)' )

    ax.set_xscale('log') 
    ax.set_xlim(1,1e3)
    ax.set_ylim(1,20)
    leg=plt.legend(loc='best')
    
    plt.savefig('vanDriest_profile.png',format='png',dpi=600)
    plt.close('all') 
    
    print(zp)
