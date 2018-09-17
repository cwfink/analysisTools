import numpy as np
import pandas as pd
import sys
sys.path.append('/scratch/cwfink/repositories/scdmsPyTools/build/lib/scdmsPyTools/BatTools')
from scdmsPyTools.BatTools.IO import *
import multiprocessing
from itertools import repeat
from qetpy.fitting import ofamp, OFnonlin, MuonTailFit, chi2lowfreq
from qetpy.utils import calc_psd, calc_offset, lowpassfilter, removeoutliers
import matplotlib.pyplot as plt

from scipy.optimize import leastsq, curve_fit
from scipy import stats

import seaborn as sns

from scipy import constants



def hist_cut(df, var, nbins = None, xlims = None, cutold = None,cutnew = None, lgcrawdata = True):
    


    plt.figure(figsize=(9,6))
    plt.title(var)
    plt.xlabel(var)
    plt.ylabel('Count')
    if xlims is not None:
        plt.xlim(xlims)
    if nbins is None:
        nbins = 'sqrt'


    if lgcrawdata:
        hist, bins, _ = plt.hist(df[var], bins = nbins,range = xlims  
                 ,histtype = 'step', label = 'full data', linewidth = 2, color = 'b')
    if cutold is not None:
        oldsum = cutold.sum()
        if cutnew is None:
            cuteff = oldsum/cutold.shape[0]
            cutefftot = cuteff
        if xlims is None:
            maxval = df[var][cutold].max()
            minval = df[var][cutold].min()
            xrange = [minval, maxval]
        else:
            xrange = xlims
        if lgcrawdata:
            nbins = bins
        plt.hist(df[var][cutold], bins = nbins,range = xrange
                 , histtype = 'step', label = 'data passing previous cuts',linewidth = 2, color= 'r')
    if cutnew is not None:
        newsum = cutnew.sum()
        if cutold is not None:
            cutnew = cutnew & cutold
            cuteff = cutnew.sum()/oldsum
            cutefftot = cutnew.sum()/cutnew.shape[0]
        else:
            cuteff = newsum/cutnew.shape[0]
            cutefftot = cuteff
        if xlims is None:
            maxval = df[var][cutnew].max()
            minval = df[var][cutnew].min()
            xrange = [minval, maxval]
        else:
            xrange = xlims   
        if lgcrawdata:
            nbins = bins
        plt.hist(df[var][cutnew], bins = nbins,range = xrange
                 , histtype = 'step', label = 'data passing all cuts',linewidth = 2, color = 'g')
    elif (cutnew is None) & (cutold is None):
        cuteff = 1
        cutefftot = 1


    plt.grid(True)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot([],[], linestyle = ' ', label = f'Efficiency of current cut: {cuteff:.3f}')
    plt.plot([],[], linestyle = ' ', label = f'Efficiency of total cut: {cutefftot:.3f}')
    plt.legend()
    return plt




def plot_cut(df, var ,xlims = None, ylims = None, cutold = None, cutnew = None, lgcrawdata = True
             , ms = 1, a = .3):
    
    if (var[0] == 'index' or len(var) < 2):

        x = df.index
        xlabel = 'index'
    else:
        x = df[var[0]]
        xlabel = var[0]
    if len(var) == 1:
        y = df[var[0]]
        ylabel = var[0]
    else:
        y = df[var[1]]
        ylabel = var[1]   
    
    plt.figure(figsize = (9,6))
    if lgcrawdata:
        plt.scatter(x,y,label = 'full data',c = 'b', s = ms, alpha = a)
    if cutold is not None:
        oldsum = cutold.sum()
        if cutnew is None:
            cuteff = cutold.sum()/cutold.shape[0]
            cutefftot = cuteff
        plt.scatter(x[cutold], y[cutold], label = 'data passing previous cuts',c = 'r', s = ms, alpha = a)
    if cutnew is not None:
        newsum = cutnew.sum()
        if cutold is not None:
            cutnew = cutnew & cutold
            cuteff = cutnew.sum()/oldsum
            cutefftot = cutnew.sum()/cutnew.shape[0]
        else:
            cuteff = newsum/cutnew.shape[0]
            cutefftot = cuteff
            
        plt.scatter(x[cutnew], y[cutnew], label = 'data passing all cuts', c='g',s = ms, alpha = a)
    
    elif (cutnew is None) & (cutold is None):
        cuteff = 1
        cutefftot = 1
        
    if xlims is None:
        if lgcrawdata:
            xrange = x.max()-x.min()
            plt.xlim([x.min()-0.05*xrange,x.max()+0.05*xrange])
        elif cutnew is not None:
            xrange = x[cutnew].max()-x[cutnew].min()
            plt.xlim([x[cutnew].min()-0.05*xrange,x[cutnew].max()+0.05*xrange])
        
         
    else:
        plt.xlim(xlims)
    if ylims is None:
        if lgcrawdata:
            yrange = y.max()-y.min()
            plt.ylim([y.min()-0.05*yrange,y.max()+0.05*yrange])
        elif cutnew is not None:
            yrange = y[cutnew].max()-y[cutnew].min()
            plt.ylim([y[cutnew].min()-0.05*yrange,y[cutnew].max()+0.05*yrange])
        
    else:
        plt.ylim(ylims)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid(True)
    plt.plot([],[], linestyle = ' ', label = f'Efficiency of current cut: {cuteff:.3f}')
    plt.plot([],[], linestyle = ' ', label = f'Efficiency of total cut: {cutefftot:.3f}')
    plt.legend(markerscale = 6, framealpha = .9)
    return plt



def scale_energy_spec(DF, cut, var, p0, title, xlabel):
    x, hist,bins = get_hist_data(DF,cut, var)
    popt = curve_fit(double_gauss,x, hist, p0 )[0]
    x_est = np.linspace(x.min(), x.max(), 1000)
    y_est = double_gauss(x_est,*popt)
    peak59 = min([popt[2],popt[3]])
    peak64 = max([popt[2],popt[3]])
    
    plt.figure(figsize=(9,6))
    plt.hist(x, bins = bins, weights = hist, histtype = 'step', linewidth = 1, label ='data')
    plt.plot(x_est, y_est, 'g', label='Convoluted Gaussian Fit', alpha = .5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.text(peak59*1.02,max(hist)/2,f"μ: {peak59:.2e} \n σ: {popt[4]:.2e}",size = 12)
    plt.text(peak64,max(hist)/3,f"μ: {peak64:.2e} \n σ: {popt[5]:.2e}",size = 12)
    plt.legend()
    plt.grid(True, linestyle = 'dashed')
    plt.xlabel(xlabel)
    plt.ylabel('Counts')
    plt.title(title)
    #plt.savefig(saveFigPath+'OF_amp_fit_fake.png')
    plt.show()
    print(popt)

    energy_per_amp59 = (5.9e3)/peak59
    energy_per_amp649= (6.49e3)/peak64

    energy_quad = scale_energy_quad(peak59,peak64, DF[var])

    DF[f'{var}_quad_energy'] = energy_quad
    DF[f'{var}_linear_energy'] = DF[var]*energy_per_amp59
    DF[f'energy_per_{var}'] = energy_per_amp59


    plt.figure(figsize=(9,6))
    plt.hist(energy_quad[cut],histtype='step', bins = 'sqrt')
    plt.axvline(5.9e3)
    plt.axvline(6.49e3)
    plt.text(5700,max(hist)/2, '5.9 keV line',size = 12)
    plt.text(6300,max(hist)/3, '6.49 keV line',size = 12)
    plt.xlabel('Energy [eV]')
    plt.ylabel('Counts')
    plt.title(f'{title} (scaled to energy quadratic)')  

    plt.show()

    plt.figure(figsize=(9,6))
    plt.hist(x*energy_per_amp59, bins = bins*energy_per_amp59, 
             weights = hist, histtype = 'step', linewidth = 1, label ='data')
    plt.axvline(5.9e3)
    plt.axvline(6.49e3)
    plt.text(5550,max(hist)/2, '5.9 keV line',size = 12)
    plt.text(6200,max(hist)/3, '6.49 keV line',size = 12)
    plt.xlabel('Energy [eV]')
    plt.ylabel('Counts')
    plt.title(f'{title} (scaled to energy)')  
    plt.grid(True, linestyle = 'dashed')
    #plt.savefig(saveFigPath+'OF_amp_fit_energy_fake.png')
    plt.show()

    print(peak59)
    print(peak64)
    
    
    return energy_per_amp59


def scale_energy_quad(peak59,peak64, data):
    x = [0, peak59, peak64]
    y = [0, 5.9e3, 6.49e3]
    popt = np.polyfit(x,y,2)
    
    return popt[0]*data**2+popt[1]*data+popt[2]


def amp_to_energy(DF,cut, clinearx, clineary, clowenergy, yvar = 'int_bsSub_short_linear_energy', 
                  xvar = 'ofAmps_tdelay', title = 'PT On, Fast Template', yerr= 65, order = 5):
    sns.set_context('notebook')
    x_full = DF[cut][xvar].values

    y_full = DF[cut][yvar].values

    cy = y_full < clineary
    cx = x_full < clinearx
    clow = y_full < clowenergy
    #x_full = x_full*energy_per_int59
    #x_full = RQ_df[cgoodevents]['int_energy'].values
    x = x_full[cy & cx]
    y = y_full[cy & cx]
    
    #print(x_full.max())
    #print(y_full.max())
    #print('-------')
    #print(x.max(), np.argmax(x))
    #print(y.max(), np.argmax(y))
    #print(y[np.argmax(x)])
    
    x_fit = np.linspace(0,x.max(),100)
    #y_plot = np.linspace(0,y_full.max(),100)
    def line(slope, x):
        return slope*x
    
    #def poly(x, A,B,C,D,E):
    #    return A*x**5 + B*x**4 + C*x**3 + D*x**2 + E*x 
    def poly(x, *params):
        rf = np.poly1d(params)
        return rf(x)*x
    def err_func(x):
        a, n, c = (0.40704751, 0.33997668, 8.02056977)

        c = .5
        return a*x**n + c

    #p0 = np.polyfit(y, x, 5)
    #popt_poly = curve_fit(poly, xdata = y, ydata = x, p0 = p0[:5], maxfev=100000)
    p0 = np.polyfit(x, y, order)
    popt_poly = curve_fit(poly, xdata = x, ydata = y, sigma = yerr*err_func(y), p0 = p0[:order], maxfev=100000)
    z = popt_poly[0]
    #z = np.polyfit(y, x, 5)
    
    p = np.poly1d(np.concatenate((z,[0])))
    
    #print(z)
    #print(p0)
    
    chi = (((y_full[clow] - p(x_full[clow]))/yerr)**2).sum()/(len(y_full[clow])-order)
    
    
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #intercept = 0
    #popt = curve_fit(line, x, y)
    #slope = popt[0][0]
    #y_fit = slope*x_fit
    
    
    
    
    
    plt.figure(figsize=(9,6))
    plt.plot(x_full, y_full, marker = '.', linestyle = ' ', label = 'Data passing cuts', ms = 3, alpha = .5)
    plt.errorbar(x,y, marker = '.', linestyle = ' ', yerr = yerr*err_func(y), label = 'Data used for Fit',
                 elinewidth=0.3, alpha =1, ms = 5,zorder = 50)
                 #elinewidth=0.3,capsize=2, capthick=0.8, alpha =.5, ms = 3)
    #plt.plot(x_fit, y_fit, label = 'Linear Fit')
    plt.ylim(0,9000)
    plt.xlim(0,x_full.max()*1.05)
    plt.grid(True)
    #plt.text(500,y_full.max()/1.5, f"y = {slope:.3e} [A/eV]x",size =12)
    plt.plot(x_fit,p(x_fit), zorder = 100)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.legend()
    plt.ylabel('Integrated Energy [eV]')
    plt.xlabel('OF Amplitude [A]')
    plt.title('OF Amplitude vs Integral Energy Calibration:  \n' + title)
    #plt.tight_layout()
    #plt.xlim(0, 1.5e-6)
    #plt.ylim(0,2e3)
    #plt.savefig('amp_vs_int_cal_fit_full_zommed.png')
    
    
    
    
    
    #print(p(0))
    #amp_per_E_fromInt = slope
    DF['ofAmps0_energy_5th'] = p(DF['ofAmps'])
    DF['ofAmpst_energy_5th'] = p(DF['ofAmps_tdelay']) 
    DF['ofAmpst_energy_5th_nocon'] = p(DF['ofAmps_tdelay_nocon']) 
    #DF['amp_per_E_fromInt'] = amp_per_E_fromInt
    #return slope
    return chi


def amp_to_energy_v2(DF, clinearx, clineary, clowenergy, yarr = 'int_bsSub_short_linear_energy', 
                  xarr = 'ofAmps_tdelay', title = 'PT On, Fast Template', yerr= 65, order = 5):
    sns.set_context('notebook')
    x_full = xarr

    y_full = yarr
    
    def poly(x, *params):
        rf = np.poly1d(params)
        return rf(x)*x
    def err_func(x):
        a, n, c = (0.40704751, 0.33997668, 8.02056977)

        c = 5
        return a*x**n + c

    cy = y_full < clineary
    cx = x_full < clinearx
    clow = y_full < clowenergy
    #x_full = x_full*energy_per_int59
    #x_full = RQ_df[cgoodevents]['int_energy'].values
    x = x_full[cy & cx]
    y = y_full[cy & cx]
    
    #print(x_full.max())
    #print(y_full.max())
    #print('-------')
    #print(x.max(), np.argmax(x))
    #print(y.max(), np.argmax(y))
    #print(y[np.argmax(x)])
    
    x_fit = np.linspace(0,x.max(),100)
    #y_plot = np.linspace(0,y_full.max(),100)
    def line(slope, x):
        return slope*x
    
    #def poly(x, A,B,C,D,E):
    #    return A*x**5 + B*x**4 + C*x**3 + D*x**2 + E*x 
    def poly(x, *params):
        rf = np.poly1d(params)
        return rf(x)*x
    
    #p0 = np.polyfit(y, x, 5)
    #popt_poly = curve_fit(poly, xdata = y, ydata = x, p0 = p0[:5], maxfev=100000)
    p0 = np.polyfit(x, y, order)
    popt_poly = curve_fit(poly, xdata = x, ydata = y, sigma = yerr*err_func(y), p0 = p0[:order], maxfev=100000)
    z = popt_poly[0]
    #z = np.polyfit(y, x, 5)
    
    p = np.poly1d(np.concatenate((z,[0])))
    
    #print(z)
    #print(p0)
    
    chi = (((y_full[clow] - p(x_full[clow]))/yerr)**2).sum()/(len(y_full[clow])-order)
    
    
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #intercept = 0
    #popt = curve_fit(line, x, y)
    #slope = popt[0][0]
    #y_fit = slope*x_fit
    plt.figure(figsize=(9,6))
    plt.plot(x_full, y_full, marker = '.', linestyle = ' ', label = 'Data passing cuts', ms = 3, alpha = .5)
    plt.errorbar(x,y, marker = '.', linestyle = ' ', yerr = yerr, label = 'Data used for Fit',
                 elinewidth=0.3, alpha =1, ms = 5,zorder = 50)
                 #elinewidth=0.3,capsize=2, capthick=0.8, alpha =.5, ms = 3)
    #plt.plot(x_fit, y_fit, label = 'Linear Fit')
    plt.ylim(0,9000)
    plt.xlim(0,x_full.max()*1.05)
    plt.grid(True)
    #plt.text(500,y_full.max()/1.5, f"y = {slope:.3e} [A/eV]x",size =12)
    plt.plot(x_fit,p(x_fit), zorder = 100)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.legend()
    plt.ylabel('Integrated Energy [eV]')
    plt.xlabel('OF Amplitude [A]')
    plt.title('OF Amplitude vs Integral Energy Calibration:  \n' + title)
    #plt.tight_layout()
    #plt.xlim(0, 1.5e-6)
    #plt.ylim(0,2e3)
    #plt.savefig('amp_vs_int_cal_fit_full_zommed.png')
    
    
    
    
    
    #print(p(0))
    #amp_per_E_fromInt = slope
    DF['ofAmps0_energy_5th'] = p(DF['ofAmps'])
    DF['ofAmpst_energy_5th'] = p(DF['ofAmps_tdelay']) 
    DF['ofAmpst_energy_5th_nocon'] = p(DF['ofAmps_tdelay_nocon']) 
    #DF['amp_per_E_fromInt'] = amp_per_E_fromInt
    #return slope
    return chi



