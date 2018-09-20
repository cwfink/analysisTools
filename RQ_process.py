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

def calcbaselinecut(arr, r0, i0, rload, dr = 0.1e-3, cut = None):
    """
    Function to automatically generate the pre-pulse baseline cut. 
    The value where the cut is placed is set by dr, which is the user
    specified change in resistance from R0
    
    Parameters
    ----------
        arr: ndarray
            Array of values to generate cut with
        r0: float
            Operating resistance of TES
        i0: float
            Quiescent operating current of TES
        rload: float
            The load resistance of the TES circuit, (Rp+Rsh)
        dr: float, optional
            The change in operating resistance where the
            cut should be placed
        cut: ndarray, optional
            Initial cut mask to use in the calculation of the pre-pulse
            baseline cut
            
    Returns:
    --------
        cbase_pre: ndarray
            Array of type bool, corresponding to values which pass the 
            pre-pulse baseline cut
            
    """
    
    if cut is None:
        cut = np.ones_like(arr, dtype = bool)
    
    base_inds = removeoutliers(arr[cut])
    meanval = np.mean(arr[cut][base_inds])
    
    di = -(dr/(r0+dr+rload)*i0)
    
    cbase_pre = (arr < (meanval + di))
    
    return cbase_pre



def getrandevents(basepath, evtnums, seriesnums, cut=None, channels=["PDS1"], convtoamps=1, fs=625e3, 
                  lgcplot=False, ntraces=1, nplot=20, seed=None):
    """
    Function for loading (and plotting) random events from a datasets. Has functionality to pull 
    randomly from a specified cut. For use with scdmsPyTools.BatTools.IO.getRawEvents
    
    Parameters
    ----------
        basepath : str
            The base path to the directory that contains the folders that the event dumps 
            are in. The folders in this directory should be the series numbers.
        evtnums : array_like
            An array of all event numbers for the events in all datasets.
        seriesnums : array_like
            An array of the corresponding series numbers for each event number in evtnums.
        cut : array_like, optional
            A boolean array of the cut that should be applied to the data. If left as None,
            then no cut is applied.
        channels : list, optional
            A list of strings that contains all of the channels that should be loaded.
        convtoamps : float, optional
            The factor that the traces should be multiplied by to convert ADC bins to Amperes.
        fs : float, optional
            The sample rate in Hz of the data.
        ntraces : int, optional
            The number of traces to randomly load from the data (with the cut, if specified)
        lgcplot : bool, optional
            Logical flag on whether or not to plot the pulled traces.
        nplot : int, optional
            If lgcplot is True, the number of traces to plot.
        seed : int, optional
            A value to pass to np.random.seed if the user wishes to use the same random seed
            each time getrandevents is called.
        
    Returns
    -------
        t : ndarray
            The time values for plotting the events.
        x : ndarray
            Array containing all of the events that were pulled.
        c_out : ndarray
            Boolean array that contains the cut on the loaded data.
    
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if type(evtnums) is not pd.core.series.Series:
        evtnums = pd.Series(data=evtnums)
    if type(seriesnums) is not pd.core.series.Series:
        seriesnums = pd.Series(data=seriesnums)
        
    if cut is None:
        cut = np.ones(len(evtnums), dtype=bool)
        
    inds = np.random.choice(np.flatnonzero(cut), size=ntraces, replace=False)
        
    crand = np.zeros(len(evtnums), dtype=bool)
    crand[inds] = True
    
    arrs = list()
    for snum in seriesnums[crand].unique():
        cseries = crand & (seriesnums == snum)
        arr = getRawEvents(f"{basepath}{snum}/", "", channelList=channels, outputFormat=3, 
                           eventNumbers=evtnums[cseries].tolist())
        arrs.append(arr)
        
    chans = list()
    for chan in channels:
        chans.append(arr["Z1"]["pChan"].index(chan))
    chans = sorted(chans)

    x = np.vstack([a["Z1"]["p"][:, chans] for a in arrs]).astype(float)
    t = np.arange(x.shape[-1])/fs
    
    x*=convtoamps
    
    if lgcplot:
        
        if nplot>ntraces:
            nplot = ntraces
    
        colors = plt.cm.viridis(np.linspace(0, 1, num=nplot), alpha=0.5)

        for ii in range(nplot):

            fig, ax = plt.subplots(figsize=(10, 6))
            for chan in chans:
                ax.plot(t * 1e6, x[ii, chan] * 1e6, color=colors[ii], label="Channel {arr['Z1']['pChan'][chan]}")
            ax.grid()
            ax.set_ylabel("Current [μA]")
            ax.set_xlabel("Time [μs]")
            ax.set_title(f"Pulses, Evt Num {evtnums[crand].iloc[ii]}, Series Num {seriesnums[crand].iloc[ii]}");
    
    
    return t, x, crand

def get_trace_gain(path, chan, det, gainfactors = {'rfb': 5000, 'loopgain' : 2.4, 'adcpervolt' : 2**(16)/2}):
    """
    
    calculates the conversion from ADC bins to TES current. To convert traces from ADC bins to current: traces[ADC]*convtoamps
    Parameters
    ----------
    path: str or list of str
        absolute path to the dump to open, or list of paths
    chan: str
        channel name, ie 'PDS1'
    det: str
        detector name, ie 'Z1'
    gainfactors: dict, optional
        dictionary containing phonon amp parameters, keys: 
            'rfb': feedback resistor
            'loopgain':gain of loop of the feeback amp
            'adcpervolt': the bitdepth divided by the voltage range of the adc
    Returns
    -------
    convtoamps: float
        conversion factor from adc bins to TES current in amps
    drivergain: float
        gain setting of the driver amplifier
    qetbias: float
        
    """
    
    series = path.split('/')[-1]
    print(path)
    print(series)
    settings = getDetectorSettings(path, series)
    qetbias = settings[det][chan]['qetBias']
    drivergain = settings[det][chan]['driverGain']*2
    convtoamps = 1/(gainfactors['rfb'] * gainfactors['loopgain'] * drivergain * gainfactors['adcpervolt'])
    
    return convtoamps, drivergain, qetbias

def get_traces_per_dump(path, chan, det, convtoamps = 1):
    """
    Function to return raw traces and event information for a single channel
    
    Parameters
    ----------
    path: str or list of str
        absolute path to the dump to open, or list of paths
    chan: str
        channel name, ie 'PDS1'
    det: str
        detector name, ie 'Z1'
    convtoamps: float, optional
        gain factor to convert ADC bins to TES current. should be units of [A]/[ADCbins]
    
    Returns
    -------
    traces: ndarray
        array of traces. If convtoamps is provided, then traces
        are scaled to be in unist of amps. Else they are in units of adc bins
    eventnumber: array
        array of the event numbers (int)
    eventtime: array
        array of event times (int)
    triggertype: array
        array of trigger types (int)
    triggeramp: array
        array of optimum filter trigger amplitudes (int)
    
        
    """
    if not isinstance(path, list):
        path = list(path)
    
    events = getRawEvents(filepath='',files_series = path, channelList=[chan],outputFormat=3)
    eventnumber = []
    eventtime = []
    triggertype = []
    triggeramp = []
    for ii in range(len(events['event'])):
        eventnumber.append(events['event'][ii]['EventNumber'])
        eventtime.append(events['event'][ii]['EventTime'])
        triggertype.append(events['event'][ii]['TriggerType'])
        triggeramp.append(events['trigger'][ii]['TriggerAmplitude1'])
    traces = events[det]['p'][:,0,:]*convtoamps
    
    return traces, np.array(eventnumber), np.array(eventtime), np.array(triggertype), np.array(triggeramp)


    

def process_RQ(file, params):
    chan, det, convtoamps, template, psd, fs ,time, ioffset, indbasepre, indbasepost, qetbias, rload = params
    traces , eventNumber, eventTime,triggertype,triggeramp = get_traces_per_dump([file], chan = chan, det = det, convtoamps = convtoamps)
    columns = ['ofAmps_tdelay','tdelay','chi2_tdelay','ofAmps','chi2','baseline_pre', 'baseline_post',
               'slope','int_bsSub','eventNumber','eventTime', 'triggerType','triggeramp','energy_integral1', 
              'ofAmps_tdelay_nocon','tdelay_nocon','chi2_tdelay_nocon','chi2_1000','chi2_5000','chi2_10000','chi2_50000',
              'seriesNumber', 'chi2_timedomain', 'ofAmps_tdelay_outside','tdelay_outside','chi2_tdelay_outside']
    
              #,'A_nonlin' , 'taurise', 'taufall', 't0nonlin', 'chi2nonlin','A_nonlin_err' , 'taurise_err', 'taufall_err', 
              # 't0nonlin_err', 'Amuon' , 'taumuon', 'Amuon_err' , 'taumuon_err']
    
    
    temp_data = {}
    for item in columns:
        temp_data[item] = []
    dump = file.split('/')[-1].split('_')[-1].split('.')[0]
    seriesnum = file.split('/')[-2]
    print(f"On Series: {seriesnum},  dump: {dump}")
    

    for ii, trace in enumerate(traces):
        baseline = np.mean(np.hstack((trace[:indbasepre], trace[indbasepost:])))
        trace_bsSub = trace - baseline
        #traceFilt = lowpassfilter(trace, cut_off_freq=50e3)
        #traceFilt_bsSub = lowpassfilter(trace_bsSub, cut_off_freq=50e3)
        powertrace = convert_to_power(trace, I_offset=ioffset,I_bias = qetbias,Rsh=5e-3, Rl=rload)
        energy_integral1 = integral_Energy_caleb(powertrace, time, indbasepre, indbasepost)
        
        

        #minval = np.min(trace)
        #maxval = np.max(trace)
        #minFilt = np.min(traceFilt)
        #maxFilt = np.max(traceFilt)

        amp_td, t0_td, chi2_td = ofamp(trace, template, psd, fs, lgcsigma = False, nconstrain = 80)
        amp_td_nocon, t0_td_nocon, chi2_td_nocon = ofamp(trace,template, psd,fs, lgcsigma = False)
        amp, _, chi2 = ofamp(trace,template, psd,fs, withdelay=False, lgcsigma = False)
        amp_td_out, t0_td_out, chi2_td_out = ofamp(trace, template, psd, fs, lgcsigma = False, nconstrain = 80,\
                                                   lgcoutsidewindow = True)
        
        
        chi2_1000 = chi2lowfreq(trace, template, amp_td, t0_td, psd, fs, fcutoff=1000)
        chi2_5000 = chi2lowfreq(trace, template, amp_td, t0_td, psd, fs, fcutoff=5000)
        chi2_10000 = chi2lowfreq(trace, template, amp_td, t0_td, psd, fs, fcutoff=10000)
        chi2_50000 = chi2lowfreq(trace, template, amp_td, t0_td, psd, fs, fcutoff=50000)
        

        
        chi2_timedomain = td_chi2(trace, template, amp_td, t0_td, fs, baseline=np.mean(trace[:indbasepre]))
        
        #nonlinof = OFnonlin(psd = psd, fs = fs, template=template)
        #fitparams,errors_nonlin,_,chi2nonlin = nonlinof.fit_falltimes(trace_bsSub, lgcdouble = True,
        #                                              lgcfullrtn = True, lgcplot=  False)
        #Anonlin, tau_r, tau_f, t0nonlin = fitparams
        #Anonlin_err, tau_r_err, tau_f_err, t0nonlin_err = errors_nonlin
        
        
        #muontail = MuonTailFit(psd = psd, fs = 625e3)
        #muonparams, muonerrors,_,chi2muon = muontail.fitmuontail(trace, lgcfullrtn = True)
        #Amuon, taumuon = muonparams
        #Amuon_err, taumuon_err = muonerrors
        
        

        temp_data['baseline_pre'].append(np.mean(trace[:indbasepre]))
        temp_data['baseline_post'].append(np.mean(trace[indbasepost:]))
        
        temp_data['slope'].append(np.mean(trace[:indbasepre]) - np.mean(trace[indbasepost:]))

        temp_data['eventNumber'].append(eventNumber[ii])
        temp_data['eventTime'].append(eventTime[ii])
        temp_data['seriesNumber'].append(seriesnum)


        temp_data['int_bsSub'].append(np.trapz(trace_bsSub, time))
        
        temp_data['energy_integral1'].append(energy_integral1)

        temp_data['ofAmps'].append(amp)
        temp_data['chi2'].append(chi2)
        temp_data['ofAmps_tdelay'].append(amp_td)
        temp_data['tdelay'].append(t0_td)
        temp_data['chi2_tdelay'].append(chi2_td)
        
        temp_data['chi2_timedomain'].append(chi2_timedomain)
        
#         temp_data['A_nonlin'].append(Anonlin)
#         temp_data['taurise'].append(tau_r)
#         temp_data['taufall'].append(tau_f)
#         temp_data['t0nonlin'].append(t0nonlin)
#         temp_data['chi2nonlin'].append(chi2nonlin)
#         temp_data['A_nonlin_err'].append(Anonlin_err)
#         temp_data['taurise_err'].append(tau_r_err)
#         temp_data['taufall_err'].append(tau_f_err)
#         temp_data['t0nonlin_err'].append(t0nonlin_err)
        
        
        temp_data['chi2_1000'].append(chi2_1000)
        temp_data['chi2_5000'].append(chi2_5000)
        temp_data['chi2_10000'].append(chi2_10000)
        temp_data['chi2_50000'].append(chi2_50000)
        
        temp_data['ofAmps_tdelay_nocon'].append(amp_td_nocon)
        temp_data['tdelay_nocon'].append(t0_td_nocon)
        temp_data['chi2_tdelay_nocon'].append(chi2_td_nocon)
        temp_data['ofAmps_tdelay_outside'].append(amp_td_out)
        temp_data['tdelay_outside'].append(t0_td_out)
        temp_data['chi2_tdelay_outside'].append(chi2_td_out)
        
        temp_data['triggerType'].append(triggertype[ii])
        temp_data['triggeramp'].append(triggeramp[ii])
        
#         temp_data['Amuon'] = Amuon
#         temp_data['taumuon'] = taumuon
#         temp_data['Amuon_err'] = Amuon_err
#         temp_data['taumuon_err'] = taumuon_err

    df_temp = pd.DataFrame.from_dict(temp_data)#.set_index('eventNumber')
    return df_temp


def multiprocess_RQ(filelist, chan, det, convtoamps, template, psd, fs ,time, ioffset, indbasepre, indbasepost, qetbias, rload ):
    
    path = filelist[0]
    pathgain = path.split('.')[0][:-19]
    convtoamps,_,_ = get_trace_gain(pathgain, det=det, chan=chan)
    nprocess = int(2)
    pool = multiprocessing.Pool(processes = nprocess)
    results = pool.starmap(process_RQ, zip(filelist, repeat([chan, det, convtoamps, template, psd, \
                                        fs ,time, ioffset, indbasepre, indbasepost, qetbias, rload ])))
    pool.close()
    pool.join()
    RQ_df = pd.concat([df for df in results], ignore_index = True)  
    return RQ_df
    #RQ_df.to_pickle('RQ_df_PTon_init_Template2.pkl')  
    
def histRQ(arr, nbins = None, xlims = None, cutold = None,cutnew = None, lgcrawdata = True, 
    lgceff = True, labeldict = None):
    """
    Function to plot histogram of RQ data. The bins are set such that all bins have the same size
    as the raw data
    
    Parameters
    ----------
        arr: array
            Array of values to be binned and plotted
        nbins:  integer or sequence or 'auto', optional
            This is the same as plt.hist() bins parameter. defaults to 'sqrt'
        xlims: list of floats, optional
            this is passed to plt.hist() range parameter
        cutold: array of booleans, optional
            mask of good values to be plotted
        cutnew: array of booleans, optional
            mask of good values to be plotted. This mask is added to cutold if cutold is not None. 
        lgcrawdata: boolean, optional
            if True, the raw data is plotted
        lgceff: boolean, optional
            if True, the cuff efficiencies are printed in the legend. The total eff will be the sum of all the 
            cuts divided by the length of the data. the current cut eff will be the sum of the current cut 
            divided by the sum of all the previous cuts, if any
        labeldict: dictionary, optional
            dictionary to overwrite the labels of the plot. defaults are : 
                labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count', 'cutnew' : 'current' 
                , 'cutold' : 'previous'}
            Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to histRQ()
    
    Returns
    -------
        fig, ax: matplotlib figure and ax object
        
    """
    
    labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count', 'cutnew' : 'current', 'cutold' : 'previous'}
    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]
    
    fig, ax = plt.subplots(figsize = (9,6))
    ax.set_title(labels['title'])
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])
    if nbins is None:
        nbins = 'sqrt'

    if lgcrawdata:
        if xlims is None:
            hist, bins, _ = ax.hist(arr, bins = nbins  
                     ,histtype = 'step', label = 'full data', linewidth = 2, color = 'b')
        else:
            hist, bins, _ = ax.hist(arr, bins = nbins , range = xlims 
                     ,histtype = 'step', label = 'full data', linewidth = 2, color = 'b')            
    if cutold is not None:
        oldsum = cutold.sum()
        if cutnew is None:
            cuteff = oldsum/cutold.shape[0]
            cutefftot = cuteff
        if lgcrawdata:
            nbins = bins
        label = f"Data passing {labels['cutold']} cut"
        if xlims is not None:
            ax.hist(arr[cutold], bins = nbins,range = xlims
                     , histtype = 'step', label = label,linewidth = 2, color= 'r')
        else:
            ax.hist(arr[cutold], bins = nbins
                     , histtype = 'step', label = label,linewidth = 2, color= 'r')
    if cutnew is not None:
        newsum = cutnew.sum()
        if cutold is not None:
            cutnew = cutnew & cutold
            cuteff = cutnew.sum()/oldsum
            cutefftot = cutnew.sum()/cutnew.shape[0]
        else:
            cuteff = newsum/cutnew.shape[0]
            cutefftot = cuteff
        if lgcrawdata:
            nbins = bins
        if lgceff:
            label = f"Data passing {labels['cutnew']} cut, eff :  {cuteff:.3f}"
        else:
            label = f"Data passing {labels['cutnew']} cut "
        if xlims is not None:
            ax.hist(arr[cutnew], bins = nbins,range = xlims
            , histtype = 'step', linewidth = 2, color = 'g', label = label)
        else:
            ax.hist(arr[cutnew], bins = nbins
            , histtype = 'step', linewidth = 2, color = 'g', label = label)
    elif (cutnew is None) & (cutold is None):
        cuteff = 1
        cutefftot = 1
    plt.grid(True)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #plt.plot([],[], linestyle = ' ', label = f'Efficiency of current cut: {cuteff:.3f}')
    if lgceff:
        ax.plot([],[], linestyle = ' ', label = f'Efficiency of total cut: {cutefftot:.3f}')
    plt.legend()
    return fig, ax
    



def plotRQ(xvals, yvals, xlims = None, ylims = None, cutold = None, cutnew = None, 
           lgcrawdata = True, lgceff=True, labeldict = None, ms = 1, a = .3):
    """
    Function to plot RQ data as a scatter plot.
    
    Parameters
    ----------
        xvals: array
            Array of x values to be plotted
        yvals: array
            Array of y values to be plotted
        xlims: list of floats, optional
            This is passed to the plot as the x limits
        ylims: list of floats, optional
            This is passed to the plot as the y limits
        cutold: array of booleans, optional
            mask of values to be plotted
        cutnew: array of booleans, optional
            mask of values to be plotted. This mask is added to cutold if cutold is not None. 
        lgcrawdata: boolean, optional
            if True, the raw data is plotted
        lgceff: boolean, optional
            if True, the cuff efficiencies are printed in the legend. The total eff will be the sum of all the 
            cuts divided by the length of the data. the current cut eff will be the sum of the current cut 
            divided by the sum of all the previous cuts, if any
        labeldict: dictionary, optional
            dictionary to overwrite the labels of the plot. defaults are : 
                labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count', 'cutnew' : 'current' 
                , 'cutold' : 'previous'}
            Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to histRQ()
        ms: float, optional
            The size of each marker in the scatter plot. Default is 1
        a: float, optional
            The opacity of the markers in the scatter plot, i.e. alpha. Default is 0.3
    
    Returns
    -------
        fig: Object
            Matplotlib figure object
        ax: Object
            Matplotlib axes object
        
    """

    labels = {'title' : 'Plot of x vs. y', 'xlabel' : 'x', 'ylabel' : 'y', 'cutnew' : 'current', 'cutold' : 'previous'}
    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]
    
    fig, ax = plt.subplots(figsize = (9,6))
    
    ax.set_title(labels['title'])
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])

    
    if xlims is not None:
        xlimitcut = (xvals>xlims[0]) & (xvals<xlims[1])
    else:
        xlimitcut = np.ones(len(xvals), dtype=bool)
    if ylims is not None:
        ylimitcut = (yvals>ylims[0]) & (yvals<ylims[1])
    else:
        ylimitcut = np.ones(len(yvals), dtype=bool)

    limitcut = xlimitcut & ylimitcut
    
    if lgcrawdata and (cutnew is not None and cutold is not None):
        ax.scatter(xvals[limitcut & ~cutold & ~cutnew], yvals[limitcut & ~cutold & ~cutnew], 
                   label = 'Full Data', c = 'b', s = ms, alpha = a)
    elif lgcrawdata and cutnew is not None: 
        ax.scatter(xvals[limitcut & ~cutnew], yvals[limitcut & ~cutnew], 
                   label = 'Full Data', c = 'b', s = ms, alpha = a)
    elif lgcrawdata and cutold is not None: 
        ax.scatter(xvals[limitcut & ~cutold], yvals[limitcut & ~cutold], 
                   label = 'Full Data', c = 'b', s = ms, alpha = a)
    elif lgcrawdata:
        ax.scatter(xvals[limitcut], yvals[limitcut], 
                   label = 'Full Data', c = 'b', s = ms, alpha = a)
        
    if cutold is not None:
        oldsum = cutold.sum()
        if cutnew is None:
            cuteff = cutold.sum()/cutold.shape[0]
            cutefftot = cuteff
            
        label = f"Data passing {labels['cutold']} cut"
        if cutnew is None:
            ax.scatter(xvals[cutold & limitcut], yvals[cutold & limitcut], 
                       label=label, c='r', s=ms, alpha=a)
        else: 
            ax.scatter(xvals[cutold & limitcut & ~cutnew], yvals[cutold & limitcut & ~cutnew], 
                       label=label, c='r', s=ms, alpha=a)
        
    if cutnew is not None:
        newsum = cutnew.sum()
        if cutold is not None:
            cutnew = cutnew & cutold
            cuteff = cutnew.sum()/oldsum
            cutefftot = cutnew.sum()/cutnew.shape[0]
        else:
            cuteff = newsum/cutnew.shape[0]
            cutefftot = cuteff
            
        if lgceff:
            label = f"Data passing {labels['cutnew']} cut, eff : {cuteff:.3f}"
        else:
            label = f"Data passing {labels['cutnew']} cut"
            
        ax.scatter(xvals[cutnew & limitcut], yvals[cutnew & limitcut], 
                   label=label, c='g', s=ms, alpha=a)
    
    elif (cutnew is None) & (cutold is None):
        cuteff = 1
        cutefftot = 1
        
    if xlims is None:
        if lgcrawdata:
            xrange = xvals.max()-xvals.min()
            ax.set_xlim([xvals.min()-0.05*xrange, xvals.max()+0.05*xrange])
        elif cutold is not None:
            xrange = xvals[cutold].max()-xvals[cutold].min()
            ax.set_xlim([xvals[cutold].min()-0.05*xrange, xvals[cutold].max()+0.05*xrange])
        elif cutnew is not None:
            xrange = xvals[cutnew].max()-xvals[cutnew].min()
            ax.set_xlim([xvals[cutnew].min()-0.05*xrange, xvals[cutnew].max()+0.05*xrange])
    else:
        ax.set_xlim(xlims)
        
    if ylims is None:
        if lgcrawdata:
            yrange = yvals.max()-yvals.min()
            ax.set_ylim([yvals.min()-0.05*yrange, yvals.max()+0.05*yrange])
        elif cutold is not None:
            yrange = yvals[cutold].max()-yvals[cutold].min()
            ax.set_ylim([yvals[cutold].min()-0.05*yrange, yvals[cutold].max()+0.05*yrange])
        elif cutnew is not None:
            yrange = yvals[cutnew].max()-yvals[cutnew].min()
            ax.set_ylim([yvals[cutnew].min()-0.05*yrange, yvals[cutnew].max()+0.05*yrange])
        
    else:
        ax.set_ylim(ylims)
        
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(True)
    
    if lgceff:
        ax.plot([],[], linestyle = ' ', label = f'Efficiency of total cut: {cutefftot:.3f}')
        
    ax.legend(markerscale = 6, framealpha = .9)
    
    return fig, ax

        

########### Fit Spectrum ###########

def norm(x,amp, mean, sd):
    return amp*np.exp(-(x - mean)**2/(2*sd**2))

def double_gauss(x, *p):
    a1, a2, m1, m2, sd1, sd2 = p
    y_fit = norm(x,a1, m1, sd1) + norm(x,a2, m2, sd2)
    return y_fit

def get_hist_data(DF, cut, var, bins = 'sqrt'):
    hist, bins = np.histogram(DF[cut][var],bins = bins)
    x = (bins[1:]+bins[:-1])/2
    return x, hist, bins

def hist_data(arr,xrange = None, bins = 'sqrt'):
    if xrange is not None:
        hist, bins = np.histogram(arr,bins = bins, range = xrange)
    else:
        hist, bins = np.histogram(arr,bins = bins)
    x = (bins[1:]+bins[:-1])/2
    return x, hist, bins

def find_peak(arr ,xrange = None, noiserange = None, lgcplotorig = False):
    """
    Function to fit normal distribution to peak in spectrum. 
    
    Parameters
    ----------
        arr: ndarray
            Array of data to bin and fit to gaussian
        xrange: tuple, optional
            The range of data to use when binning
        noiserange: tuple, optional
            nested 2-tuple. should contain the range before 
            and after the peak to be used for subtracting the 
            background
        lgcplotorig: bool, optional
            If True, the original spectrum will be plotted as well 
            as the background subtracted spectrum
    Returns
    -------
        peakloc: float
            The mean of the distribution
        peakerr: float
            The full error in the location of the peak
        fitparams: tuple
            The best fit parameters of the fit; A, mu, sigma
        errors: ndarray
            The uncertainty in the fit parameters
        
            
    """
    
    x,y, bins = hist_data(arr,  xrange)
    if noiserange is not None:
        if noiserange[0][0] >= xrange[0]:
            clowl = noiserange[0][0]
        else:
            clow = xrange[0]
        clowh = noiserange[0][1]
        chighl = noiserange[1][0]
        if noiserange[1][1] <= xrange[1]:
            chighh = noiserange[1][1] 
        else:
            chighh = xrange[1]
        indlowl = (np.abs(x - clowl)).argmin()
        indlowh = (np.abs(x - clowh)).argmin() 
        indhighl = (np.abs(x - chighl)).argmin()
        indhighh = (np.abs(x - chighh)).argmin() - 1
        background = np.mean(np.concatenate((y[indlowl:indlowh],y[indhighl:indhighh])))
        y_noback = y - background
         
    if noiserange is not None:
        y_to_fit = y_noback
    else:
        y_to_fit = y
        
    A0 = np.max(y_to_fit)
    mu0 = x[np.argmax(y_to_fit)]
    sig0 = np.abs(mu0 - x[np.abs(y_to_fit - np.max(y_to_fit)/2).argmin()])
    p0 = (A0, mu0, sig0)
    #y_to_fit[y_to_fit < 0] = 0
    y_to_fit = np.abs(y_to_fit)
    yerr = np.sqrt(y_to_fit)
    yerr[yerr <= 0 ] = 1
    fitparams, cov = curve_fit(norm, x, y_to_fit, p0, sigma = yerr,absolute_sigma = True)
    errors = np.sqrt(np.diag(cov))
    x_fit = np.linspace(xrange[0], xrange[-1], 250)
    
    plt.figure(figsize=(9,6))
    plt.plot([],[], linestyle = ' ', label = f' μ = {fitparams[1]:.2f} $\pm$ {errors[1]:.3f}')
    plt.plot([],[], linestyle = ' ', label = f' σ = {fitparams[2]:.2f} $\pm$ {errors[2]:.3f}')
    plt.plot([],[], linestyle = ' ', label = f' A = {fitparams[0]:.2f} $\pm$ {errors[0]:.3f}')
    if lgcplotorig:
        plt.hist(x, bins = bins, weights = y, histtype = 'step', linewidth = 1, label ='original data', alpha = .3)
        plt.axhline(background, label = 'average background rate', linestyle = '--', alpha = .3)
    if noiserange is not None:
        plt.hist(x, bins = bins, weights = y_noback, histtype = 'step', linewidth = 1, label ='background subtracted data')
        
    plt.plot(x_fit, norm(x_fit, *fitparams))
    plt.legend()
    plt.grid(True, linestyle = 'dashed')
    
    peakloc = fitparams[1]
    peakerr = np.sqrt((fitparams[2]/np.sqrt(fitparams[0]))**2 + errors[1]**2)
    
    return peakloc, peakerr, fitparams, errors

def scale_energy_spec(DF, cut, var, p0, title, xlabel):
    x, hist,bins = get_hist_data(DF,cut, var)
    yerr = np.sqrt(hist)
    yerr[yerr <= 0 ] = 1
    popt, pcov = curve_fit(double_gauss,x, hist, p0, sigma = yerr, absolute_sigma = True )
    error = np.sqrt(np.diag(pcov))
    #print(error)
    x_est = np.linspace(x.min(), x.max(), 1000)
    y_est = double_gauss(x_est,*popt)
    
    maxtemp = np.array([popt[2],popt[3]])
    minint = np.argmin(maxtemp)
    
    if minint == 0:
        amp59, peak59, sig59 = popt[0], popt[2], popt[4]
        err59 = (error[0], error[2], error[4])
        amp64, peak64, sig64 = popt[1], popt[3], popt[5]
        err64 = (error[1], error[3], error[5])
    else:
        amp59, peak59, sig59 = popt[1], popt[3], popt[5]
        err59 = (error[1], error[3], error[5])
        amp64, peak64, sig64 = popt[0], popt[2], popt[4]
        err64 = (error[0], error[2], error[4])
    

    
    peak59_err = np.sqrt((sig59/np.sqrt(amp59))**2 + err59[1]**2) 
    peak64_err = np.sqrt((sig64/np.sqrt(amp64))**2 + err64[1]**2) 
    
    plt.figure(figsize=(9,6))
    plt.hist(x, bins = bins, weights = hist, histtype = 'step', linewidth = 1, label ='data')
    plt.plot(x_est, y_est, 'g', label='Convoluted Gaussian Fit', alpha = .5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.text(peak59*1.02,max(hist)/2,f"μ: {peak59:.2e} $\pm$ {err59[1]:.2e} \n σ: {sig59:.2e}",size = 12)
    plt.text(peak64,max(hist)/3,f"μ: {peak64:.2e} $\pm$ {err64[1]:.2e} \n σ: {sig64:.2e}",size = 12)
    plt.legend()
    plt.grid(True, linestyle = 'dashed')
    plt.xlabel(xlabel)
    plt.ylabel('Counts')
    plt.title(title)
    #plt.savefig(saveFigPath+'OF_amp_fit_fake.png')
    plt.show()

    energy_per_amp59 = (5.9e3)/peak59
    energy_per_amp649= (6.49e3)/peak64


    DF[f'{var}_linear_energy'] = DF[var]*energy_per_amp59
    DF[f'energy_per_{var}'] = energy_per_amp59


   

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
       
    return peak59, peak64, peak59_err, peak64_err



def amp_to_energy(DF,cut, clinearx, clineary, clowenergy, yvar = 'int_bsSub_short_linear_energy', 
                  xvar = 'ofAmps_tdelay', title = 'PT On, Fast Template', yerr= 65, order = 5):
    sns.set_context('notebook')
    x_full = DF[cut][xvar].values

    y_full = DF[cut][yvar].values

    cy = y_full < clineary
    cx = x_full < clinearx
    clow = y_full < clowenergy
    x = x_full[cy & cx]
    y = y_full[cy & cx]

    x_fit = np.linspace(0,x.max(),100)

    def poly(x, *params):
        rf = np.poly1d(params)
        return rf(x)*x

    p0 = np.polyfit(x, y, order)
    popt_poly, pcov = curve_fit(poly, xdata = x, ydata = y, sigma = yerr*np.ones_like(y), p0 = p0[:order], maxfev=100000, 
                                absolute_sigma = True)
    z = popt_poly
    errors = np.sqrt(np.diag(pcov))
    linear_err = errors[-1]
    
    p = np.poly1d(np.concatenate((z,[0])))
    
    p_linear = np.poly1d(np.array([z[-1], 0]))
    chi = (((y_full[clow] - p(x_full[clow]))/yerr)**2).sum()/(len(y_full[clow])-order)

    plt.figure(figsize=(9,6))
    plt.plot(x_full[~(cy & cx)], y_full[~(cy & cx)], marker = '.', linestyle = ' ', label = 'Data passing cuts', ms = 3, alpha = .5)
    plt.errorbar(x,y, marker = '.', linestyle = ' ', yerr = yerr*np.ones_like(y), label = 'Data used for Fit',
                 elinewidth=0.3, alpha =.5, ms = 5,zorder = 50)
    plt.ylim(0,9000)
    plt.xlim(0,x_full.max()*1.05)
    plt.grid(True)
    plt.plot(x_fit,p(x_fit), zorder = 100, label = 'polynomial fit')
    plt.plot(x_fit, p_linear(x_fit),zorder = 200, linestyle = '--', label = 'linear approximation')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.legend()
    plt.ylabel('Integrated Energy [eV]')
    plt.xlabel('OF Amplitude [A]')
    plt.title('OF Amplitude vs Integral Energy Calibration:  \n' + title)

    DF['ofAmps0_poly'] = p(DF['ofAmps'])
    DF['ofAmpst_poly'] = p(DF['ofAmps_tdelay']) 
    DF['ofAmpst_poly_nocon'] = p(DF['ofAmps_tdelay_nocon']) 
    return z[-1], linear_err


def amp_to_energy_v2(xarr, yarr, clinearx, clineary, clowenergy, title = 'PT On, Fast Template', yerr= 65, order = 5):

    x_full = xarr
    y_full = yarr
    def poly(x, *params):
        rf = np.poly1d(params)
        return rf(x)*x
    cy = y_full < clineary
    cx = x_full < clinearx
    clow = y_full < clowenergy
    x = x_full[cy & cx]
    y = y_full[cy & cx]
    x_fit = np.linspace(0,x.max(),100)

    def poly(x, *params):
        rf = np.poly1d(params)
        return rf(x)*x

    p0 = np.polyfit(x, y, order)
    popt_poly = curve_fit(poly, xdata = x, ydata = y, sigma = yerr*np.ones_like(y), p0 = p0[:order], maxfev=100000)
    z = popt_poly[0]
    
    p = np.poly1d(np.concatenate((z,[0])))

    
    chi = (((y_full[clow] - p(x_full[clow]))/yerr)**2).sum()/(len(y_full[clow])-order)
    

    plt.figure(figsize=(9,6))
    plt.plot(x_full, y_full, marker = '.', linestyle = ' ', label = 'Data passing cuts', ms = 3, alpha = .5)
    plt.errorbar(x,y, marker = '.', linestyle = ' ', yerr = yerr*np.ones_like(y), label = 'Data used for Fit',
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

    DF['ofAmps0_poly'] = p(DF['ofAmps'])
    DF['ofAmpst_poly'] = p(DF['ofAmps_tdelay']) 
    DF['ofAmpst_poly_nocon'] = p(DF['ofAmps_tdelay_nocon']) 
    return chi
    
def baseline_res(DF, cut, template, psd, scalefactor ,fs = 625e3, var = 'ofAmps0_energy', title = 'PT Off'):

    nbins = len(template)
    timelen = nbins/fs
    df = fs/nbins
    s = np.fft.fft(template)/nbins
    psd[0]=np.inf
    phi = s.conjugate()/psd
    sigma = (1/(np.dot(phi, s).real*timelen)**0.5)*scalefactor

    x,y, bins = get_hist_data(DF, cut, var)
    x_energy = x

    fit = stats.norm.fit(DF[cut][var])
    plt.figure(figsize=(9,6))
    sns.distplot(DF[cut][var]*scalefactor, kde=False, fit = stats.norm, norm_hist=False
                 , hist_kws = {'histtype': 'step','linewidth':3})
    plt.plot([],[], linestyle = ' ', label = f'Baseline Fit: σ = {fit[1]*scalefactor:.3e} eV')
    plt.plot([],[], linestyle = ' ', label = f'OF Estimate: σ = {sigma:.3e} eV')
    plt.legend()
    plt.grid(True, linestyle = 'dashed')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Normalized PDF')
    plt.title(f'PD2 Baseline Energy Resolution')
    plt.xlim(-20,20)
    
    
    y_to_fit = y  
    A0 = np.max(y_to_fit)
    mu0 = x[np.argmax(y_to_fit)]
    sig0 = np.abs(mu0 - x[np.abs(y_to_fit - np.max(y_to_fit)/2).argmin()])
    p0 = (A0, mu0, sig0)
    fitparams, cov = curve_fit(norm, x, y_to_fit, p0, absolute_sigma = True)
    errors = np.sqrt(np.diag(cov))
    x_fit = np.linspace(x[0], x[-1], 250)
    plt.figure(figsize=(9,6))
    plt.plot([],[], linestyle = ' ', label = f' μ = {fitparams[1]:.2e} $\pm$ {errors[1]:.3e}')
    plt.plot([],[], linestyle = ' ', label = f' σ = {fitparams[2]:.2e} $\pm$ {errors[2]:.3e}')
    plt.plot([],[], linestyle = ' ', label = f' A = {fitparams[0]:.2f} $\pm$ {errors[0]:.3f}')

    plt.hist(x, bins = bins, weights = y, histtype = 'step', linewidth = 1, label ='noise data', alpha = .3)
   
    plt.plot(x_fit, norm(x_fit, *fitparams))
    plt.legend()
    plt.grid(True, linestyle = 'dashed')
    
    
    #plt.savefig('baseline_res_Amps.png')
    
    

def IntegralEnergy(deltaI,Ib=-97e-6,R0 =30.5e-3,deltaT = 1.6e-6,loopL=None,Rp=8e-3,Rsh=5e-3,order='second'):
    JouleToEv=1.0/1.602e-19
 
    Rl=R0+Rp+Rsh
    Rpass=Rp+Rsh
 
    A=(2.0*Rpass/Rl)-1.0
    B=Rpass
 
    if(loopL != None):
        A-=1/loopL
        B+=(Rsh-Rp-R0)/loopL
 
    if(order == 'first'):
        integrand=A*Ib*Rsh*deltaI
    elif(order == 'second'):
        integrand=A*Ib*Rsh*deltaI+B*deltaI**2
    else:
        raise ValueError("Higher order integrals not implemented")
 
    integral=np.sum(integrand)*deltaT
    integral*=JouleToEv
 
    return integral



def get_muon_cut(arr, thresh_pct = 0.95, peak_range = 600):
    #arr = np.abs(arr)

    muons = []
    muon_cut = np.zeros(shape = len(arr), dtype = bool)
    for ii, trace in enumerate(arr):
        trace_max = np.max(trace)
        # check that the maximum value of the trace is above the threshold and
        # that the maximum is decently larger than the minimum

        peak_loc = np.argmax(trace)
        # check that the peak is saturated (this should be true for muons that saturate the
        # detector or muon that rail the amplifier) 
        if ((peak_loc + peak_range) < arr.shape[-1]):
            if (trace[peak_loc+peak_range] >= trace_max*thresh_pct):
                muon_cut[ii] = True                    
    return muon_cut

def get_offset(muon_mean,I_bias=-97e-6, Rsh = 5e-3,Rn = 88.9e-3,Rl=13.15e-3, nbaseline= 6000):
    muon_max = np.max(muon_mean)
    baseline = np.mean(muon_mean[:nbaseline])
    peak_loc = np.argmax(muon_mean)
    muon_saturation = np.mean(muon_mean[peak_loc:peak_loc+200 ])
    muon_deltaI =  muon_saturation - baseline
    V_bias = I_bias*Rsh
    I_n = V_bias/(Rl+Rn)
    I_0 = I_n - muon_deltaI
    I_offset = baseline - I_0
    R_0 = I_n*(Rn+Rl)/I_0 -Rl
    P_0 = I_0*Rsh*I_bias - Rl*I_0**2
    return I_offset,R_0, I_0, P_0

def convert_to_power(trace, I_offset,I_bias,Rsh, Rl):
    V_bias = I_bias*Rsh
    trace_I0 = trace - I_offset
    trace_P = trace_I0*V_bias - (Rl)*trace_I0**2
    return trace_P

def integral_Energy_caleb(trace_power, time, indbasepre, indbasepost):

    baseline_p0 = np.mean(np.hstack((trace_power[:indbasepre],trace_power[indbasepost:])))
    return  np.trapz(baseline_p0 - trace_power, x = time)/constants.e

def td_chi2(signal, template, amp, tshift, fs, baseline=0):
    
    signal_bssub= signal - baseline
    tmplt = amp*np.roll(template, int(tshift*fs))
    chi2 = np.sum((signal_bssub-tmplt)**2)
    
    return chi2
    
    
    
def correct_integral(xenergies, ypeaks, errors, DF):    
    
    def saturation_func(x,a,b,c):
        return a*x+b*x**c
    x = xenergies
    y = ypeaks
    yerr = errors
    
    popt, cov = curve_fit(saturation_func, x, y, sigma = yerr, absolute_sigma=True, maxfev = 10000)

    x_fit = np.linspace(0, xenergies[-1], 100)
    y_fit = saturation_func(x_fit, *popt )


    plt.figure(figsize=(12,8))
    plt.grid(True, linestyle = 'dashed', label = 'Spectral Peaks')
    plt.scatter(x,y)
    plt.errorbar(x,y, yerr=yerr, linestyle = ' ')
    plt.plot(x_fit, y_fit, label = 'y = $ax+bx^c$')
    plt.plot(x_fit,x_fit*ypeaks[0]/xenergies[0],linestyle = '--', c= 'g', label = 'linear calibration from Al fluorescence')
    plt.plot(x_fit,x_fit*ypeaks[-2]/xenergies[-2],linestyle = '--', c = 'r', label = 'linear calibration from Kα')
    plt.fill_between(x_fit, x_fit*(ypeaks[0]-2*errors[0])/xenergies[0], x_fit*(ypeaks[0]+2*errors[0])/xenergies[0] 
                     ,color= 'g', alpha = .3)
    plt.fill_between(x_fit, x_fit*(ypeaks[-2]-2*errors[-2])/xenergies[-2], x_fit*(ypeaks[-2]+2*errors[-2])/xenergies[-2] 
                     ,color= 'r', alpha = .3)
    plt.ylabel('Calculated Integral Energy[eV]')
    plt.xlabel('True Energy [eV]')
    plt.title('Integrated Energy Saturation Correction')

    plt.legend()

    plt.xlim(0, 6800)
    plt.ylim(0, 1100)
    
    slope = ypeaks[0]/xenergies[0]
    
    DF['saturation_corr_int'] = DF.energy_integral1*slope
    DF['dEdI'] = 1/slope
    DF['dEdI_err'] = np.sqrt((xenergies[0]/((ypeaks[0])**2)*yerr[0])**2 + (.5/ypeaks[0])**2)
    
    
    return slope

    
def setplot_style():
    sns.reset_defaults()
    sns.set_context('notebook')
    sns.set_style( {'axes.axisbelow': True,
                    'axes.edgecolor': '.15',
                    'axes.facecolor': 'white',
                    'axes.grid': True,
                    'axes.grid.axis' : 'both',
                    'axes.grid.which' : 'both',
                    'grid.linestyle' : '--',
                    'axes.labelcolor': '.15',
                    'axes.spines.bottom': True,
                    'axes.spines.left': True,
                    'axes.spines.right': True,
                    'axes.spines.top': True,
                    'figure.facecolor': 'w',
                    'legend.frameon': True,
                    'legend.shadow' : False,
                    'legend.borderpad' : 0.5,
                    'legend.fancybox' : False,
                    'legend.framealpha' : 1,
                    'legend.edgecolor' : '.6',
                    'font.family': ['sans-serif'],
                    'font.sans-serif': ['Arial',
                                        'DejaVu Sans',
                                        'Liberation Sans',
                                        'Bitstream Vera Sans',
                                        'sans-serif'],
                    'grid.color': '.8',
                    'image.cmap': 'viridis',
                    'lines.solid_capstyle': 'round',
                    'patch.edgecolor': '.15',
                    'patch.force_edgecolor': True,
                    'text.color': '.15',
                    'xtick.bottom': True,
                    'xtick.color': '.15',
                    'xtick.direction': 'in',
                    'xtick.top': True,
                    'ytick.color': '.15',
                    'ytick.direction': 'in',
                    'ytick.left': True,
                    'ytick.right': True});
