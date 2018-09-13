import numpy as np
from tescal.utils import removeoutliers
from glob import glob
import os





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



def savecut(cutarr, name):
    """
    Function to save cut arrays. The function first checks the current_cut/ directory
    to see if the desired cut exists. If not, the cut is saved. If the current cut does
    exist, the fuction checkes if the cut has changed. If it has, the old version is archived
    and the new version of the cut is saved. If nothing in the cut has changed, nothing is done.
    
    Note, the function expects current cuts to be in directory: current_cuts/, and archived cuts
    to be in directory: archived_cuts/. If these folders do not exist in the given path, they will
    be created by save_cut()
    
    Parameters
    ----------
        cutarr: ndarray
            Array of bools
        name: str
            The name of cut to be saved
            
    Returns
    -------
        None
    
    """
    path = os.path.dirname(os.path.abspath(__file__))
    
    # check if 'current_cuts/' and 'archived_cuts/' exist. If not, make them
    if not os.path.isdir(f'{path}/current_cuts'):
        print('folder: current_cuts/ does not exist, it is being created now')
        os.makedirs(f'{path}/current_cuts')
    if not os.path.isdir(f'{path}/archived_cuts'):
        print('folder: archived_cuts/ does not exist, it is being created now')
        os.makedirs(f'{path}/archived_cuts')
    
    # check if there is a current cut, then check if it has been changed
    try:
        ctemp = np.load(f'{path}/current_cuts/{name}.npy')
        
        if np.array_equal(ctemp, cutarr):
            print(f'cut: {name} is already up to date.')
        else:
            print(f'updating cut: {name} in directory: {path}/current_cuts/ and achiving old version')
            np.save(f'{path}/current_cuts/{name}.npy', cutarr)
            
            files_old = glob(f'{path}/archived_cuts/{name}_v*')
            if len(files_old) > 0:
                latestversion = sorted(files_old)[-1].split('_v')[-1].split('.')[0]
                version = int(latestversion +1)
            else:
                version = 0
            np.save(f'{path}/archived_cuts/{name}_v{version}.npy', ctemp)
            print(f'old cut is saved as: {path}/archived_cuts/{name}_v{version}.npy')
        
    except FileNotFoundError:
        print(f'No existing version of cut: {name}. \n Saving cut: {name}, to directory: current_cuts/')
        np.save(f'{path}/current_cuts/{name}.npy', cutarr)
        
    
    
                          
                          
    