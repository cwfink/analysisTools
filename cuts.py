import numpy as np
from qetpy.utils import removeoutliers
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



def savecut(cutarr, name, description):
    """
    Function to save cut arrays. The function first checks the current_cut/ directory
    to see if the desired cut exists. If not, the cut is saved. If the current cut does
    exist, the fuction checkes if the cut has changed. If it has, the old version is archived
    and the new version of the cut is saved. If nothing in the cut has changed, nothing is done.
    
    Cut arrays are saved as .npz files, with keys: 'cut' -> the cut array, and 
    'cutdescription' -> a short message about how the cut was calculated.
    
    Note, the function expects current cuts to be in directory: current_cuts/, and archived cuts
    to be in directory: archived_cuts/. If these folders do not exist in the given path, they will
    be created by save_cut()
    
    Parameters
    ----------
        cutarr: ndarray
            Array of bools
        name: str
            The name of cut to be saved
        description: str
            Very short description of how the cut was calculated
            
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
        ctemp = np.load(f'{path}/current_cuts/{name}.npz')['cut']
        
        if np.array_equal(ctemp, cutarr):
            print(f'cut: {name} is already up to date.')
        else:
            print(f'updating cut: {name} in directory: {path}/current_cuts/ and achiving old version')
            np.savez_compressed(f'{path}/current_cuts/{name}', cut = cutarr, cutdescription=description)
            
            files_old = glob(f'{path}/archived_cuts/{name}_v*')
            if len(files_old) > 0:
                latestversion = int(sorted(files_old)[-1].split('_v')[-1].split('.')[0])
                version = int(latestversion +1)
            else:
                version = 0
            np.savez_compressed(f'{path}/archived_cuts/{name}_v{version}', cut = ctemp, cutdescription=description)
            print(f'old cut is saved as: {path}/archived_cuts/{name}_v{version}.npz')
        
    except FileNotFoundError:
        print(f'No existing version of cut: {name}. \n Saving cut: {name}, to directory: current_cuts/')
        np.savez_compressed(f'{path}/current_cuts/{name}', cut = cutarr, cutdescription=description)
        
    return

 


def listcuts(whichcuts = 'current'):
    """
    Function to return all the available cuts saved in current_cuts/
    or archived_cuts/
    
    Parameters
    ----------
        whichcuts: str, optional
            String to specify which cuts to return. Can be 'current' or
            'archived'. If 'current', only the cuts in the current_cuts/ 
            directory are returned. If 'archived', the old cuts in the 
            archived_cuts/ directory are returned
        
    Returns
    -------
        allcuts: list
            List of names of all current cuts available
            
    Raises
    ------
        ValueError
            If whichcuts is not 'current' or 'archived'
            
    """
    
    allcuts = []
    path = os.path.dirname(os.path.abspath(__file__))
    
    
    if whichcuts == 'current':
        cutdir = 'current_cuts'
    elif whichcuts == 'archived':
        cutdir = 'archived_cuts'
    else:
        raise ValueError("Please select either 'current' or 'archived'")
    
    if not os.path.isdir(f'{path}/{cutdir}'):
        print('No cuts have been generated yet')
        return
    
    files = glob(f'{path}/{cutdir}/*')
    
    if len(files) == 0:
        print('No cuts have been generated yet')
        return
    else:
        for file in files:
            allcuts.append(file.split('/')[-1].split('.')[0])
        return allcuts
        
def loadcut(name, lgccurrent = True):
    """
    Function to load a cut mask from disk into memory. The name should just be the 
    base name of the cut, with no file extension. If an archived cut is desired, the 
    version of the cut must be part of the name, i.e. 'cbase_v3'
    
    Parameters
    ----------
        name: str
            The name of the cut to be loaded
        lgccurrent: bool, optional
            If True, the current cut with corresponding name is loaded,
            if False, the archived cut is loaded
    
    Returns
    -------
        cut: ndarray
            Array of booleans
    
    Raises
    ------
        FileNotFoundError
            If the user specified cut cannot be loaded
            
    """
    path = os.path.dirname(os.path.abspath(__file__))
    
    
    if lgccurrent:
        cutdir = 'current_cuts'
    else:
        cutdir = 'archived_cuts'
    
    
    try:
        cut = np.load(f'{path}/{cutdir}/{name}.npz')['cut']
        return cut
    except FileNotFoundError:
        raise FileNotFoundError(f'{name} not found in {path}/{cutdir}/')
        
        
    
        
def loadcutdescription(name, lgccurrent = True):
    """
    Function to load the description of a cut. The name should just be the 
    base name of the cut, with no file extension. If an archived cut is desired, the 
    version of the cut must be part of the name, i.e. 'cbase_v3'
    
    Parameters
    ----------
        name: str
            The name of the cut to be loaded
        lgccurrent: bool, optional
            If True, the current cut with corresponding name is loaded,
            if False, the archived cut is loaded
    
    Returns
    -------
        cutmessage: str
            the description of the cut stored with the array
    
    Raises
    ------
        FileNotFoundError
            If the user specified cut cannot be loaded
            
    """
    path = os.path.dirname(os.path.abspath(__file__))
    
    
    if lgccurrent:
        cutdir = 'current_cuts'
    else:
        cutdir = 'archived_cuts'
    
    
    try:
        cutmessage = np.load(f'{path}/{cutdir}/{name}.npz')['cutdescription']
        return cutmessage
    except FileNotFoundError:
        raise FileNotFoundError(f'{name} not found in {path}/{cutdir}/')    
            
        
        
        
        
        
        
        

                          
                          
    