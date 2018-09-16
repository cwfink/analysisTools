import numpy as np
from qetpy.utils import removeoutliers
from glob import glob
import os
import git





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


class Error(Exception):
   """Base class for other exceptions"""
   pass

class GitError(Error):
   """Raised when there is a git related issue"""
   pass

class CutUtils(object):
    """
    Class to manage the saveing, loading, and archiving of cut masks for a decector analysis/DM search. 
    This class keeps track of the most current version of user defined cuts, and archives previous versions
    of the same cut, saving all versions. A user message is also saved with each cut, where the user provides
    a short description of how the cut was generatred. This class will create the directory structure to save 
    all the cuts, and if working with multiple people, this class will also automatically backup the directory
    structure, and cut masks, on the shared github repository. Every time a cut is saved or loaded, this class
    will make sure the data on the repository, and the local machine are up to with each other.
    
    This class assumes that it is being run inside the remote github repository directory. 
    
    Note, if the automatic syncing with the repository functionality is to be used, the user must provide git 
    with thier ssh key, as well as clone the repository using ssh protocal, not HTTP
    
    
    Attributes
    ----------
    relativepath: str
        This is the user supplied path to create the cut mask directory. This will be a relative path 
        within the repository. For example, if working on an analysis for a particular run and dataset, 
        set relativepath = 'Run44_dataset1'. Note, this path need not exist before creation of the CutUtils 
        object, it will be created upon instantiation.
    fullpath: str
        The absolute path to the save path for the cut masks. It will be the absolute path to the repository 
        where the analysis is being done, plus the relativepath
    lgcsync: bool
        If True, the local working directory will be synced with the github repository. If False, the saved
        cuts will only be kept locally. 
    repo: GitPython Repo object
        The github repository address to be used if connecting CutUtils to github
        
        
    """
    
    def __init__(self,relativepath, lgcsync = True):
        """
        Initialization of the CutUtils object. If the directory structure to save the cuts is not already in 
        place, they will be created upon instantiation of a CutUtils object
        
        Parameters
        ----------
            relativepath: str
                This is the user supplied path to create the cut mask directory. This will be a relative path 
                within the repository. For example, if working on an analysis for a particular run and dataset, 
                set relativepath = 'Run44_dataset1'. Note, this path need not exist before creation of the CutUtils 
                object, it will be created upon instantiation.
            lgcsync: bool, optional
                If True, the local working directory will be synced with the github repository. If False, the saved
                cuts will only be kept locally. 
                
        """
        self.lgcsync = lgcsync
        
        if relativepath[0] != '/':
            relativepath = '/' + relativepath
        if relativepath[-1] == '/':
            relativepath = relativepath[:-1]
            
        self.relativepath = relativepath
        self.fullpath = os.path.dirname(os.path.abspath(__file__)) + self.relativepath
        
        if not os.path.isdir(f'{self.fullpath}/current_cuts'):
            print(f'folder: {self.relativepath}/current_cuts/ does not exist, it is being created now')
            os.makedirs(f'{self.fullpath}/current_cuts')
        if not os.path.isdir(f'{self.fullpath}/archived_cuts'):
            print(f'folder: {self.relativepath}/archived_cuts/ does not exist, it is being created now')
            os.makedirs(f'{self.fullpath}/archived_cuts')
        if lgcsync:  
            print('Connecting to GitHub Repository, please insure that your ssh keys have been uploaded to GitHub account')
            self.repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.repo = None

    def savecut(self,cutarr, name, description):
        """
        Function to save cut arrays. The function first checks the current_cut/ directory
        to see if the desired cut exists. If not, the cut is saved. If the current cut does
        exist, the fuction checkes if the cut has changed. If it has, the old version is archived
        and the new version of the cut is saved. If nothing in the cut has changed, nothing is done.

        Cut arrays are saved as .npz files, with keys: 'cut' -> the cut array, and 
        'cutdescription' -> a short message about how the cut was calculated.

 

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
            
        Raises
        ------
            GitError
                If there is an issue with the status of the remote repo, or 
                a problem with saving, a GitError is raised

        """
        
        # If connecting with GitHub repo, first do a pull to make sure remote is
        # up to date. 
        if self.lgcsync:
            self.dopull()
        
        path = self.fullpath

        # check if there is a current cut, then check if it has been changed
        try:
            ctemp = np.load(f'{path}/current_cuts/{name}.npz')['cut']

            if np.array_equal(ctemp, cutarr):
                print(f'cut: {name} is already up to date.')
            else:
                print(f'updating cut: {name} in directory: {path}/current_cuts/ and achiving old version')
                np.savez_compressed(f'{path}/current_cuts/{name}', cut = cutarr, cutdescription=description)
                if self.lgcsync:
                    print('syncing new cut with GitHub repo...')
                    self.pushcut(f'{self.relativepath[1:]}/current_cuts/{name}.npz', description)

                files_old = glob(f'{path}/archived_cuts/{name}_v*')
                if len(files_old) > 0:
                    latestversion = int(sorted(files_old)[-1].split('_v')[-1].split('.')[0])
                    version = int(latestversion +1)
                else:
                    version = 0
                np.savez_compressed(f'{path}/archived_cuts/{name}_v{version}', cut = ctemp, cutdescription=description)
                print(f'old cut is saved as: {path}/archived_cuts/{name}_v{version}.npz')
                if self.lgcsync:
                    print('syncing old cut with GitHub repo...')
                    self.pushcut(f'{self.relativepath[1:]}/archived_cuts/{name}_v{version}.npz', f'archived cut {name}')

        except FileNotFoundError:
            print(f'No existing version of cut: {name}. \n Saving cut: {name}, to directory: {path}/current_cuts/')
            np.savez_compressed(f'{path}/current_cuts/{name}', cut = cutarr, cutdescription=description)
            if self.lgcsync:
                print('syncing new cut with GitHub repo...')
                self.pushcut(f'{self.relativepath[1:]}/current_cuts/{name}.npz', description)

        return




    def listcuts(self, whichcuts = 'current'):
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
            GitError
                If there is an issue with the status of the remote repo, 
                a GitError is raised

        """
        
        # If connecting with GitHub repo, first do a pull to make sure remote is
        # up to date. 
        if self.lgcsync:
            self.dopull()

        allcuts = []
        path = self.fullpath


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

    def loadcut(self, name, lgccurrent = True):
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
            GitError
                If there is an issue with the status of the remote repo,
                a GitError is raised

        """
        # If connecting with GitHub repo, first do a pull to make sure remote is
        # up to date. 
        if self.lgcsync:
            self.dopull()
        
        path = self.fullpath


        if lgccurrent:
            cutdir = 'current_cuts'
        else:
            cutdir = 'archived_cuts'


        try:
            cut = np.load(f'{path}/{cutdir}/{name}.npz')['cut']
            return cut
        except FileNotFoundError:
            raise FileNotFoundError(f'{name} not found in {path}/{cutdir}/')




    def loadcutdescription(self, name, lgccurrent = True):
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
            GitError
                If there is an issue with the status of the remote repo,
                a GitError is raised

        """
        
        # If connecting with GitHub repo, first do a pull to make sure remote is
        # up to date. 
        if self.lgcsync:
            self.dopull()
        
        path = self.fullpath


        if lgccurrent:
            cutdir = 'current_cuts'
        else:
            cutdir = 'archived_cuts'


        try:
            cutmessage = np.load(f'{path}/{cutdir}/{name}.npz')['cutdescription']
            return str(cutmessage)
        except FileNotFoundError:
            raise FileNotFoundError(f'{name} not found in {path}/{cutdir}/')    


    
    def dopull(self):
        """
        Simple function to do a git pull from the master branch. It also checks the 
        git status after the pull and checks that the remote is up to date
        
        Parameters
        ----------
            None
        
        Returns
        -------
            None
        
        Raises
        ------
            GitError
                If there is an issue with the status of the remote repo, or 
                a problem with saving, a GitError is raised
        
        
        """
        self.repo.git.pull('origin', 'master')
        if not self.repo.git.status().split('\n')[1] == "Your branch is up to date with 'origin/master'.":
            raise GitError('Remote repository is not up to date with branch master, this may cause issues with saving \
            \n please make sure repositories are in sync before proceeding')
        
        
    def pushcut(self, file, commitmessage):
        """
        Function to push newly made cut to master branch of repository
        
        Paramters
        ---------
            file: str
                The relative path to the file to be added
            commitmessage: str
                The commit message for the added file
                
        Returns
        -------
            None
            
        Raises
        ------
            GitError
                If there is an issue with the status of the remote repo, or 
                a problem with saving, a GitError is raised
                
        """
        
        
        self.dopull()
        
        self.repo.git.add(file)
        self.repo.git.commit(m = commitmessage)
        try: 
            self.repo.git.push('origin', 'master')
        except:
            raise GitError(f'Unable to push new {file} to master')
            

        
        
        
        

                          
                          
    