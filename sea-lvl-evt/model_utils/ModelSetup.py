import numpy as np
import pandas as pd
import pytensor as pyt
from pytensor.tensor import TensorVariable

def test_runs(sealvl_BM, run_starts=1950):
    '''Function that takes in a sub_sealvl_BM frame, and corresponding sites_coords dict,
    and run_starts with default 1950 and returns subset dataframe with only sites that have
    observation prior to run_starts,
    sub_sealvl_noeva_BM type pandas dataframe, 
    runs_starts type: int, default=1950'''
    sites_in_run = []
    site_years = lambda df, xy: df[df[xy].isnull()==False].index.tolist()
    sites = list(sealvl_BM.columns)
    sites_coords = {i:[sites[i], site_years(sealvl_BM, sites[i])] for i in range(len(sites))}
    for i, site in enumerate(sites_coords):
        if int(sites_coords[site][1][0]) <= run_starts:
            sites_in_run.append(sites[i])
    return sealvl_BM[sites_in_run]

def model_setup(sub_sealvl_BM, common=False, knots=None):
    '''Function that takes in a sealv_BM pandas dataframe and returns
    the triple: XY, coord, dims. 
    XY is a numpy array of shape (n,2) with n the number of sites, 
    and 2 the is the (lat,lon) for each site,
    coord is a dictonary contaning coordinates to use for the pm.Model(),
    and dims is the shape of sub_sealvl_BM (number of years, number of sites) '''
    XY = np.array(list(sub_sealvl_BM.columns.to_numpy()))
    run_sites_idx = list(range(XY.shape[0]))
    years_idx = list(range(sub_sealvl_BM.shape[0]))
    coord = {"years":years_idx, "station": run_sites_idx, "feature": ["longitude", "latitude"]}
    dims = sub_sealvl_BM.shape
    if common:
        return {"years":years_idx, "station": run_sites_idx}, dims, sub_sealvl_BM
    elif knots is None:
        return XY, coord, dims, sub_sealvl_BM
    else:
        n_knots = [i for i in range(knots)]
        knots_sites = np.ones(shape=(len(n_knots),len(run_sites_idx)))
        coord = {"years":years_idx, "knots": n_knots, "station": run_sites_idx, "feature": ["longitude", "latitude"]}
        return XY, n_knots, knots_sites, coord, dims, sub_sealvl_BM

def new_normalized_gauss_kernel(x, n_knots=5, tau=1):
    """
    Function that takes in a np.array of x,y coordinates and computes
    the normalized gaussian radial kernel, with a normailzing constant,
    ensuring that np.array(normalized).sum() = 1, the function returns 
    a numpy array
    """
    w = 2 * tau**2
    knots = np.linspace(np.floor(x.min()), np.ceil(x.max()), n_knots)
    normalized = []
    for k in knots:
        norm_xk = np.linalg.norm(x-k, axis=1)
        K = 1/(np.pi * w ) * np.array(np.exp(-norm_xk/w))
        wl = K / np.array( [1/(np.pi * w ) * np.exp(- np.linalg.norm(x-k, axis=1) / w) for k in knots]).sum()
        normalized.append(wl)
    return np.array(normalized)

def Gauss_Kernel_Normalized(XY: TensorVariable, K: TensorVariable, Tau: TensorVariable):
    """
        Function that takes in three tensor variables, evaluates each of the variables and computes the new normalized 
        gauss kernel returned as a shared tensor variable. 
        
        XY: type TensorVariable,
        K: type TensorVariable
        Tau: type TensorVariable
        
    """
    x = XY.eval()
    n_knots=K
    tau=Tau.eval()
    return pyt.shared(new_normalized_gauss_kernel(x, n_knots, tau))


def get_same_site_coords(train0, train1, key="station", compare_cmems_smhi=True):
    '''Function that takes in two pandas.Dataframes and returns two coordinate dict
    where the key a pymc model dimension, and the value is a positional column list of the 
    mutual stations present both data sets'''
    train0_sites = [i for i in train0.columns] 
    train1_sites = [i for i in train1.columns]
    train0_sites.sort(), train1_sites.sort()
    same_sites0 = []
    same_sites1 = []
    for sites in train0_sites:
        if sites in train1_sites:
            i = train0_sites.index(sites)
            j = train1_sites.index(sites)
            same_sites0.append(i)
            same_sites1.append(j)
    return {"station": same_sites0}, {"station": same_sites1}

