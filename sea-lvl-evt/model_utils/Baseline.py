import pandas as pd
import numpy as np
from scipy.stats import genextreme as gev

def return_level_Baseline(sub_sealvl_BM):
    '''Function that takes in a Block Maxima pandas dataframe,
    fits the columns (the sequnce of annual maxima for each
    site) to a GEV using scipy.stats.genextrme, and 
    drives the shape, location and scale paramters. From there,
    the function computes the return level, 
    for 5, 10, 30, 50, 100, 500, 1000, 5000, 10000
    '''
    #xy = list(sub_sealvl_BM.T.index)
    xy = list(sub_sealvl_BM.columns)
    return_level_all = {}
    return_periods = np.array([1/5, 1/10, 1/20, 1/30, 1/50, 1/100,1/200, 1/500, 1/1000,1/5000, 1/10000, 1/50000, 1/100000])
    for site in xy:
        notNaNs = sub_sealvl_BM[site].notna().sum()
        data = sub_sealvl_BM[site].dropna().to_numpy(dtype="float")
        shape, loc, scale = gev.fit(data)
        ci = gev.interval(confidence=0.95, c=shape, loc=loc, scale=scale)
        std = gev.std(c=shape, loc=loc, scale=scale)
        site_eva_info = [shape, loc, scale, notNaNs,  std, (ci[0].round(4), ci[1].round(4))]
        for return_period in return_periods:
            site_eva_info.append(gev.isf(return_period, shape, loc, scale))
        return_level_all.update({site:site_eva_info})
    return return_level_all

def Baseline(sub_sealvl_BM, scipy=False):
    '''Function that takes in a Block Maxima pandas dataframe
    and returns dataframe, with site coordinates as index, 
    shape, location, scale, number annual maxima used (do not allow NaNs), std of the distribution, 
    CI of the distributiion, togther with return level for return 
    periods 5, 10, 30, 50, 100, 500, 1000, 5000, 10000
    scipy: type boolen default is False
    If scipy is True, then the shape parameters is returned using the scipy convention,
    else -1*shape is returned'''
    xy = list(sub_sealvl_BM.columns)
    column_name = ["xi_i", "mu_i", "sigma_i", "notNaNs", "std_of_dist", "CI_of_dist", "Zs_5", "Zs_10", "Zs_20", "Zs_30", "Zs_50", "Zs_100", "Zs_200","Zs_500","Zs_1000", "Zs_5000", "Zs_10000", "Zs_50000", "Zs_100000"]
    separate_eva_dict = return_level_Baseline(sub_sealvl_BM)
    Baseline_df = pd.DataFrame.from_dict(separate_eva_dict, orient='index', columns=column_name)
    if scipy is False:
        Baseline_df["xi_i"] = -1*Baseline_df["xi_i"]
    Baseline_df["supp_GEV"] = Baseline_df["mu_i"] - Baseline_df["sigma_i"] / Baseline_df["xi_i"]
    return Baseline_df
