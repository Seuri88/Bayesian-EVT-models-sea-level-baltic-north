import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as plt
import xarray as xr
import arviz as az
from scipy.stats import genextreme as gev
from scipy.interpolate import CloughTocher2DInterpolator
RANDOM_SEED = 2024
rng = np.random.default_rng(RANDOM_SEED)

def missing_dates(df):
    '''Function checking for missing dates in dataframe and returning an array of missing dates '''
    date_index_array = df.index
    start_date, end_date = str(date_index_array[0]), str(date_index_array[-1])
    return pd.date_range(start = start_date, end= end_date).difference(df.index)

def lat_lon_in_BM(swe_sealvl_BM):
    '''Function that takes in a BM data fframe and returns a triple of lists
    corresponding to the coordinates (lat,lon) in the columns of the BM'''
    #List of coordinates, and 
    XY_ = swe_sealvl_BM.columns
    xy = list(XY_)
    xs_ = [XY_[i][1] for i in range(XY_.shape[0])] #long coordinates
    ys_ = [XY_[i][0] for i in range(XY_.shape[0])] #lat coordinates
    return xy, xs_, ys_


def BHM_idata_mean_df(idata, train_sites=None, group="post", return_lvl=False, msb_var_names=None):
    var_names = ["xi_i", "mu_i", "sigma_i", "Zs_5","Zs_10","Zs_20", "Zs_30","Zs_50","Zs_100","Zs_200","Zs_500","Zs_1000","Zs_5000","Zs_10000","Zs_50000", "Zs_100000"]
    if msb_var_names is not None:
        var_names = msb_var_names
    if group == "pred":
        var_names = ["c_"+i for i in var_names]
        idata_to_mean = idata.predictions
    elif group == "prior":
        idata_to_mean = idata.prior
    else:
        idata_to_mean = idata.posterior
    if return_lvl is True:
        var_names = var_names[3:]
    df_idata_mean = idata_to_mean[var_names].mean(dim=["chain", "draw"]).to_pandas()
    if train_sites is not None:
        df_idata_mean["sites"] = train_sites
    return df_idata_mean

def BHM_idata_qtiles_df(idata, qtiles=[0.05, 0.95],group="post", return_lvl=True, msb_var_names=None):
    var_names =  ["xi_i", "mu_i", "sigma_i", "Zs_5","Zs_10","Zs_20", "Zs_30","Zs_50","Zs_100","Zs_200","Zs_500","Zs_1000","Zs_5000","Zs_10000","Zs_50000", "Zs_100000"]
    if msb_var_names is not None:
        var_names = msb_var_names
    if group == "pred":
        var_names = ["c_"+i for i in var_names]
        idata_to_qtiles = idata.predictions
    elif group == "prior":
        idata_to_qtiles = idata.prior
    else:
        idata_to_qtiles = idata.posterior
    if return_lvl is True:
        var_names = var_names[3:]
    df_idata_qtiles = idata_to_qtiles[var_names].quantile(qtiles, dim=["chain", "draw"]).to_dataframe() 
    return df_idata_qtiles

def BHM_idata_hpdi_df(idata, hdi_prob=0.95, group="post", return_lvl=True, msb_var_names=None):
    var_names =  ["xi_i", "mu_i", "sigma_i", "Zs_5","Zs_10","Zs_20", "Zs_30","Zs_50","Zs_100","Zs_200","Zs_500","Zs_1000","Zs_5000","Zs_10000","Zs_50000", "Zs_100000"]
    if msb_var_names is not None:
        var_names = msb_var_names
    groups = {"post":"posterior", "prior":"prior", "pred":"predictions"}
    if group == "pred":
        var_names = ["c_"+i for i in var_names]
    if return_lvl is True:
        var_names = var_names[3:]
    idata_hpdi = az.hdi(idata, hdi_prob=hdi_prob, group=groups[group], var_names=var_names)
    df_idata_hpdi = idata_hpdi.to_dataframe()
    return df_idata_hpdi

def BHM_idata_std_df(idata, train_sites=None, group="post", return_lvl=False, msb_var_names=None):
    var_names = ["xi_i", "mu_i", "sigma_i", "Zs_5","Zs_10","Zs_20", "Zs_30","Zs_50","Zs_100","Zs_200","Zs_500","Zs_1000","Zs_5000","Zs_10000","Zs_50000", "Zs_100000"]
    if msb_var_names is not None:
        var_names = msb_var_names
    if group == "pred":
        var_names = ["c_"+i for i in var_names]
        idata_to_std = idata.predictions
    elif group == "prior":
        idata_to_std = idata.prior
    else:
        idata_to_std = idata.posterior
    if return_lvl is True:
        var_names = var_names[3:]   
    df_idata_std = idata_to_std[var_names].std(dim=["chain", "draw"]).to_pandas()
    if train_sites is not None:
        df_idata_std["sites"] = train_sites
    return df_idata_std

def Baseline_MLM_for_comparison(Baseline_data_train, column=0, round_to=4, as_list=True, as_scipy=False):
    '''Function that takes in a Baseline_data_frame along with argumentes and returns as list or array of MLM estiamtions for 
    the given column of flaots.
    
    Baseline_data_train: type pandas.DataFrame
    column: type int default is 0
    round_to:  type int default is 4
    as_list: type boolean default is true
    as_scipy: type boolean default is false
    
    if as_list is true, then return list, else return np.array 
    if as_scipy is false, then the shape parameter of the GEV is given a negative sign 
    scipy.stats.genextreme uses pposite convention for the sign of the shape parameter). The sign
    should be the same as in the pymc.Model, where the BHM models uses scipy=False as default. 
    
    '''
    if column == 0 and as_scipy is False:
        MLM_for_comparison = np.round(-1*Baseline_data_train.iloc[:,column].to_numpy(dtype="float"), round_to)
    else:
        MLM_for_comparison = np.round(Baseline_data_train.iloc[:,column].to_numpy(dtype="float"), round_to)
    if as_list is True:
        return list(MLM_for_comparison)
    return MLM_for_comparison

def Baseline_ProbPlot(Baseline_base, column=0, dist=None, sparams=None):
    '''
    Function that takes in a Baseline dataframe, a specify column number,
    and returns the scipy stats probplot
    column: type int {0, 1, 2} default==0,
    dist: type scipy.stats.rv_continuous default is None, if dist=None,
    then standard normal,
    sparams: type tuple, if dist is scipy.stats.rv_continuous, then
    sparams need be specfied, see scipy.stats.rv_continuous for detail
    '''
    if dist is not None:
        stats.probplot(Baseline_base.iloc[:,column], dist=dist, sparams=sparams, plot=plt)
    else:
         stats.probplot(Baseline_base.iloc[:,column], dist="norm", plot=plt)
    return plt.show()

def range_and_stats_param_gev_Baseline(Baseline_BM, column=0):
    '''Function that takes in a Baseline frame and returns the
    range, mean and std of the GEV parameter columns
    column: typ int default = 0'''
    return (Baseline_BM.iloc[:,column].max(), Baseline_BM.iloc[:,column].min()), Baseline_BM.iloc[:,column].mean(), Baseline_BM.iloc[:,column].std()

def missing_values_quantify(swe_train_BM, rows=0, thresh=0, case=None):
    '''Function that takes in a data frame, which row to slice from, a thresh hold and case number.
    and returns a data frame obtained by slicing and thresholding, along with a dictionary, contaning
    the case number, rows and thresh hold, the shape of the returned data frame, the number of total
    entries, number of missing values, and the quota of missing values '''
    swe_train_0NaNs = swe_train_BM.iloc[rows:-1,].dropna(axis=1, thresh=thresh)
    n_entrys =swe_train_0NaNs.shape[0]*swe_train_0NaNs.shape[1] 
    n_NaNs = swe_train_0NaNs.isnull().sum().sum()
    quota_NaNs = "Zero Devision is not allowed" 
    if n_NaNs != 0:
        quota_NaNs = round(n_NaNs/n_entrys, 4)
    return swe_train_0NaNs, {"Case ": case,"Row":rows, "Tresh":thresh, "BM shape":swe_train_0NaNs.shape, "#Entrys": n_entrys, "#NaNs": n_NaNs, "NaNs/Entrys":quota_NaNs}


def model_validation_set(swe_train_BM, swe_train_NaNs): return swe_train_BM.drop(swe_train_NaNs.columns, axis=1)


def model_validation(swe_train_BM, swe_train_NaNs, case=None): 
    '''Function that takes in the original data frame, and test set and returns a model validation set,
    along with dict of missing values info
    '''
    return missing_values_quantify(model_validation_set(swe_train_BM, swe_train_NaNs), rows=0, thresh=0, case=case)


def train_test_split_with_info(swe_BM, baseline_tresh=10, rows=50, thresh=34, case=0):
    '''Function that takes in a BlockMaxima dataframe and returns a train and test 
    data frame, along with train and test info dict 
    '''
    swe_train_BM = swe_BM.dropna(axis=1, thresh=baseline_tresh)
    train, train_info = missing_values_quantify(swe_train_BM, rows=rows, thresh=thresh, case=case)
    test, test_info = model_validation(swe_train_BM, train, case=case)
    return train, train_info,  test, test_info

def sup_supprt_Ys(xi_i, mu_i, sigma_i):
    '''Function that takes the three GEV parameters (xarray) arrays 
     and computes the supremum of the support for the GEV dist '''
    func = lambda xi_i, mu_i, sigma_i: mu_i + sigma_i/np.abs(xi_i)
    return xr.apply_ufunc(func, xi_i, mu_i, sigma_i)


def LOO_interpolation(region_base_BM, max_noNaNs_idx=0): return region_base_BM.isnull().sum().sort_values().index[max_noNaNs_idx]
def XY_LOO_interpolation(region_base_BM, max_noNaNs_idx=0): return np.array([np.array(list(np.array(LOO_interpolation(region_base_BM, max_noNaNs_idx))))])

def LatLonMSB(filename_latlon_coordinates):
    '''MSB function'''
    latlon_coords_df = pd.read_csv(filename_latlon_coordinates,delim_whitespace=True, header=0, index_col=["Name"],encoding='latin-1')
    XY_latlon = latlon_coords_df.to_numpy(dtype="float32")
    return latlon_coords_df, XY_latlon

def LOO_interpolation_validation(region_base_BM, max_noNaNs_idx=0):
    '''Function that takes in a BlockMaxaima dataframe, a and a idx and
     returns the a tuple consisting of the column name (lat,lon) and
     a (1,2)-dim-array to be used for BHM model interpolation '''
    return LOO_interpolation(region_base_BM, max_noNaNs_idx), XY_LOO_interpolation(region_base_BM, max_noNaNs_idx)

def Baseline_GEV_param_resampeling(data_train, n_draws):
    shape = []
    location = []
    scale = []
    nsites = data_train.shape[1]
    for site in range(nsites):
        obs_max = data_train.iloc[:,site].dropna().to_numpy(dtype="float")
        for i in range(n_draws):
            xi, mu, sigma = gev.fit(rng.choice(obs_max, size=obs_max.size, replace=True))
            xi = -1 * xi
            if xi > -0.5:
                shape.append(xi)
                location.append(mu)
                scale.append(sigma)
    GEV_MLM_resampeling_dict = {"xi": shape, "mu":location, "sigma":scale}
    Baseline_GEV_param_resampeling_global_df =pd.DataFrame.from_dict(GEV_MLM_resampeling_dict)
    return Baseline_GEV_param_resampeling_global_df

def Baseline_GEV_MLM_param_samples(train, train_sites, n_draws=10000):
    '''
    Function that computes the MLM for the GEV parameters by fiting 
    the data and using resampeling and returns a dict with site as key
    and list of length n_draws, where each entry is (shape, loc, scale) 
    of the GEV.

    train: type BlockMaxima dataframe
    train_sites: type list or int
    n_draws: type int, default is 1000
    
    '''
    site_params = {}
    # generate 1000 samples by resampling data with replacement
    if isinstance(train_sites, int):
        obs_max = train.iloc[:,train_sites].dropna().to_numpy(dtype="float")
        params = []
        for i in range(n_draws):
            xi, mu, sigma = gev.fit(rng.choice(obs_max, size=obs_max.size, replace=True))
            if -1* xi > -0.5:
                params.append((xi, mu, sigma))
            #params.append(gev.fit(rng.choice(obs_max, size=obs_max.size, replace=True)))
        site_params.update({train_sites:params})
    else:
        for site in train_sites:
            obs_max = train.iloc[:,site].dropna().to_numpy(dtype="float")
            params = []
            for i in range(n_draws):
                xi, mu, sigma = gev.fit(rng.choice(obs_max, size=obs_max.size, replace=True))
                if -1* xi > -0.5:
                    params.append((xi, mu, sigma))
                #params.append(
                #    gev.fit(np.random.choice(obs_max, size=obs_max.size, replace=True)))
            site_params.update({site:params})
    return site_params
def Baseline_GEV_params_or_supremum_of_suppGEV_by_bootsraping_MLE(train, train_sites, n_draws=10000, supp=False):
    site_params, supremum_suppGEV_sites = {}, {}
    if isinstance(train_sites, int):
        obs_max = train.iloc[:,train_sites].dropna().to_numpy(dtype="float")
        n_xi, n_mu, n_sigma = [], [], []
        sup_support = []
        for i in range(n_draws):
            xi, mu, sigma = gev.fit(rng.choice(obs_max, size=obs_max.size, replace=True))
            xi = -1*xi
            if xi > -0.5:
                supremum_of_suppGEV = mu + sigma/np.abs(xi) 
                n_xi.append(xi)
                n_mu.append(mu)
                n_sigma.append(sigma)
                sup_support.append(supremum_of_suppGEV)
            #params.append(gev.fit(rng.choice(obs_max, size=obs_max.size, replace=True)))
        n_xi, n_mu, n_sigma = np.array(n_xi, dtype="float"), np.array(n_mu, dtype="float"), np.array(n_sigma, dtype="float")
        params = [n_xi, n_mu, n_sigma]
        site_params.update({train_sites:params})
        supremum_suppGEV_sites.update({train_sites:sup_support})
    else:
        for site in train_sites:
            obs_max = train.iloc[:,site].dropna().to_numpy(dtype="float")
            n_xi, n_mu, n_sigma = [], [], []
            sup_support = []
            for i in range(n_draws):
                xi, mu, sigma = gev.fit(rng.choice(obs_max, size=obs_max.size, replace=True))
                xi = -1*xi
                if xi > -0.5:
                    supremum_of_suppGEV = mu + sigma/np.abs(xi) 
                    n_xi.append(xi)
                    n_mu.append(mu)
                    n_sigma.append(sigma)
                    sup_support.append(supremum_of_suppGEV)
                #params.append(gev.fit(rng.choice(obs_max, size=obs_max.size, replace=True)))
            n_xi, n_mu, n_sigma = np.array(n_xi, dtype="float"), np.array(n_mu, dtype="float"), np.array(n_sigma, dtype="float")
            params = [n_xi, n_mu, n_sigma]
            site_params.update({site:params})
            supremum_suppGEV_sites.update({site:sup_support})
    if supp:
        return site_params, supremum_suppGEV_sites
    else:
        return site_params
    

def Baseline_GEV_r_year_return_lvl_bootsraping_MLE(train, train_sites, return_periods, n_draws=10000):
    site_params = Baseline_GEV_MLM_param_samples(train, train_sites, n_draws=n_draws)
    site_return_lvl = {}
    prob = 1/ return_periods
    for site in train_sites:
            params = site_params[site]
            levels = []
            draws = len(params)
            for i in range(draws):
                levels.append(gev.isf(prob, *params[i]))
            levels = np.array(levels)
            site_return_lvl.update({site:levels})
    return site_return_lvl 


    

def Baseline_GEV_return_lvl_samples_by_MLM(train, train_sites, n_draws=10000, msb=False):
    # intialize list for return levels
    site_params = Baseline_GEV_MLM_param_samples(train, train_sites, n_draws=n_draws)
    return_periods = np.array([1/5, 1/10, 1/20, 1/30, 1/50, 1/100,1/200, 1/500, 1/1000,1/5000, 1/10000, 1/50000, 1/100000])
    if msb:
        return_periods = np.array([1/5, 1/10, 1/15, 1/20, 1/25, 1/30, 1/35, 1/40, 1/45, 1/50, 1/55, 1/60, 1/65, 1/70, 
                                   1/75, 1/80, 1/85, 1/90, 1/95, 1/100, 1/150, 1/200, 1/250, 1/300, 1/350, 1/400, 1/450,
                                   1/500]) # , 1/550, 1/600, 1/650, 1/700, 1/750, 1/800, 1/850, 1/900, 1/950, 1/1000]) #1/1500,
                                   #1/2000, 1/2500, 1/3000, 1/3500, 1/4000, 1/4500, 1/5000, 1/5500, 1/6000, 1/6500,  1/7000,
                                   #1/7500, 1/8000, 1/8500, 1/9000, 1/9500, 1/10000, 1/15000, 1/20000, 1/25000,1/30000, 1/35000, 1/40000,
                                    #  1/45000, 1/50000, 1/55000, 1/60000, 1/65000, 1/70000, 1/75000, 1/80000, 1/85000, 1/90000, 1/95000, 1/100000])
    # calculate return levels for each of the 1000 samples
    site_return_lvl = {}
    if isinstance(train_sites, int):
        params = site_params[train_sites]
        levels = []
        n_draws = len(params)
        for i in range(n_draws):
            levels.append(gev.isf(return_periods, *params[i]))
        levels = np.array(levels)
        site_return_lvl.update({train_sites:levels})
    else:
        for site in train_sites:
            params = site_params[site]
            levels = []
            n_draws = len(params)
            for i in range(n_draws):
                levels.append(gev.isf(return_periods, *params[i]))
            levels = np.array(levels)
            site_return_lvl.update({site:levels})
    return site_return_lvl 

def Baseline_GEV_return_lvl_mean_ci_by_MLM(train, train_sites, n_draws=10000, alpha=0.95, msb=False):
    site_return_lvl = Baseline_GEV_return_lvl_samples_by_MLM(train, train_sites, n_draws=n_draws, msb=msb)
    site_ci_return_lvl = {}
    if isinstance(train_sites, int):
        mean = site_return_lvl[train_sites].mean(axis=0)
        ci_min, ci_max = np.quantile(a=site_return_lvl[train_sites], q=[(1 - alpha) / 2, (1 + alpha) / 2], axis=0)
        ci = np.array([ci_min, mean, ci_max])
        site_ci_return_lvl.update({train_sites:ci})
    else:
        for site in train_sites:
            mean = site_return_lvl[site].mean(axis=0)
            ci_min, ci_max = np.quantile(a=site_return_lvl[site], q=[(1 - alpha) / 2, (1 + alpha) / 2], axis=0)
            ci = np.array([ci_min, mean, ci_max])
            site_ci_return_lvl.update({site:ci})
    return site_ci_return_lvl


def BHM_Max_or_Min_notNaNs_site_idx(train, min=False):
    '''Function that returns the site idx of the site with 
    max number of notNaNs, return is int.
    
    train: type pd.DataFrame
    min: type boolean default False. If min=False, return 
    site idx for max. If min=True, return site idx for mmin.
    '''
    columns_list = list(train.columns)
    column_name = train.notna().sum().idxmax()
    if min is True:
        column_name = train.notna().sum().idxmin()
    return columns_list.index(column_name)

def BHM_models_idata_stacking(idata, var_name, train_sites, group="post", gev_dist=False):
    '''Function that takes in an BHM inferrence object, a variable name in the model,
    and list of sites (coordinates) in the inferrence object, and returns a list np.arrays
    obtained by stacking an xarray dataset w.r.t
    idata: type xarray.Dataset, 
    var_name: type str,
    train_sites: type list
    gev_dist: type boolean, default gev_dist=False. If gev_dist is False,
    then stacking is done using dims ("chain","draw"). If gev_dist is True,
    then stacking is done using ("chain", "draw","years"), only for Ys variable
    return is list of (chain*draws,)-dim np.arrays
    '''
    if group == "prior":
        idata_group =  idata.prior
    elif group == "pred":
        idata_group =  idata.predictions
    elif group == "post":
        idata_group = idata.posterior
    else:
        idata_group = idata.posterior
    if gev_dist is True:
        stack_samples = ("chain", "draw","years")
        idata_var_name_stacked = idata.posterior_predictive[[var_name]].stack(samples=stack_samples).to_array()
    else:
        stack_samples=("chain", "draw")
        idata_var_name_stacked = idata_group[[var_name]].stack(samples=stack_samples).to_array()
    if isinstance(train_sites, int):
        var_name_stacked_sites = idata_var_name_stacked.sel(station=train_sites).to_numpy()[0,:]
    else: 
        var_name_stacked_sites = [idata_var_name_stacked.sel(station=i).to_numpy()[0,:] for i in train_sites]
    return var_name_stacked_sites

def BHM_models_var_name_stacked_dict(models_names, models_idata, var_name, train_sites, group="post",  gev_dist=False):
    '''Function that takes in a list of model_names, and a list of BHM infeerrence objects, var_name
    and list of sites (coordinates) in the inferrence object, and returns dict where the keys are 
    the model_names, and the values are lists of np.arrays obtained by stacking the BHM inferrence
    xarray dataset w.r.t  '''
    return {models_names[i]:BHM_models_idata_stacking(models_idata[i], var_name, train_sites,group, gev_dist) for i in range(len(models_names))}

def BHM_interpolate_train_and_test_latlon_value(idata, out_of_sample, XY_train, XY_test_longer_array, X, Y, var_name, mean_or_std="mean", hdi_prob=0.95):

    Zs_mean_or_std_var_name = idata.posterior[[var_name]]
    Zs_mean_or_std_c_var_name = out_of_sample.predictions[["c_"+ var_name]] 
    if mean_or_std.lower() =="std":
        Zs_mean_or_std_var_name = Zs_mean_or_std_var_name.std(dim=("chain","draw")).to_array().to_numpy()[0]
        Zs_mean_or_std_c_var_name = Zs_mean_or_std_c_var_name.std(dim=("chain","draw")).to_array().to_numpy()[0]
    elif mean_or_std.lower() == "lower_hpdi":
        Zs_mean_or_std_var_name = az.hdi(idata,hdi_prob=hdi_prob, group="posterior", var_names=var_name).to_array().to_numpy()[0][:,0]
        Zs_mean_or_std_c_var_name = az.hdi(out_of_sample,hdi_prob=hdi_prob, group="predictions", var_names="c_"+ var_name).to_array().to_numpy()[0][:,0]
    elif mean_or_std.lower() == "higher_hpdi":
        Zs_mean_or_std_var_name = az.hdi(idata,hdi_prob=hdi_prob, group="posterior", var_names=var_name).to_array().to_numpy()[0][:,1]
        Zs_mean_or_std_c_var_name = az.hdi(out_of_sample,hdi_prob=hdi_prob, group="predictions", var_names="c_"+ var_name).to_array().to_numpy()[0][:,1]
    else:
        Zs_mean_or_std_var_name = Zs_mean_or_std_var_name.mean(dim=("chain","draw")).to_array().to_numpy()[0]
        Zs_mean_or_std_c_var_name = Zs_mean_or_std_c_var_name.mean(dim=("chain","draw")).to_array().to_numpy()[0]
    XY_train_and_test_longer_array = np.array(list(XY_train) + list(XY_test_longer_array))
    Zs_mean_in_and_out_sample_var_name = np.array(list(Zs_mean_or_std_var_name)+list(Zs_mean_or_std_c_var_name))
    interp = CloughTocher2DInterpolator(XY_train_and_test_longer_array, Zs_mean_in_and_out_sample_var_name)
    Z = interp(X, Y)
    return Z

def Baseline_geographical_spread_GEV_Params(train, Baseline_region_BM, geographical_spread, key_station="station", supp_GEV=True):
    columns_list = list(train.columns)
    geographical_spread_stations = geographical_spread[key_station]
    geographical_spread_lat_lons = [columns_list[i] for i in geographical_spread_stations]
    Baseline_region_BM_geographical_spred = Baseline_region_BM.iloc[:,:3].loc[geographical_spread_lat_lons]
    if supp_GEV is True:
         Baseline_region_BM_geographical_spred["supp_GEV"] =  Baseline_region_BM_geographical_spred["mu_i"]-Baseline_region_BM_geographical_spred["sigma_i"] / Baseline_region_BM_geographical_spred["xi_i"]
    return Baseline_region_BM_geographical_spred
def Baseline_geographical_spread_GEV_Parameter(train, Baseline_region_BM, geographical_spread, param, key_station="station", supp_GEV=True):
    Baseline_region_BM_geographical_spred = Baseline_geographical_spread_GEV_Params(train, Baseline_region_BM, geographical_spread, key_station, supp_GEV)
    return list(np.round(Baseline_region_BM_geographical_spred[param].to_numpy(dtype="float"), 4))

def Baseline_geographical_spread_GEV_Params(train, Baseline_region_BM, geographical_spread, key_station="station", supp_GEV=True):
    columns_list = list(train.columns)
    geographical_spread_stations = geographical_spread[key_station]
    geographical_spread_lat_lons = [columns_list[i] for i in geographical_spread_stations]
    Baseline_region_BM_geographical_spred = Baseline_region_BM.iloc[:,:3].loc[geographical_spread_lat_lons]
    if supp_GEV is True:
         Baseline_region_BM_geographical_spred["supp_GEV"] =  Baseline_region_BM_geographical_spred["mu_i"]-Baseline_region_BM_geographical_spred["sigma_i"] / Baseline_region_BM_geographical_spred["xi_i"]
    return Baseline_region_BM_geographical_spred
def Baseline_geographical_spread_GEV_Parameter(train, Baseline_region_BM, geographical_spread, param, key_station="station",round_to=4, supp_GEV=True):
    Baseline_region_BM_geographical_spred = Baseline_geographical_spread_GEV_Params(train, Baseline_region_BM, geographical_spread, key_station, supp_GEV)
    return list(np.round(Baseline_region_BM_geographical_spred[param].to_numpy(dtype="float"), round_to))

def BHM_supp_GEV(idata, group, train_sites):
    if group == "pred":
        xi =  BHM_models_idata_stacking(idata,"c_xi_i" , train_sites, group=group)
        mu =  BHM_models_idata_stacking(idata,"c_mu_i" , train_sites, group=group)
        sigma =  BHM_models_idata_stacking(idata,"c_sigma_i" , train_sites, group=group)
    else:
        xi =  BHM_models_idata_stacking(idata,"xi_i" , train_sites, group=group)
        mu =  BHM_models_idata_stacking(idata,"mu_i" , train_sites, group=group)
        sigma =  BHM_models_idata_stacking(idata,"sigma_i" , train_sites, group=group)
    if isinstance(train_sites, int):
        supp_GEV = mu + sigma/np.abs(xi)
    else:
        supp_GEV = [mu[i] + sigma[i]/np.abs(xi[i]) for i in range(len(train_sites))]
    return supp_GEV
