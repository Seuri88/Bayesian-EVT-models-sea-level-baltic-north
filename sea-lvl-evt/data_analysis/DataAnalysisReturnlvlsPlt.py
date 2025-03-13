import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import math
from DataAnalysisUtil import (BHM_idata_mean_df, BHM_idata_qtiles_df,  
                              BHM_idata_hpdi_df, Baseline_GEV_return_lvl_mean_ci_by_MLM)
from matplotlib.ticker import ScalarFormatter
plt.style.use('tableau-colorblind10')



def BHM_hpdi_and_MLM_ci_return_lvl_Loo_plot(out_of_sample, test_sites, region_BM, model_name, Loo_column, LOO=True, region_sites=None, hdi_prob=0.95, alpha=0.95, n_draws=50000, figsize=(10,10),show=False):
    '''Function that takes in a BHM (pymc) out of sample inference object, a site in the test_sites list, the regional BM, the
    model name, and the leave one out column name. From there the function returns a comparative return curve plot, comparing the model with the maximum likelihood
    estimated  (MLM) return level, along with HDI and CI. The model compares the return level to the MLM computaed for all annual
    obs in the regional Block Maxima matrix (dataframe).

    out_of_sample: type xarray
    test_site: type int
    region_BM: type pd.Dataframe
    model_name: type str
    Loo_column: type int
    LOO: type boolean default LOO=True. If LOO is true, than function knows that there is only on column in the 
    BHM_idata_mean data frame. Else it assumes multiple columns. 
    hdi_prob: type float between (0,1) default hid_prob=0.95. The value to use when computing the HDI
    alpha: type float between (0,1) default alpha=0.95. The value used to compute the alpah confidence interval
    n_draws: type int default n_draws=10000. The number of resampelings used for the MLM estimations
    figsize: type tuple default figsize=(15,15). The size of the figure. '''
    if LOO is True:
        df_idata_mean = BHM_idata_mean_df(idata=out_of_sample, group="pred", return_lvl=True).T
    else:
        df_idata_mean = BHM_idata_mean_df(idata=out_of_sample, group="pred", return_lvl=True).T[test_sites]
    df_idata_hpdi = BHM_idata_hpdi_df(idata=out_of_sample, hdi_prob=hdi_prob, group="pred", return_lvl=True).T[test_sites]
    Loo_column_index = list(region_BM.columns).index(Loo_column)
    site_ci_return_lvl = Baseline_GEV_return_lvl_mean_ci_by_MLM(train=region_BM, train_sites=Loo_column_index, n_draws=n_draws, alpha=alpha)
    line = region_BM.iloc[:,Loo_column_index].max()
    return_periods =[5, 10, 20, 30, 50, 100,200, 500,1000,5000, 10000, 50000, 100000]
    number_of_obs_at_loo = region_BM[Loo_column].notna().sum()

    fig, ax = plt.subplots(figsize=figsize, dpi=96)
    ax.semilogx()
    ax.grid(True, which="both")
    site_supp_Ys, site_supp_Ys_MLM = 1, 1
    for end_point in ["lower", "higher"]:
        ax.plot(return_periods,
                df_idata_hpdi[end_point].values/ site_supp_Ys,
                color="Black",
                lw=1,
                ls="-",
                zorder=15,
                )
    ax.fill_between(return_periods,
                    df_idata_hpdi["lower"].values/ site_supp_Ys, df_idata_hpdi["higher"].values/ site_supp_Ys,
            facecolor='C1',
            edgecolor="None",
            alpha=0.25,
            zorder=10,
            label = f" the {hdi_prob} HPDI- for {model_name}"
            )       
    ax.plot(return_periods,
        df_idata_mean.values/ site_supp_Ys, 
        color="#F85C50",
        lw=2,
        ls="-",
        zorder=25,
        label = f"the mean return lvl for {model_name}"
        )
    
    #Compute the CI 

    for bound in [0, 2]:
        ax.plot(return_periods,
                site_ci_return_lvl[ Loo_column_index][bound] / site_supp_Ys_MLM,
                color="Black",
                lw=1,
                ls="--",
                zorder=15,
                )
    ax.fill_between(return_periods,
                    site_ci_return_lvl[ Loo_column_index][0] / site_supp_Ys_MLM, site_ci_return_lvl[ Loo_column_index][2] / site_supp_Ys_MLM,
            facecolor='C2',
            edgecolor="None",
            alpha=0.25,
            zorder=10,
            label = f" The {alpha} CI by MLM using {n_draws} samples"
            )       
    ax.plot(return_periods,
            site_ci_return_lvl[ Loo_column_index][1] / site_supp_Ys_MLM,
        color="#F85C50",
        lw=2,
        ls="--",
        zorder=25,
        label = f"the mean return lvl by MLM using {n_draws} samples",
        )
    plt.axhline(y=line, color='black', linestyle= (5, (10, 3)), lw=2.5, label=f"max of all obs at site: {Loo_column_index}:{Loo_column}")
    ax.legend()
    ax.set_title(f" BHM {model_name} vs {alpha} CI  return level plot, using the LOO_site {Loo_column} with #obs-max={number_of_obs_at_loo}, for MLM {n_draws} samples")
    if region_sites is not None:
        ax.set_title(f" Comparative return level plot for {model_name} at the traning station {region_BM[Loo_column]}")
        if LOO is True:
            ax.set_title(f" Comparative return level plot for {model_name} at the LOO station {region_BM[Loo_column]}")
    if show is False:
        return fig, ax
    fig.show();










def BHM_hpdi_and_MLM_ci_return_lvl_site_compare_plots(idata, site, train, region_BM, model_name, group="post", region_sites=None, hdi_prob=0.95, alpha=0.95, n_draws=50000, figsize=(10,10),show=True):
    '''Function that takes in a BHM (pymc) inference object, a site in the traning data, the train data, the regional BM, and the
    model names. From there the function returns a comparative return curve plot, comparing the model with the maximum likelihood
    estimated  (MLM) return level, along with HDI and CI. The model compares the return level to the MLM computaed for all annual
    obs in the regional Block Maxima matrix (dataframe).

    idata: type xarray
    site: type int
    train: type pd.Dataframe 
    region_BM: type pd.Dataframe
    model_name: type str
    group: type str default group="post" {post, prior}. If group is "prior", then comparison is made w.r.t. the prior distribution,
    else post and thus comparison is made w.r.t. the posterior distributions. 
    hdi_prob: type float between (0,1) default hid_prob=0.95. The value to use when computing the HDI
    alpha: type float between (0,1) default alpha=0.95. The value used to compute the alpah confidence interval
    n_draws: type int default n_draws=10000. The number of resampelings used for the MLM estimations
    figsize: type tuple default figsize=(15,15). The size of the figure. '''
    #Get the mean of the return levels for each site    
    df_idata_mean = BHM_idata_mean_df(idata=idata, group=group, return_lvl=True).T[site]
    df_idata_hpdi = BHM_idata_hpdi_df(idata, hdi_prob=hdi_prob, group= group, return_lvl=True)

    #supp_Ys_mean = idatat_mean[["supp_Ys"]].to_pandas()
    #site_supp_Ys_MLM
    site_coord = list(train.columns)[site]
    region_sites = list(region_BM.columns)
    region_site_idx = region_sites.index(site_coord)
    site_ci_return_lvl = Baseline_GEV_return_lvl_mean_ci_by_MLM(region_BM, train_sites=region_site_idx, n_draws=n_draws, alpha=alpha)
    number_of_obs_at_loo = region_BM[site_coord].notna().sum()
    line = region_BM.iloc[:,region_site_idx].max()
    return_periods =[5, 10, 20, 30, 50, 100,200, 500,1000,5000, 10000, 50000, 100000]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=96)
    ax.semilogx()
    ax.grid(True, which="both")
    site_supp_Ys, site_supp_Ys_MLM = 1, 1
    for end_point in ["lower", "higher"]:
        ax.plot(return_periods,
                df_idata_hpdi.loc[(site,end_point)].values/ site_supp_Ys,
                color="Black",
                lw=1,
                ls="-",
                zorder=15,
                )
    ax.fill_between(return_periods,
                    df_idata_hpdi.loc[(site,"lower")].values/ site_supp_Ys, df_idata_hpdi.loc[(site,"higher")].values/ site_supp_Ys,
            facecolor='C1',
            edgecolor="None",
            alpha=0.25,
            zorder=10,
            label = f" the {hdi_prob} HPDI- for {model_name}"
            )       
    ax.plot(return_periods,
        df_idata_mean.values/ site_supp_Ys, 
        color="#F85C50",
        lw=2,
        ls="-",
        zorder=25,
        label = f"the mean return lvl for {model_name}"
        )
    
    #Compute the CI    
    for bound in [0, 2]:
        ax.plot(return_periods,
                site_ci_return_lvl[  region_site_idx ][bound] / site_supp_Ys_MLM,
                color="Black",
                lw=1,
                ls="--",
                zorder=15,
                )
    ax.fill_between(return_periods,
                    site_ci_return_lvl[  region_site_idx ][0] / site_supp_Ys_MLM, site_ci_return_lvl[  region_site_idx ][2] / site_supp_Ys_MLM,
            facecolor='C2',
            edgecolor="None",
            alpha=0.25,
            zorder=10,
            label = f" The {alpha} CI by MLM using {n_draws} samples"
            )       
    ax.plot(return_periods,
            site_ci_return_lvl[ region_site_idx ][1] / site_supp_Ys_MLM,
        color="#F85C50",
        lw=2,
        ls="--",
        zorder=25,
        label = f"the mean return lvl by MLM using {n_draws} samples",
        )
    plt.axhline(y=line, color='black', linestyle= (5, (10, 3)), lw=2.5, label=f"max of all obs at site: {site} : {site_coord}  ")
    ax.legend()
    ax.set_title(f" BHM {model_name} vs {alpha} CI  return level plot, using for BHM the {group} data, for MLM with #obs-max={number_of_obs_at_loo}, and {n_draws} samples, at site :" + str(site))
    if region_sites is not None:
        ax.set_title(f" Comparative return level plot for {model_name} at the training station {region_BM[site_coord]}")
    if show is False:
        return fig,ax
    fig.show();






def BHM_hpdi_and_MLM_ci_return_lvl_Loo_compare_plot(out_of_sample_comp, test_sites, region_BM, Loo_column, region_sites=None, hdi_prob=0.95, alpha=0.95, n_draws=50000, figsize=(10,10),show=True):
    '''Function that takes in a dictionary with model names as keys and out_of_sample inference xarray as values, test_sites, region_BM and
    a leave one out validation site, and returns a comparative return level plot. 

     out_of_sample_comp: type dict
    test_site: type int
    region_BM: type pd.Dataframe
    Loo_column: type int
    hdi_prob: type float between (0,1) default hid_prob=0.95. The value to use when computing the HDI
    alpha: type float between (0,1) default alpha=0.95. The value used to compute the alpah confidence interval
    n_draws: type int default n_draws=10000. The number of resampelings used for the MLM estimations
    figsize: type tuple default figsize=(15,15). The size of the figure.
    '''
    model_names = list(out_of_sample_comp.keys())
    df_idata_means = [BHM_idata_mean_df(idata=out_of_sample_comp[key], group="pred", return_lvl=True).T for key in model_names]
    df_idata_hpdis = [BHM_idata_hpdi_df(idata=out_of_sample_comp[key], hdi_prob=hdi_prob, group="pred", return_lvl=True).T[test_sites] for key in model_names]
    Loo_column_index = list(region_BM.columns).index(Loo_column)
    site_ci_return_lvl = Baseline_GEV_return_lvl_mean_ci_by_MLM(train=region_BM, train_sites=Loo_column_index, n_draws=n_draws, alpha=alpha)
    line = region_BM.iloc[:,Loo_column_index].max()
    return_periods =[5, 10, 20, 30, 50, 100,200, 500,1000,5000, 10000, 50000, 100000]
    number_of_obs_at_loo = region_BM[Loo_column].notna().sum()
    bounds = ["lower", "higher"]
    fig, ax = plt.subplots(figsize=figsize, dpi=96)
    ax.semilogx()
    ax.grid(True, which="both")
    ls_dict = {0:"solid", 1: "dotted", 2:"dashed",3:"dashdot"}
    for index, df_idata_hpdi in enumerate(df_idata_hpdis):
        for end_point in bounds:
                ax.plot(return_periods,
                    df_idata_hpdi[end_point].values,
                    color="black",
                    lw=2.5,
                    ls=ls_dict[index],
                    zorder=15,
                    )
        ax.plot(return_periods,
                df_idata_means[index],
                color="#F85C50",
                lw=2.5,
                ls=ls_dict[index],
                zorder=25,
                label = f"mean for {model_names[index]}"
            )
        ax.fill_between(return_periods,
            df_idata_hpdi["lower"].values, df_idata_hpdi["higher"].values,
            facecolor='C'+str(index),
            edgecolor="None",
            alpha=0.25,
            zorder=10,
            label = f" the {hdi_prob} HPDI- for {model_names[index]}"
            )
    
    #Compute the CI 

    for bound in [0, 2]:
        ax.plot(return_periods,
                site_ci_return_lvl[ Loo_column_index][bound],
                color="Black",
                lw=1,
                ls=(0, (3, 1, 1, 1, 1, 1)),
                zorder=15,
                )
    ax.fill_between(return_periods,
                    site_ci_return_lvl[ Loo_column_index][0], site_ci_return_lvl[ Loo_column_index][2] ,
            facecolor='C4',
            edgecolor="None",
            alpha=0.25,
            zorder=10,
            label = f" The {alpha} CI by MLM using {n_draws} samples"
            )       
    ax.plot(return_periods,
            site_ci_return_lvl[Loo_column_index][1] ,
        color="#F85C50",
        lw=2,
        ls=(0, (3, 1, 1, 1, 1, 1)),
        zorder=25,
        label = f"the mean return lvl by MLM using {n_draws} samples",
        )
    plt.axhline(y=line, color='black', linestyle= (5, (10, 3)), lw=2.5, label=f"max of all obs at site: {Loo_column_index}:{Loo_column}")
    ax.legend()
    ax.set_title(f" BHM {model_names} vs {alpha} CI  return level plot, using the LOO_site {Loo_column} with #obs-max={number_of_obs_at_loo}, for MLM {n_draws} samples")
    ax.tick_params(axis="both", labelsize=20)
    if region_sites is not None:
        ax.set_title(f" Comparative return level plot for {model_names} at the LOO station {region_BM[Loo_column]}")
    if show is False:
        return fig, ax
    fig.show();


def BHM_hdpi_and_MLM_ci_return_lvl_multiple_sites_plot(idata, subcollection_dict, train, region_BM, model_name, region_sites, 
                                                       key_station="station", key_region="location", msb_var_names=None, hdi_prob=0.95, alpha=0.95, n_draws=10000, fontsize_rows=17, fontsize_title=30, figsize=(30,30), show=True):
    sites = subcollection_dict[key_station]
    locations = subcollection_dict[key_region]
    df_idata_mean = BHM_idata_mean_df(idata=idata, group="post", return_lvl=True, msb_var_names=msb_var_names).T[sites]
    df_idata_hpdi = BHM_idata_hpdi_df(idata=idata, hdi_prob=hdi_prob, group="post", return_lvl=True, msb_var_names=msb_var_names)
    if msb_var_names is None:
        msb = False
    else:
        msb = True
    #supp_Ys_mean = idatat_mean[["supp_Ys"]].to_pandas()
    train_columns = list(train.columns)
    sites_coords = [train_columns[site] for site in sites]
    region_stations = list(region_BM.columns)
    region_site_idx_list = [region_stations.index(site_coord) for site_coord in sites_coords]
    sites_ci_return_lvl = Baseline_GEV_return_lvl_mean_ci_by_MLM(region_BM, train_sites=region_site_idx_list, n_draws=n_draws, alpha=alpha, msb=msb)
    number_of_obs_at_sites = [region_BM[site_coord].notna().sum() for site_coord in sites_coords]
    lines = [region_BM.iloc[:,region_site_idx].max() for region_site_idx in region_site_idx_list]
    return_periods =[5, 10, 20, 30, 50, 100,200, 500,1000,5000, 10000, 50000, 100000]
    if msb:
        return_periods =[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                        80, 85, 90, 95, 100, 150, 200, 250, 300, 350, 400, 450, 500] 
                        #,550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
        #return_periods =[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
        #                  80, 85, 90, 95, 100, 150, 200, 250, 300, 350, 400, 450, 500, 
        #                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1500, 2000, 
        #                  2500,3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 
        #                  8000,8500, 9000, 9500, 10000, 15000, 20000, 25000, 30000, 35000, 
        #                  40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]
    rows = 2
    if len(sites) > 3:
        rows = math.ceil((len(sites))/3)
    fig, ax = plt.subplots(nrows=rows,ncols=3, figsize=figsize, dpi=96)
    columns_count = 0
    rows_count = 0
    for i, site in enumerate(sites):
        if columns_count == 3:
            columns_count = 0
            rows_count += 1
        ax[rows_count, columns_count].semilogx()
        ax[rows_count, columns_count].grid(True, which="both")
        for end_point in ["lower", "higher"]:   
            ax[rows_count, columns_count].plot(return_periods,
                    df_idata_hpdi.loc[(site,end_point)].values ,
                    color="Black",
                    lw=1,
                    ls="-",
                    zorder=15,
                    )
        ax[rows_count, columns_count].fill_between(return_periods,
                        df_idata_hpdi.loc[(site,"lower")].values , df_idata_hpdi.loc[(site,"higher")].values ,
                facecolor='C1',
                edgecolor="None",
                alpha=0.25,
                zorder=10,
                label = f" the {hdi_prob} HPDI- for {model_name}"
                )       
        ax[rows_count, columns_count].plot(return_periods,
            df_idata_mean[site].values , 
            color="#F85C50",
            lw=2,
            ls="-",
            zorder=25,
            label = f"the mean for {model_name}"
            )
        
        #Compute the CI    
        for bound in [0, 2]:
            ax[rows_count, columns_count].plot(return_periods,
                    sites_ci_return_lvl[region_site_idx_list[i]][bound],
                    color="Black",
                    lw=1,
                    ls="--",
                    zorder=15,
                    )
        ax[rows_count, columns_count].fill_between(return_periods,
                        sites_ci_return_lvl[region_site_idx_list[i]][0], sites_ci_return_lvl[region_site_idx_list[i]][2],
                facecolor='C2',
                edgecolor="None",
                alpha=0.25,
                zorder=10,
                label = f" The {alpha} CI"
                )       
        ax[rows_count, columns_count].plot(return_periods,
                sites_ci_return_lvl[region_site_idx_list[i]][1],
            color="#F85C50",
            lw=2,
            ls="--",
            zorder=25,
            label = f"the mean for MLM",
            )
        ax[rows_count, columns_count].axhline(y=lines[i], color='black', linestyle= (5, (10, 3)), lw=2.5, label="the maximum observation")
        ax[rows_count, columns_count].legend(loc='upper left', fontsize=16)
        if msb:
            ax[rows_count, columns_count].xaxis.set_major_formatter(ScalarFormatter())
        ax[rows_count, columns_count].set_title(f"Station {region_sites[sites_coords[i]]} ({site}) with #Total-obs = {number_of_obs_at_sites[i]} ", fontdict = {'fontsize':fontsize_rows} )
        ax[rows_count, columns_count].tick_params(axis="both", labelsize=17)
        columns_count += 1
    fig.suptitle(f"Comparative return level plot for {model_name} over the subcollection of station: {locations}", fontsize=fontsize_title)
    if msb:
        fig.suptitle(f"MSB: Comparative return level plot for {model_name} over the subcollection of station: {locations}", fontsize=fontsize_title)
    for axes in ax.flatten():
        if not axes.has_data():
            fig.delaxes(axes)
    if show is False:
        return fig,ax
    fig.show();

def BHM_hpdi_and_MLM_ci_return_lvl_Loo_multiple_models_plot(out_of_sample_comp, test_sites, region_BM, Loo_column, region_sites=None, hdi_prob=0.95, alpha=0.95, n_draws=10000, fontsize_rows=17, fontsize_title=30, figsize=(15,10), show=True):
    model_names = list(out_of_sample_comp.keys())
    test_site = test_sites[0]
    df_idata_means = [BHM_idata_mean_df(idata=out_of_sample_comp[key], group="pred", return_lvl=True).T for key in model_names]
    df_idata_hpdis = [BHM_idata_hpdi_df(idata=out_of_sample_comp[key], hdi_prob=hdi_prob, group="pred", return_lvl=True).T[test_sites] for key in model_names]
    Loo_column_index = list(region_BM.columns).index(Loo_column)
    site_ci_return_lvl = Baseline_GEV_return_lvl_mean_ci_by_MLM(train=region_BM, train_sites=Loo_column_index, n_draws=n_draws, alpha=alpha)
    line = region_BM.iloc[:,Loo_column_index].max()
    return_periods =[5, 10, 20, 30, 50, 100,200, 500,1000,5000, 10000, 50000, 100000]
    number_of_obs_at_loo = region_BM[Loo_column].notna().sum()
    bounds = ["lower", "higher"]
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=figsize, dpi=96)
    for i, model_name in enumerate(model_names):
        df_idata_mean = df_idata_means[i]
        df_idata_hpdi = df_idata_hpdis[i]
        ax[i].semilogx()
        ax[i].grid(True, which="both")
        for end_point in bounds:
            ax[i].plot(return_periods,
                    df_idata_hpdi[(test_site, end_point)].values,
                    color="Black",
                    lw=1,
                    ls="-",
                    zorder=15,
                    )
        ax[i].fill_between(return_periods,
                        df_idata_hpdi[(test_site, "lower")].values, df_idata_hpdi[(test_site, "higher")].values,
                facecolor='C1',
                edgecolor="None",
                alpha=0.25,
                zorder=10,
                label = f" the {hdi_prob} HPDI- for {model_name}"
                )       
        ax[i].plot(return_periods,
            df_idata_mean.values,
            color="#F85C50",
            lw=2,
            ls="-",
            zorder=25,
            label = f"the mean for {model_name}"
            )
        
        #Compute the CI 

        for bound in [0, 2]:
            ax[i].plot(return_periods,
                    site_ci_return_lvl[ Loo_column_index][bound],
                    color="Black",
                    lw=1,
                    ls="--",
                    zorder=15,
                    )
        ax[i].fill_between(return_periods,
                        site_ci_return_lvl[ Loo_column_index][0],site_ci_return_lvl[ Loo_column_index][2],
                facecolor='C2',
                edgecolor="None",
                alpha=0.25,
                zorder=10,
                label = f" The {alpha} CI"
                )       
        ax[i].plot(return_periods,
                site_ci_return_lvl[ Loo_column_index][1],
            color="#F85C50",
            lw=2,
            ls="--",
            zorder=25,
            label = f"the mean for MLM"
            )
        ax[i].axhline(y=line, color='black', linestyle= (5, (10, 3)), lw=2.5, label="the maximum observation")
        ax[i].legend(loc='upper left', fontsize=16)
        ax[i].set_title(f"Station {region_sites[Loo_column]} with #Total-obs = {number_of_obs_at_loo} ", fontdict = {'fontsize':fontsize_rows} )
        ax[i].tick_params(axis="both", labelsize=17)
    fig.suptitle(f"Comparative return level plot for {model_names[0]} and {model_names[1]} at the LOO station {region_sites[Loo_column]}", fontsize=fontsize_title)
    if show is False:
        return fig,ax
    fig.show();








def BHM_hpdi_MSB_LOO_stations_plot(out_of_sample_L,msb_var_names,  msb_sites, msb_sites_names,model_name, group="pred", hdi_prob=0.95, figsize=(30,30),return_periods_short_mbs=True, show=True):
    if return_periods_short_mbs is True:
        return_periods =[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                        80, 85, 90, 95, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                        #550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    else:
        return_periods =[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                        80, 85, 90, 95, 100, 150, 200, 250, 300, 350, 400, 450, 500, 
                        550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1500, 2000, 
                        2500,3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 
                        8000,8500, 9000, 9500, 10000, 15000, 20000, 25000, 30000, 35000, 
                        40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]
    return_lvls_to_use = len(return_periods)
    df_idata_mean = BHM_idata_mean_df(idata=out_of_sample_L, group=group, return_lvl=True, msb_var_names=msb_var_names).iloc[:,:return_lvls_to_use].T[msb_sites]
    df_idata_hpdi = BHM_idata_hpdi_df(idata=out_of_sample_L, hdi_prob=hdi_prob, group=group, return_lvl=True, msb_var_names=msb_var_names).iloc[:,:return_lvls_to_use]
    df_idata_quantile = BHM_idata_qtiles_df(idata=out_of_sample_L, qtiles=[0.01,0.05,0.10, 0.90, 0.95, 0.99], group=group, return_lvl=True, msb_var_names=msb_var_names).iloc[:,:return_lvls_to_use]
    rows = 2
    if len(msb_sites) > 3:
        rows = math.ceil((len(msb_sites))/3)
    fig, ax = plt.subplots(nrows=rows,ncols=3, figsize=figsize, dpi=96)
    columns_count = 0
    rows_count = 0
    for i, site in enumerate(msb_sites):
        if columns_count == 3:
            columns_count = 0
            rows_count += 1
        ax[rows_count, columns_count].semilogx()
        ax[rows_count, columns_count].grid(True, which="both")
        for end_point in ["lower", "higher"]:   
            ax[rows_count, columns_count].plot(return_periods,
                    df_idata_hpdi.loc[(site,end_point)].values ,
                    color="Black",
                    lw=1,
                    ls="-",
                    zorder=15,
                    )
        ax[rows_count, columns_count].fill_between(return_periods,
                        df_idata_hpdi.loc[(site,"lower")].values , df_idata_hpdi.loc[(site,"higher")].values ,
                facecolor='C0',
                edgecolor="None",
                alpha=0.25,
                zorder=10,
                label = f" the {hdi_prob} HPDI- for {model_name}"
                )       
        ax[rows_count, columns_count].plot(return_periods,
            df_idata_mean[site].values , 
            color="#F85C50",
            lw=2,
            ls="-",
            zorder=25,
            label = f"the mean for {model_name}"
            )
        ax[rows_count, columns_count].fill_between(return_periods, 
            df_idata_quantile.loc[(0.01, site)].values, 
            df_idata_quantile.loc[(0.99, site)].values,
            facecolor='C1',
            edgecolor="None",
            alpha=0.25,
            zorder=10,
            label = "(0.01, 0.99)-quantiles", linestyle="dotted")
        ax[rows_count, columns_count].fill_between(return_periods, 
            df_idata_quantile.loc[(0.05, site)].values, 
            df_idata_quantile.loc[(0.95, site)].values,
            facecolor='C2',
            edgecolor="Black",
            alpha=0.25,
            zorder=10,
            label = "(0.05, 0.95)-quantiles")
        #ax[rows_count, columns_count].fill_between(return_periods, df_idata_quantile.loc[(0.10, site)].values,df_idata_quantile.loc[(0.90, site)].values,
        #        facecolor='C3', edgecolor="None",
        #        alpha=0.25,
        #        zorder=10,
        #        label = "(0.10, 0.90)-quantiles")
        ax[rows_count, columns_count].set_title(f"Station {msb_sites_names[i]}  ({site}) ", fontdict = {'fontsize':15} )
        ax[rows_count, columns_count].legend(loc='upper left', fontsize=20)
        ax[rows_count, columns_count].xaxis.set_major_formatter(ScalarFormatter())
        ax[rows_count, columns_count].tick_params(axis="both", labelsize=20)

        columns_count += 1
    for axes in ax.flatten():
        if not axes.has_data():
            fig.delaxes(axes)
    fig.suptitle(f" Comparative return level plot for MSB out-of-sample data", fontsize=30)
    if show is False:
        return fig, ax
    fig.show();
