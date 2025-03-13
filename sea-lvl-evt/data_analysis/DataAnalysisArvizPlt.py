import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import pandas as pd
import arviz as az
from scipy.stats import genextreme as gev
from DataAnalysisUtil import Baseline_geographical_spread_GEV_Params, BHM_supp_GEV



def BHM_subset_denisty_comparsion_plot(all_idata_models_dict,  geographical_spread, var_names, key_station="station", key_region="location", group="posterior", hdi_prob=0.95, figsize=(22,25)):
    ''' '''
    all_model_names = list(all_idata_models_dict.keys()) 
    subset_of_stations = geographical_spread[key_station]
    region = geographical_spread[key_region]
    all_idata_only_subset = [all_idata_models_dict[model].isel(station=subset_of_stations) for model in all_model_names]
    combine_dims = {"chain", "draw"}
    if var_names == "Ys":
        combine_dims = {"chain", "draw", "years"}
        group="posterior_predictive"
    axes = az.plot_density(all_idata_only_subset, var_names=[var_names], combine_dims=combine_dims, hdi_prob=hdi_prob, group=group, figsize=figsize, data_labels=all_model_names)
    fig = axes.flatten()[0].get_figure()
    fig.suptitle(f"Comparative density plots for models:{all_model_names[:]} over the sub-region {region}" )
    plt.show()

def BHM_subset_pair_plot(idata, geographical_spread, var_names, model_name, single_station=None, key_station="station", key_region="location", 
                         group="posterior", hdi_probs= [0.5, 0.75,0.90, 0.95,0.99], quantiles=False, divergences=True, color="blues", fontsize_title=30,figsize=(19,22)):
    colors = {"Blues":"C0", "Oranges":"C1", "Greens":"C2", "Reds":"C3"}
    color = color.lower().capitalize()
    geographical_spread_coords = {key_station: geographical_spread[key_station]}
    locations = geographical_spread[key_region]
    qtiles = None
    combine_dims = {"chain", "draw"}

    if quantiles is True:
        qtiles = [0.01, 0.05, 0.1, 0.25, 0.75, 0.90, 0.95, 0.99]
    if color not in list(colors.keys()):
        color = "Blues"
    if var_names == "Ys":
        combine_dims = {"chain", "draw", "years"}
    if single_station is not None:
        geographical_spread_coords = {key_station: geographical_spread[key_station][single_station]}
    az.plot_pair(idata, group=group,
             var_names=var_names,combine_dims=combine_dims,
              coords=geographical_spread_coords, marginals=True, 
             kind="kde",
               kde_kwargs={
        "hdi_probs": hdi_probs,  # Plot 30%,A 60% and 90% HDI contours
        "contourf_kwargs": {"cmap": color}
    },marginal_kwargs={"color":colors[color],"quantiles": qtiles },divergences=divergences,
    figsize=figsize);
    #if var_names[:2] == "Zs":
    #    param = f"The {var_names[3:]}-year return period for the stations{one_station}: {geographical_spread[key_region]} "
    #elif var_names[:4] == "c_Zs":
    #    param = f"The {var_names[5:]}-year return period for the stations{one_station}: {geographical_spread[key_region]} "
    #elif var_names == "Ys":
    #    param = f"The GEV {group} dist over for the stations{one_station}: {geographical_spread[key_region]} "
    #else:
    #    param = f"The {var_names} GEV params over the region {geographical_spread[key_region]} "
    plt.suptitle(f"\n Pair plots for {model_name} using the {group} inference data over the subset {locations}", fontsize=fontsize_title)
    if single_station is not None:
        geographical_spread_station = geographical_spread[key_station][single_station]
        plt.suptitle(f"\n Pair plots for {model_name} using the {group} inference data at station {geographical_spread_station}", fontsize=fontsize_title)
    #if quantiles is True:
    #    plt.suptitle(param + f" \n Pair plots for {model_name} using the {group} data \n with q-quantiles in the marginals, for q in {qtiles} ")
    plt.show()


def BHM_geographical_spread_GEV_param_posterior_plot(idataH,train, Baseline_region_BM, geographical_spread_Skagerrak, model_name, key_station="station", key_location="location", group='posterior', hdi_prob=0.95,supp_GEV=False):
    stations = geographical_spread_Skagerrak[key_station]
    coords = {"station": stations}
    nstations = len(stations)
    fig, axes = plt.subplots(nrows=3,ncols=nstations, figsize=(35,20))
    var_names=["xi_i", "mu_i", "sigma_i"]
    if supp_GEV is True:
        var_names = ["xi_i", "mu_i", "sigma_i", "supp_GEV"]
    location = geographical_spread_Skagerrak[key_location]
    ref_params_df = Baseline_geographical_spread_GEV_Params(train, Baseline_region_BM, geographical_spread_Skagerrak, supp_GEV=supp_GEV)
    for row, var in enumerate(var_names):
        ref_var_mlm = list(np.round(ref_params_df[var].to_numpy(dtype="float"), 2))
        az.plot_posterior(idataH, var_names=var, coords=coords, hdi_prob=hdi_prob, group= group, ref_val=ref_var_mlm, ax=axes[row,:],**{"textsize": 20} );
        for column in range(nstations):
            axes[row,column].tick_params(axis="x", labelsize=18)
            #axes[row, column].axes.labelsize
    fig.suptitle(f"The 95%HPI posterior plot of the three GEV params for {model_name} over the subset {location}", fontsize=25)
    fig.show();


def BHM_GEV_param_LOO_posterior_plot(out_of_sample_comp_dict,Loo_column, region_BM, region_sites, hdi_prob=0.95, supp_GEV=False, round_to=2, figsize=(14,14), show=True):
    Loo_station_name = region_sites[Loo_column]
    shape_Loo, loc_Loo, scale_Loo = gev.fit(region_BM[Loo_column].dropna().to_numpy(dtype="float"))
    ref_param_mlm_Loo = [-1*shape_Loo, loc_Loo, scale_Loo]
    model_names = list(out_of_sample_comp_dict.keys())
    var_names=["c_xi_i", "c_mu_i", "c_sigma_i"]
    if supp_GEV is True:
        var_names = ["c_xi_i", "c_mu_i", "c_sigma_i", "c_supp_GEV"]
        ref_param_mlm_Loo.append(loc_Loo + scale_Loo/np.abs(shape_Loo))
    ref_var_mlm =  list(np.round(ref_param_mlm_Loo, round_to))
    nvars = len(var_names)
    fig, axes = plt.subplots(nrows=2,ncols=nvars, figsize=figsize)
    for row, model_name in enumerate(model_names):
        out_of_sample = out_of_sample_comp_dict[model_name]
        az.plot_posterior(out_of_sample, var_names=var_names,hdi_prob=hdi_prob, group="predictions",ref_val=ref_var_mlm, ax=axes[row,:]);
        axes[row,0].set_ylabel(model_names[row], fontsize=20)
        for column in range(nvars):
            axes[row,column].tick_params(axis="x", labelsize=20)
    fig.suptitle(f"The 95%HPI posterior plot of the three GEV params for {model_names[0]} and {model_names[1]} at the LOO station {Loo_station_name }", fontsize=25)
    if show is False:
        return fig, axes
    fig.show();

def BHM_supp_GEV_KDE_plot(idata, group, subcollection_dict, train, region_BM, model_name, region_sites,Baseline_region_BM, key_station="station", key_region="location", 
                          cumulative=False, hdi_probs=0.95,qtiles=[0.75,0.9,0.95,0.99],rug=True, MLM_supp_GEVs=True, figsize=(20, 15), show=True):
    colors = {"Common":"C2","Hilbert":"C0", "Latent":"C1", "Separate":"C3"}
    color_model = colors[model_name]
    train_sites = subcollection_dict[key_station]
    locations = subcollection_dict[key_region]
    supp_GEV = BHM_supp_GEV(idata, group, train_sites)
    sites_coords = [list(train.columns)[site] for site in train_sites]
    region_site_idx_list = [list(region_BM.columns).index(site_coord) for site_coord in sites_coords]
    if MLM_supp_GEVs is True:
        Baseline_region_BM = Baseline_region_BM.loc[sites_coords]
        Baseline_region_BM = Baseline_region_BM.drop(Baseline_region_BM[Baseline_region_BM.xi_i < -1.0000].index)
        MLM_xi_param = Baseline_region_BM["xi_i"]
        MLM_supp_GEV = Baseline_region_BM["supp_GEV"]
        MLM_supp_GEV_sites = list(MLM_supp_GEV.index)
    rows = 2
    if len(train_sites) > 3:
        rows = math.ceil((len(train_sites))/3)
    fig, ax = plt.subplots(nrows=rows,ncols=3, figsize=figsize, dpi=96)
    columns_count = 0
    rows_count = 0
    for i, site_coord in enumerate(sites_coords):
        if columns_count == 3:
            columns_count = 0
            rows_count += 1
        supp_GEV_i = supp_GEV[i]
        az.plot_kde(supp_GEV_i,cumulative=cumulative, rug=rug,quantiles=qtiles,hdi_probs=hdi_probs,plot_kwargs={"color":color_model},
                                                    ax=ax[rows_count, columns_count], legend=True);
        MLE_xi_not_consistent = "no consistent MLE"
        if  MLM_supp_GEVs is True and site_coord in MLM_supp_GEV_sites:
            MLM_supp_GEV_i = MLM_supp_GEV.loc[[site_coord]].to_numpy(dtype="float") 
            MLM_xi_param_i = np.round(Baseline_region_BM["xi_i"].loc[[site_coord]].to_numpy(dtype="float"), 4)[0] 
            ax[rows_count, columns_count].axvline(x=MLM_supp_GEV_i, color='black', linestyle= (5, (10, 3)), lw=2.5, label=f"MLE supp(GEV) shape={MLM_xi_param_i}");
            MLE_xi_not_consistent = "consistent MLE"
        ax[rows_count, columns_count].legend(loc='upper left', fontsize=15)
        ax[rows_count, columns_count].set_title(f"Station {region_sites[site_coord]} ({train_sites[i]}) with {MLE_xi_not_consistent} for the GEV params", fontdict = {'fontsize':10} );
        columns_count += 1
    fig.suptitle(f"{model_name} KDE plots for the support of the GEV, supp(GEV), over the station in the subset: {locations}", fontsize=20);
    for axes in ax.flatten():
        if not axes.has_data():
            fig.delaxes(axes)
    if show is False:
        return fig,ax
    fig.tight_layout()
    fig.show();

def LOO_pair_plot(out_of_sample_L, model_name,region_sites, Loo_column, var_names=["c_xi_i", "c_mu_i", "c_sigma_i"], colors="blue", figsize=(10,10)):
    if colors.lower() == "blue":
        contourf_colors= {"cmap": "Blues"}
        marginal_colors={"color":"C0"}
    else:
        contourf_colors= {"cmap": "Oranges"}
        marginal_colors={"color":"C1"}
    az.plot_pair(
    out_of_sample_L,group="predictions",
    var_names=var_names,
     kind="kde",
               kde_kwargs={
        "hdi_probs": [0.5, 0.75,0.90, 0.95,0.99],  # Plot 30%,A 60% and 90% HDI contours
        "contourf_kwargs": contourf_colors},
        
    divergences=True,
    textsize=25,
    #coords = test_LOO_idx_coords,
    marginals=True,
    marginal_kwargs= marginal_colors,
    figsize = figsize);
    plt.suptitle(f"Pair plots for {model_name} using out-of-sample inference at the LOO station: {region_sites[Loo_column]}", fontsize=20)
    plt.show()
