
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import numpy.ma as ma
from DataAnalysisUtil import (lat_lon_in_BM, BHM_idata_mean_df, BHM_idata_std_df, BHM_idata_hpdi_df, BHM_models_var_name_stacked_dict, BHM_interpolate_train_and_test_latlon_value, Baseline_GEV_params_or_supremum_of_suppGEV_by_bootsraping_MLE, 
                              Baseline_GEV_r_year_return_lvl_bootsraping_MLE, Baseline_GEV_param_resampeling, LOO_interpolation,Baseline_GEV_params_or_supremum_of_suppGEV_by_bootsraping_MLE )
from DataAnalysisReturnlvlsPlt import (BHM_hpdi_and_MLM_ci_return_lvl_Loo_plot, BHM_hpdi_and_MLM_ci_return_lvl_site_compare_plots, 
                                       BHM_hpdi_and_MLM_ci_return_lvl_Loo_compare_plot, BHM_hdpi_and_MLM_ci_return_lvl_multiple_sites_plot, BHM_hpdi_and_MLM_ci_return_lvl_Loo_multiple_models_plot, BHM_hpdi_MSB_LOO_stations_plot)
 
#from DataAnalysisOldFunctions import (BHM_compare_return_lvl_qtlie_site_plot, 
#                                       BHM_compare_return_lvl_hpdi_site_plot, BHM_return_lvl_qtils_site_plot,BHM_return_lvl_hpdi_site_plots)
from DataAnalysisArvizPlt import BHM_subset_denisty_comparsion_plot, BHM_subset_pair_plot, BHM_GEV_param_LOO_posterior_plot, BHM_supp_GEV_KDE_plot, BHM_geographical_spread_GEV_param_posterior_plot, LOO_pair_plot
from HilbertSpaceLatLon import calculate_Kapprox, calculate_K
import pandas as pd
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)
import plotly.graph_objects as go
from IPython.display import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
plt.style.use('tableau-colorblind10')



def BHM_cov_heat_map(cov_var, XY_train, title, show=True):
    m = plt.imshow(cov_var(XY_train).eval(), cmap="inferno", interpolation="none")
    plt.colorbar(m)
    plt.title(title)
    if show is True:
        return plt.show()
    

def plot_MLM_Gev_Baseline(Baseline_data_train, Baseline_data_train1,Baseline_swe_BM, figsize=(10,10), show=True):
    '''
    Function that takes in three baseline data frames and plots the KDE of the three GEV paramaters

    figsize: type tuple default is (10,10), 
    show: type boolean default is True

    '''
    fig, axes = plt.subplots(nrows=3, figsize=figsize);
    for i in range(3):
        Baseline_data_train.iloc[:,i].plot(kind="kde", ax=axes[i], legend=True)
        Baseline_data_train1.iloc[:,i].plot(kind="kde", ax=axes[i],legend=True)
        Baseline_swe_BM.iloc[:,i].plot(kind="kde", ax=axes[i],legend=True)
    if show is True:
        return plt.show()
    else:
        return fig, axes

def plot_ts_data_and_max_values(swe_cmems_20_BM, swe_cmems_20_BM_maxinfo, pre_dt_swe_cmems_BM, train_cmems_sites=None, show=False, figsize=(80, 80)):
    '''
    Function that takes in a BlockMaxima dataframe, BlocMaxima_info dict, time serires list of tuples 
    corresponding to the columns of the BlockMaxima dataframe and returns a matplotlib subplot object
    consisting of a line plot combined with a scatterplot. The line plot corresponds to the time series data
    that generates 
    
    train_cmems_sites: type list default None, 
    show: type boolean default False
    
    '''
    if train_cmems_sites is None:
        rows = swe_cmems_20_BM.shape[1]
        train_cmems_sites = list(swe_cmems_20_BM.columns)
    else:
        rows = len(train_cmems_sites)
    fig, axes = plt.subplots(nrows=rows, ncols=1, figsize=figsize);
    i = 0
    for site in train_cmems_sites:
        Y,X=swe_cmems_20_BM[site].dropna(),swe_cmems_20_BM_maxinfo[site].dropna() 
        axes[i].plot(pre_dt_swe_cmems_BM[site]);
        axes[i].scatter(x=X, y=Y, color="red");
        for j in range(Y.shape[0]): 
            axes[i].annotate(X.iloc[j,0], (X.iloc[j,0], Y.iloc[j,])) 
        i +=1
    fig.suptitle("Site based time series data 20-MA plots along with max values for the translated calender year" , fontsize=30)
    if show is True:
        return plt.show()
    else:
        return fig, axes
    
def Baseline_GEV_param_resampeling_KDE_plots(data_train,resampeling=1000, Baseline_region_BM=None, resample_test=False, mean=True, median=True,figsize=(30,20), show=True):
    '''
    Function that plots the KDE, for each of the three GEV parameters obtained using of the reseampled training data. The model allows for comparing the
    each of the KDE plots, with the Baseline model over the region BM or the data_train, along iwth mean and meadian for each of the three parameters

    data_train: type Pandas.Dataframe Block Maxima 

    resampeling:type int, default is resampeling=1000. The number of resampleing done for each station in the data_train (Block Maxima) dataframe.

    Baseline_region_BM:type Pandas.DatFrame Baseline_models, default is Baseline_region_BM=None. If Baseline_region_BM is not None, then the model plots
    the KDE for each of the three GEV parameters in the Baseline_model dataframe using no resampleing. 

    resample_test: type boolean, defualt is resample_test=False. If resample_test=False, and Baseline_region_BM is not none, the legend of the plots
    show that comparison is made w.r.t. the Baseline_region_BM model. If resample_test=True, and Baseline_region_BM is not none, then the legend of 
    the plots show that comparison is made w.r.t. the Baseline_data_train model. 

    mean: type boolean, defualt is mean=True. If mean=True, then plot the mean of each the GEV parameters for each of the KDE. If mean=False, then no plot is made
    median: type boolean, defualt is median=True. If median=True, then plot the median of each the GEV parameters for each of the KDE. If median=False, then no plot is made
    figsize: type tuple inte, defualt is figsize=(30,20). The size of the figure 
    show: type boolean, defualt is show=True. If show=False, then return plt.subplots tuple (fig, axes). If show=True, then return plt.show()   
         
    '''
    GEV_MLM_param_train_sites = Baseline_GEV_param_resampeling(data_train, resampeling)
    GEV_param_list = ["xi (shape)", "mu (location)", "sigma (scale)"]
    fig, axes = plt.subplots(nrows=3, figsize=figsize);
    for i in range(3):
        GEV_MLM_param_train_sites.iloc[:,i].plot(kind="kde", ax=axes[i],label=f"The KDE for the {GEV_param_list[i]} param of the GEV w.r.t. the training data", legend=True)
        if mean is True:
            axes[i].axvline(GEV_MLM_param_train_sites.iloc[:,i].mean(), label=f"The mean value for the {GEV_param_list[i]} param of the GEV w.r.t. the training data", linestyle=":")
        if median is True:
            axes[i].axvline( GEV_MLM_param_train_sites.iloc[:,i].median(), label=f"The median valeu for the {GEV_param_list[i]} param of the GEV w.r.t. the training data", linestyle="--")
        if Baseline_region_BM is not None:
            region_base = "non-resampled Basseline model about the BM data over the entire region.",
            if resample_test is True:
                region_base = "non-resampled Basseline model about the BM training data"
            Baseline_region_BM.iloc[:,i].plot(kind="kde", ax=axes[i],label=f"The KDE for the {GEV_param_list[i]} param of the GEV  w.r.t  the {region_base}",legend=True)
            if mean is True:
                axes[i].axvline(Baseline_region_BM.iloc[:,i].mean(),color="C1", label=f"The mean value of the {GEV_param_list[i]} param of the GEV w.r.t the {region_base}", linestyle=":")
            if median is True:
                axes[i].axvline(Baseline_region_BM.iloc[:,i].median(),color="C1", label=f"The median value of {GEV_param_list[i]} param of the GEV w.r.t the {region_base}",  linestyle="--")
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        axes[i].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        axes[i].legend()
        axes[i].set_title(f"The KDE {GEV_param_list[i]} of the GEV w.r.t training data using {resampeling} resampelings")
    fig.suptitle("KDE plots for the resampled MLM estimated GEV parameters.")
    if Baseline_region_BM is not None:
        fig.suptitle("Comparative KDE plots for the resampled and non-resampled MLM estimated GEV parameters.")
    if show is False:
        return fig, axes
    plt.show()


def HSGP_LatLon_Gram_matrix_approx_comparison_plots(chosen_ell, XY_base, variable_name, approx_dict, cov_func_name="matern52", figsize=(14,7),show=True):
    '''Function that takes in a length_scale, XY (lat,lon) cordinates, variable, name, a nested dict, where the keys are columns,
     values are dicts where the keys are rows and values are list of len(2). Contaning number of basis vectors and scaling factor.
      A covariance function name corresponding to the covariance one wants to approximate.  
     '''
    n_columns = len(approx_dict)+1
    fig, axs = plt.subplots(2, n_columns, figsize=figsize, sharey=True)
    K = calculate_K(XY_base,chosen_ell, cov_func_name=cov_func_name)
    axs[0, 0].imshow(K, cmap="inferno", vmin=0, vmax=1)
    axs[0, 0].set(xlabel="x1", ylabel="x2", title=f"True Gram matrix\nTrue $\\ell$ = {chosen_ell}")
    axs[1, 0].axis("off")
    im_kwargs = {
        "cmap": "inferno",
        "vmin": 0,
        "vmax": 1,
        "interpolation": "none",
    }
    for i, columns in enumerate(approx_dict):
        for j, rows in enumerate(approx_dict[columns]):
            m, c = approx_dict[columns][rows][0],approx_dict[columns][rows][1]
            m_star = np.prod(m)
            K_approx = calculate_Kapprox(XY_base, c, m,chosen_ell, cov_func_name=cov_func_name)
            axs[j, i+1].imshow(K_approx, **im_kwargs)
            axs[j, i+1].set_title(f"m = {m}, c = {c}, m* = {m_star}")
    for ax in axs.flatten():
        ax.grid(False)
    fig.suptitle(f'The true Gram maxtrix for cov_{variable_name}:{cov_func_name}Chordal with ls={chosen_ell} and HS approximated ', fontsize=16)
    fig.tight_layout()
    if show is False:
        return fig, axs
    else:
        return plt.show()
    
def Baseline_GEV_param_KDE_comparison_plot(Baseline_data_train, Baseline_region_BM, figsize=(20,15), MLE_consistent=True, CI_consistent=True, show=True):
    '''
    Function that takes in two Baseline dataframs and return for each of the three parameters a two KDE plots,
    one for the Baseline_data_train and one for Baseline_region_BM. For each parameter and each Baseline dataframe,
    the plot also show their respective mean and median. The plot return either a fig, axes tuple or a plot.

    Baseline_data_train: type Pandas.dataframe
    Baseline_region_BM: type Pandas.dataframe
    figsize: type tuple, default figsize=(20,15). Figsize is the size of the figure contaning the three subplots
    show: type boolean, default show=True. If show=False, then the function return fig, axes. If show=True, the
    function return plt.show()
    '''
    fig, axes = plt.subplots(nrows=3, figsize=figsize);
    GEV_param_list = ["shape", "location", "scales"]
    if MLE_consistent is True:
        Baseline_region_BM = Baseline_region_BM.drop(Baseline_region_BM[Baseline_region_BM.xi_i <= -1.0000].index)
        Baseline_data_train = Baseline_data_train.drop(Baseline_data_train[Baseline_data_train.xi_i <= -1.0000].index)
    elif CI_consistent is True:
        Baseline_region_BM = Baseline_region_BM.drop(Baseline_region_BM[Baseline_region_BM.xi_i <= -.5000].index)
        Baseline_data_train = Baseline_data_train.drop(Baseline_data_train[Baseline_data_train.xi_i <= -.5000].index)
    else:
        Baseline_region_BM =Baseline_region_BM
        Baseline_data_train =  Baseline_data_train
    for i in range(3):
        Baseline_data_train.iloc[:,i].plot(kind="kde", ax=axes[i],label=f"The KDE for the {GEV_param_list[i]} of the GEV w.r.t the training data", legend=True)
        Baseline_region_BM.iloc[:,i].plot(kind="kde", ax=axes[i],label=f"The KDE for the {GEV_param_list[i]} of the GEV w.r.t the entire region",legend=True)
        axes[i].axvline(Baseline_data_train.iloc[:,i].median(), label=f"The median value of the {GEV_param_list[i]} w.r.t the training data", linestyle="--")
        axes[i].axvline(Baseline_region_BM.iloc[:,i].median(),color="C1", label=f"The median value of the {GEV_param_list[i]} w.r.t the entire region",  linestyle="--")                                                                                                                                              
        axes[i].axvline(Baseline_data_train.iloc[:,i].mean(), label=f"The mean value of the {GEV_param_list[i]} w.r.t the training data", linestyle=":")
        axes[i].axvline(Baseline_region_BM.iloc[:,i].mean(),color="C1", label=f"The mean value of the {GEV_param_list[i]} w.r.t the entire region", linestyle=":")
        axes[i].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        axes[i].legend()
        axes[i].set_title(f"Comparative KDE plots for the Baseline model estimated GEV parameters, for the training data and the entire region")
    if show is False:
        return fig, axes
    plt.show()
    
def BHM_mean_GeoScatter_train_test_split(region_base_BM, train, test, train_test_split="both", top_N=4, width=1800, height=1000, static=False):
    '''Function that plots the training stations and test stations on a plotly GeoScatter map, the stations are marked with 
     circel-dot, and asterisk for training, respectivley, test stations. The number of the sations as asigned by the BHN (PYMC)
    models is alsow shown for the stations. The stations with the top N most observations can be asigned a diffrent marker (diamond-dot).

    region_base_BM: type Pandas.DataFrame Block Maxima data frame contaning the lat,lon coordinates as columns entire region
    train: type Pandas.DataFrame contaning the lat,lon coordinates as columns only training stations
    test: type Pandas.DataFrame contaning the lat,lon coordinates as columns only test stations
    train_test_split: type str, default is train_test_split="both" {"both", "train", "test" }. If train_test_split is train, then only
    training stations are ploted, analogues for test. If both or invalide input, then both train and test are plotted.
    top_N: type int, default is top_N=4. If top_N=N, then the top N stations (most observations are given a diamond-dot marker )
    width: type, int default is width=1800. This assignes the width of the GeoScatter plot 
    height: type int, default is  height=1000. This assignes the height of the GeoScatter plot 
    static: type boolean, default is static=False. If static=False, then return a interactive plot, suitable for a jupyter notebook session. 
    If static=True, then return a static image, sutiable for batch job. 
           
    '''
    xy_base, xs_base, ys_base = lat_lon_in_BM(region_base_BM)
    markers_train_test = pd.DataFrame(None, columns=xy_base, index=["marker", "site_number"])
    top_N_columns = []
    low_N_columns = []
    if top_N is not None:
        top_N_columns = [LOO_interpolation(region_base_BM, max_noNaNs_idx=i) for i in range(top_N)]
        low_N_columns = [LOO_interpolation(region_base_BM, max_noNaNs_idx=-i) for i in range(1, top_N+1)]
    for xy in xy_base:
        if xy in list(train.columns) and (xy not in top_N_columns) and (xy not in low_N_columns):
            markers_train_test.at["marker", xy] = "circle-dot"
            markers_train_test.at["site_number", xy] = list(train.columns).index(xy)
        elif xy in list(train.columns) and xy in top_N_columns:
            markers_train_test.at["marker", xy] = "diamond-dot"
            markers_train_test.at["site_number", xy] = list(train.columns).index(xy)
        elif xy in list(train.columns) and xy in low_N_columns:
            markers_train_test.at["marker", xy] = "cross-thin"
            markers_train_test.at["site_number", xy] = list(train.columns).index(xy)
        else:
            markers_train_test.at["marker", xy] = "asterisk"
            markers_train_test.at["site_number", xy] = list(test.columns).index(xy) + train.shape[1]
    mode_to_use ='text+markers'
    size_to_use = 10
    if train_test_split == "both":
        text_to_use = markers_train_test.iloc[1,:]
        symbol_to_use =  markers_train_test.iloc[0,:]
        lon_ = xs_base
        lat_ = ys_base
        mode_to_use ='markers'
        size_to_use =10
    elif train_test_split == "train":
        xy_train, xs_train, ys_train = lat_lon_in_BM(train)
        text_to_use = markers_train_test[xy_train].iloc[1,:]
        symbol_to_use =  markers_train_test[xy_train].iloc[0,:]
        lon_ = xs_train
        lat_ = ys_train
    elif train_test_split == "test":
        xy_test, xs_test, ys_test = lat_lon_in_BM(test)
        text_to_use = markers_train_test[xy_test].iloc[1,:]
        symbol_to_use =  markers_train_test[xy_test].iloc[0,:]
        lon_ = xs_test
        lat_ = ys_test
    else:
        text_to_use = markers_train_test.iloc[1,:]
        symbol_to_use =  markers_train_test.iloc[0,:]
        lon_ = xs_base
        lat_ = ys_base
        mode_to_use ='markers'
        size_to_use =10
    fig = go.Figure(data=go.Scattergeo(
            locationmode = 'country names',
            lon = lon_,
            lat = lat_,
            text = text_to_use,
            textposition = "bottom right",
            mode = mode_to_use,
            marker = dict(
                size = size_to_use,
                opacity = 1,
                reversescale = True,
                autocolorscale = False,
                symbol = symbol_to_use,
                line = dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
            ) ))
    fig.update_layout(
            title = f"The test and train sites data",
            geo = dict(
                scope='europe',
                projection_type='albers',
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 5,
                subunitwidth = 5
            ), width=width, height=height
        )
    fig.update_layout(
        mapbox = {
            'style': "outdoors", 'zoom': 1.7},
        showlegend = False,)
    if static is True:
        img_bytes = fig.to_image(format="png")
        return Image(img_bytes)      
    else:
        return fig.show()

def BHM_mean_GeoScatter_plt(idata, train, train_sites, std=False, var_name="xi_i", group="post", scatter_size=10, gev_param=None, show=True):
    '''Plotly GeoScatter plot for the BHM inference data. The function allows for ploting both mean oand std of the GEV parameters,
     along with the r-return levels in the model. The hover text shows the coordinates of the model in the first row, and the 
      value along with site numner on the second row ''' 
    xy, xs_, ys_ = lat_lon_in_BM(train)
    if std is False:  
        df_idata = BHM_idata_mean_df(idata=idata, train_sites=train_sites, group=group)
        param_stat = "mean"
    else: 
        df_idata = BHM_idata_std_df(idata,train_sites=train_sites, group=group)
        param_stat = "std"
    df_idata.index = xy
  
    
    color_dict = {"shape":"Blues", "loc": "Greens", "scale":"Reds"}
    color_keys = list(color_dict.keys())
    if gev_param in color_keys:
        color_to_use = color_dict[gev_param]
    else:
        color_to_use = "Oranges"   
    fig = go.Figure(data=go.Scattergeo(
            locationmode = 'country names',
            lon = xs_,
            lat = ys_,
            text = df_idata[[var_name, "sites"]],
            mode = 'markers',
            marker = dict(
                size = scatter_size,
                opacity = 1,
                reversescale = True,
                autocolorscale = False,
                symbol = 'circle',
                line = dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
                colorscale =  color_to_use,
                cmin =  df_idata[var_name].min(),
                color =  df_idata[var_name],
                cmax = df_idata[var_name].max(),
                colorbar_title= f"The {param_stat} for {var_name} "
            )))

    fig.update_layout(
            title = f"The {param_stat} for {var_name} using the {group} data",
            geo = dict(
                scope='europe',
                projection_type='albers',
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 0.5,
                subunitwidth = 0.5
            ),
        )
    fig.show()

def BHM_interpolated_train_test_var_name_surface_mean_or_std(idata, out_of_sample, XY_train, XY_test_longer_array, X, Y, var_name, model_name, mean_or_std="mean",hdi_prob=0.95, figsize=(18,18)):
    Z = BHM_interpolate_train_and_test_latlon_value(idata, out_of_sample, XY_train, XY_test_longer_array, X, Y, var_name, mean_or_std=mean_or_std,hdi_prob=hdi_prob)
    fig = plt.figure(figsize=figsize)
    axs = plt.axes(projection=ccrs.PlateCarree())
    axs.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    if mean_or_std.lower() != "mean" and mean_or_std.lower() != "std":
        cmap="magma"
    elif mean_or_std.lower() == "mean":
        cmap="viridis"
    else:
        cmap="plasma"
    plot = plt.pcolormesh(Y, X, Z,  shading='gouraud',cmap=cmap, vmin=Z.min(), vmax=Z.max(),offset_transform=ccrs.PlateCarree(central_longitude=180))
    plt.plot(XY_train[:,1], XY_train[:,0], "ok", marker="o", color="red",  markeredgecolor="black", markersize=12, label="sites point")
    axs.coastlines(resolution="10m")
    axs.add_feature(cfeature.BORDERS)
    axs.set_xticks(Y.flatten(), crs=ccrs.PlateCarree())
    axs.set_yticks(X.flatten(), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    plt.colorbar(plot,  orientation='horizontal')
    axs.axis("equal")
    axs.xaxis.set_major_formatter(lon_formatter)
    axs.yaxis.set_major_formatter(lat_formatter)
    axs.set_title(f"The {var_name} surface ({mean_or_std} values) for {model_name} using both point from the idata prediction (observed) and out of sample prediction")
    fig.show();

def BHM_GeoScatter_train_and_new_sites_plot(train, XY_test_longer, width=1800, height=1000, static=False):
    train_columns = list(train.columns)
    xy_train_and_test_longer = train_columns + XY_test_longer
    markers_train_test = pd.DataFrame(None, columns=xy_train_and_test_longer, index=["marker", "site_number"])
    xy_base, xs_base, ys_base = lat_lon_in_BM(markers_train_test)
    for xy in xy_base:
        if xy in XY_test_longer:
            markers_train_test.at["marker", xy] = "hash"
            markers_train_test.at["site_number", xy] = XY_test_longer.index(xy) + train.shape[1]
        else:
            markers_train_test.at["marker", xy] = "circle-dot"
            markers_train_test.at["site_number", xy] = train_columns.index(xy) + train.shape[1]
    text_to_use = markers_train_test.iloc[1,:]
    symbol_to_use =  markers_train_test.iloc[0,:]
    fig = go.Figure(data=go.Scattergeo(
            locationmode = 'country names',
            lon = xs_base,
            lat = ys_base,
            text = text_to_use,
            textposition = "bottom right",
            mode = "markers",
            marker = dict(
                size = 10,
                opacity = 1,
                reversescale = True,
                autocolorscale = False,
                symbol = symbol_to_use,
                line = dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
            ) ))
    fig.update_layout(
            title = f"The test and train sites data",
            geo = dict(
                scope='europe',
                projection_type='albers',
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 5,
                subunitwidth = 5
            ), width=width, height=height
        )
    fig.update_layout(
        mapbox = {
            'style': "outdoors", 'zoom': 1.7},
        showlegend = False,)
    if static is True:
        img_bytes = fig.to_image(format="png")
        return Image(img_bytes)      
    else:
        return fig.show()


def BHM_violinplot_models_comparison_old(models_names,models_idata,train_sites,var_name,showmeans=False, showextrema=False, showmedians=True,points=4000, widths=0.35, figsize=(55, 20),show=True):
    '''
     '''
    var_name_all_models_dict =BHM_models_var_name_stacked_dict(models_names, models_idata, var_name, train_sites)
    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
        )
    ax.set_ylabel("")
    colors = [f"C{i}" for i in range(len(models_names))]
    start, stop = -1*float( (len(models_names)-1 )/10), float((len(models_names))/10)
    posistion_range = np.arange(start,stop, step=0.2)
    for site in train_sites:
        i_mask = site
        models_var_name_at_site = []
        for models in models_names:
            models_var_name_at_site.append(var_name_all_models_dict[models][site])

        vp = ax.violinplot(dataset=models_var_name_at_site, positions=[i_mask + j for j in posistion_range],widths=widths,
                            showmeans=showmeans, showextrema=showextrema, showmedians=showmedians, points=points)
    
        for j in range(len(models_var_name_at_site)):
            vp["bodies"][j].set_facecolor(colors[j])
            if models_var_name_at_site[j].min() == models_var_name_at_site[j].max():
                plt.setp(vp["cmedians"][j], color=colors[j], linewidth=2)
        ax.set_xticks(
                np.arange(len(train_sites)),
                labels=train_sites,
                size=40,
                rotation=25,
            )
        legend_handles = [mpatches.Patch(color=colors[i], label=models_names[i]) for i in range(len(models_names))]

    ax.legend(loc="upper right",
            handles=legend_handles,
            frameon=False,
            bbox_to_anchor=(1, 1),
                fontsize=35
        )
    if var_name[0] == "Z":
        ax.set_title(f"The "+ var_name[3:] + "-year return level")
    else:
        var_name_dict = {"xi_i": "shape", "mu_i": "location", "sigma_i": "scale"}
        ax.set_title(f"The {var_name_dict[var_name]} parameter of the GEV")
    ax.patch.set_visible(False)
            # ax.axis("off")
    ax.tick_params(axis="y", labelsize=40)
    if show is False:
        return fig, ax
    else:
        ax.set_xlim(-0.99, len(train_sites) + 0.99)
        return plt.show();

def BHM_violinplot_models_comparison(models_names,models_idata,train_sites,var_name,train, region_BM,  Baseline_region_BM,gev_dist=False,showmeans=False, showextrema=True, showmedians=True,quantiles=False,ci_alpha=0.95,points=4000,n_draws=10000, widths=0.35, figsize=(55, 20),show=True):
    '''
     '''
    var_name_all_models_dict =BHM_models_var_name_stacked_dict(models_names, models_idata, var_name, train_sites,gev_dist)
    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
        )
    ax.set_ylabel("")
    models_names_new = models_names.copy()
    models_names_new.insert(0,"Baseline resampled")
    n_models_names = len(models_names_new)
    colors = [f"C{i}" for i in range(n_models_names)]
    start, stop = -1*float( (n_models_names-1 )/10), float((n_models_names)/10)
    posistion_range = np.arange(start,stop, step=0.2)
    train_columns = list(train.columns)
    sites_coords = [train_columns[site] for site in train_sites]
    region_stations = list(region_BM.columns)
    region_site_idx_list = [region_stations.index(site_coord) for site_coord in sites_coords]
    baseline_gev_params =  Baseline_GEV_params_or_supremum_of_suppGEV_by_bootsraping_MLE(region_BM, region_site_idx_list, n_draws=n_draws)
    baseline_var_names_to_gev_param = {"xi_i": 0, "mu_i": 1, "sigma_i":2}
    param_idx = baseline_var_names_to_gev_param[var_name]
    for i, site in enumerate(train_sites):
        i_mask = site
        models_var_name_at_site = []
        if Baseline_region_BM.iloc[i,0] > -0.50:
            baseline_gev_params_ref_MLM = Baseline_region_BM.iloc[i,param_idx]
        else:
            baseline_gev_params_ref_MLM = None
        for j, models in enumerate(models_names_new):
            if j == 0:
                param =  baseline_gev_params[region_site_idx_list[i]][param_idx]
                ci_min, ci_max = np.quantile(a=param, q=[(1 - ci_alpha) / 2, (1 + ci_alpha) / 2], axis=0)
                param_ci_bounded = ma.masked_outside(param, ci_min, ci_max)
                models_var_name_at_site.append(param_ci_bounded)
            else:
                idata_model = models_idata[j-1]
                df_idata_hpdi = BHM_idata_hpdi_df(idata=idata_model, hdi_prob=ci_alpha, group="post", return_lvl=False, msb_var_names=var_name).T[site]
                lower_hpdi, higher_hpdi = df_idata_hpdi["lower"].to_numpy(dtype="float"), df_idata_hpdi["higher"].to_numpy(dtype="float")
                param = var_name_all_models_dict[models][site]
                param_hpdi_bounded = ma.masked_outside(param, lower_hpdi, higher_hpdi)
                models_var_name_at_site.append(param_hpdi_bounded)
        qtiles = None
        if quantiles is True:
            qtiles = [0.05, 0.1, 0.9, 0.95]
        vp = ax.violinplot(dataset=models_var_name_at_site, positions=[i_mask + j for j in posistion_range],widths=widths,
                            showmeans=showmeans, showextrema=showextrema, showmedians=showmedians, quantiles=qtiles, points=points)
        if baseline_gev_params_ref_MLM is not None:
            ax.scatter(site, baseline_gev_params_ref_MLM, marker='x', color='black', s=200, zorder=3)
        for j in range(len(models_var_name_at_site)):
            vp["bodies"][j].set_facecolor(colors[j])
            vp["bodies"][j].set_edgecolor(colors[j])
            #if showmedians is True:
            #    vp["cmedians"][j].set_color("w")
            #if showextrema is True:
            #    vp["cmax"][j].set_color(colors[j])
            #    vp["cmin"][j].set_color(colors[j])
            if models_var_name_at_site[j].min() == models_var_name_at_site[j].max():
                plt.setp(vp["cmedians"][j], color=colors[j], linewidth=2)
        ax.set_xticks(
                np.arange(len(train_sites)),
                labels=train_sites,
                size=40,
                rotation=25,
            )
        legend_handles = [mpatches.Patch(color=colors[i], label=models_names_new[i]) for i in range(n_models_names)]
        if baseline_gev_params_ref_MLM is not None:
            legend_handles.append(mpatches.Patch(color='black', label="Baseline nonresampled"))
    ax.legend(loc="upper right",
            handles=legend_handles,
            frameon=False,
            bbox_to_anchor=(1, 1),
                fontsize=35
        )
    if var_name[0] == "Z":
        fig.suptitle(f"The "+ var_name[3:] + "-year return level", fontsize=30)
    elif gev_dist is True:
        fig.suptitle("The posterior_predictive dist for the GEV",  fontsize=30)
    else:
        var_name_dict = {"xi_i": "shape", "mu_i": "location", "sigma_i": "scale"}
        fig.suptitle(f"The {var_name_dict[var_name]} parameter of the GEV",  fontsize=35)
    ax.patch.set_visible(False)
            # ax.axis("off")
    ax.tick_params(axis="y", labelsize=40)
    if show is False:
        return fig, ax
    else:
        ax.set_xlim(-0.99, len(train_sites) + 0.99)
        return plt.show();
 


    
def BHM_boxplot_models_comparison(models_names, models_idata, train_sites, var_name, region_BM, train, gev_dist=False, n_draws=10000,whis=[5,95], figsize=(25,10), show=True):
    '''Function that takes in a list of BHM's model_names, a list of BHMs idata wiht posterior, list of sites 
    (coordinates for the BHMs models), a variable name and returns a matplotlib boxplot.
    models_names_new: type list
    models_idata: type list
    train_sites: type list 
    var_name: type str
    whis: type list, default whis=[5,95] the sizes of the whiskers
    figsize: type tuple, default figsize=(25,10)
    show: type boolean, default show=True. If true, then return a plot, if false returns a matplotlib.plotly fig,ax tuple
    return ax.
     '''
    var_name_all_models_dict =BHM_models_var_name_stacked_dict(models_names, models_idata, var_name, train_sites, gev_dist)
    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
        )
    ax.set_ylabel("")
    models_names_new = models_names.copy()
    models_names_new.insert(0,"Baseline resampled")
    n_models_names_new = len(models_names_new)
    colors = [f"C{i}" for i in range( n_models_names_new)]
    start, stop = -1*float(  (n_models_names_new-1 )/10), float(n_models_names_new/10)
    posistion_range = np.arange(start,stop, step=0.2)
    if n_models_names_new == 5:
        posistion_range = np.arange(start,stop, step=0.15)
        posistion_range =posistion_range[1:]
    train_columns = list(train.columns)
    sites_coords = [train_columns[site] for site in train_sites]
    region_stations = list(region_BM.columns)
    region_site_idx_list = [region_stations.index(site_coord) for site_coord in sites_coords]
    return_periods = int(var_name.split("_")[-1])
    baseline_r_period_return_lvl = Baseline_GEV_r_year_return_lvl_bootsraping_MLE(region_BM, region_site_idx_list, return_periods, n_draws=n_draws)
    for i, site in enumerate(train_sites):
        i_mask = site
        models_var_name_at_site = []
        for j, models in enumerate(models_names_new):
            if j == 0:
                models_var_name_at_site.append(baseline_r_period_return_lvl[region_site_idx_list[i]])
            else:
                models_var_name_at_site.append(var_name_all_models_dict[models][site])

        bp = ax.boxplot(
                    models_var_name_at_site,
                    patch_artist=True,
                    showfliers=False,
                    positions=[  # x_pos[i_mask]                                                                                                                                                                                                                  
                        i_mask + j for j in posistion_range],
                    whis=whis,
                    manage_ticks=True)
        for med in bp["medians"]:
            plt.setp(med, color="w", linewidth=2)

        for j in range(len(models_var_name_at_site)):
            bp["boxes"][j].set_facecolor(colors[j])
            if models_var_name_at_site[j].min() == models_var_name_at_site[j].max():
                plt.setp(bp["medians"][j], color=colors[j], linewidth=2)
        ax.set_xticks(
                np.arange(len(train_sites)),
                labels=train_sites,
                size=24,
                rotation=25,
            )
        legend_handles = [mpatches.Patch(color=colors[i], label=models_names_new[i]) for i in range(len(models_names_new))]

    ax.legend(
                loc="best",
                handles=legend_handles,
                frameon=False,
                bbox_to_anchor=(0, 0),
                fontsize = 18
            )
    
    if var_name[0] == "Z":
        fig.suptitle(f"The "+ var_name[3:] + "-year return level", fontsize=30)
    elif gev_dist is True:
        fig.suptitle("The posterior_predictive dist for the GEV",  fontsize=30)
    else:
        var_name_dict = {"xi_i": "shape", "mu_i": "location", "sigma_i": "scale"}
        fig.suptitle(f"The {var_name_dict[var_name]} parameter of the GEV",  fontsize=30)
    ax.patch.set_visible(False)
            # ax.axis("off")
    ax.tick_params(axis="y", labelsize=30)
    if show is False:
        return fig, ax
    else:
        return plt.show()   
 
def BHM_comparative_mean_and_hpdi_var_name_surface_shared_cbar_plots(models_intrp_in_out_sample_dict, XY_train, XY_test_longer_array, X, Y, var_name, return_period=None,hdi_prob=0.95,figsize=(25,25)):
    model_names = list(models_intrp_in_out_sample_dict.keys())
    idataH, out_of_sample_H = models_intrp_in_out_sample_dict[model_names[0]][0], models_intrp_in_out_sample_dict[model_names[0]][1]
    idataL, out_of_sample_L = models_intrp_in_out_sample_dict[model_names[1]][0], models_intrp_in_out_sample_dict[model_names[1]][1]
    lower_mean_higher_hdi = ["lower_hpdi", "mean", "higher_hpdi"]
    gev_params = {"xi_i":"shape", "mu_i":"location", "sigma_i":"scale"}
    Z_H = [BHM_interpolate_train_and_test_latlon_value(idataH, out_of_sample_H, XY_train, XY_test_longer_array, X, Y, var_name, stat,hdi_prob) for stat in lower_mean_higher_hdi]
    Z_L = [BHM_interpolate_train_and_test_latlon_value(idataL, out_of_sample_L, XY_train, XY_test_longer_array, X, Y, var_name, stat,hdi_prob) for stat in lower_mean_higher_hdi]
    Z_list = [Z_H, Z_L]
    color = "viridis"
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(projection=projection))
    fig  = plt.figure(figsize=figsize)
   
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    #if mean_or_std.lower() != "mean" and mean_or_std.lower() != "std":
    #    cmap="magma"
    #elif mean_or_std.lower() == "mean":
    #    cmap="viridis"
    #else:
    #    cmap="plasma"
    Zvmin = sorted([min((Z_list[0][i].min(),Z_list[1][i].min())) for i in range(3)])[0]
    Zvmax = sorted([max((Z_list[0][i].max(),Z_list[1][i].max())) for i in range(3)])[-1]
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 2),
                    axes_pad=0.8,
                    share_all=True,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.25,
                    cbar_size='3%',
                    label_mode='keep')
    column_count = 0
    for i, axs in enumerate(axgr):
        #axs.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
        if i % 2 == 0:
            Zs = Z_list[0]
            to_plot = column_count % 3
            Z = Zs[to_plot]
        else:
            Zs = Z_list[1]
            to_plot = column_count % 3
            Z = Zs[to_plot]
            column_count +=1
        plot = axs.pcolormesh(Y, X, Z,  shading='gouraud',cmap=color, vmin=Zvmin, vmax=Zvmax,offset_transform=ccrs.PlateCarree(central_longitude=180))
        axs.plot(XY_train[:,1], XY_train[:,0], "ok", marker="o", color="red",  markeredgecolor="black", markersize=8, label="sites point")
        axs.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
        axs.coastlines(resolution="10m")
        axs.add_feature(cfeature.BORDERS)
        axs.set_xticks(Y.flatten(), crs=ccrs.PlateCarree())
        axs.set_yticks(X.flatten(), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        if i == 0 or i == 1:
           axs.set_title(f"{model_names[i]}", fontdict={'fontsize':30})
        #plt.colorbar(plot,  orientation='horizontal')
        #axs.axis("equal")
        axs.xaxis.set_major_formatter(lon_formatter)
        axs.yaxis.set_major_formatter(lat_formatter)
        if i == 5:
            axgr.cbar_axes[0].colorbar(plot)
            axgr.cbar_axes[0].tick_params(labelsize=20)
        axs.tick_params(axis="both", labelsize=18)
    if return_period is not None:
        fig.suptitle(f"Comparative {return_period}-year return level maps for {model_names[0]} and {model_names[1]} ", fontsize=30)
    else:
        fig.suptitle(f"Comparative GEV {gev_params[var_name]} parameter surface plots for {model_names[0]} and {model_names[1]} ", fontsize=30)
    fig.show();

def BHM_comparative_mean_and_hpdi_var_name_surface_plots(models_intrp_in_out_sample_dict, XY_train, XY_test_longer_array, X, Y, var_name, return_period=None,hdi_prob=0.95,figsize=(30,30)):
    model_names = list(models_intrp_in_out_sample_dict.keys())
    idataH, out_of_sample_H = models_intrp_in_out_sample_dict[model_names[0]][0], models_intrp_in_out_sample_dict[model_names[0]][1]
    idataL, out_of_sample_L = models_intrp_in_out_sample_dict[model_names[1]][0], models_intrp_in_out_sample_dict[model_names[1]][1]
    lower_mean_higher_hdi = ["lower_hpdi", "mean", "higher_hpdi"]
    gev_params = {"xi_i":"shape", "mu_i":"location", "sigma_i":"scale"}
    Z_H = [BHM_interpolate_train_and_test_latlon_value(idataH, out_of_sample_H, XY_train, XY_test_longer_array, X, Y, var_name, stat,hdi_prob) for stat in lower_mean_higher_hdi]
    Z_L = [BHM_interpolate_train_and_test_latlon_value(idataL, out_of_sample_L, XY_train, XY_test_longer_array, X, Y, var_name, stat,hdi_prob) for stat in lower_mean_higher_hdi]
    Z_list = [Z_H, Z_L]
    color_list = {0:"plasma",1:"viridis", 2:"plasma"}
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(projection=projection))
    fig  = plt.figure(figsize=figsize)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    Zvmin_list = [min((Z_list[0][i].min(),Z_list[1][i].min())) for i in range(3)]
    Zvmax_list = [max((Z_list[0][i].max(),Z_list[1][i].max())) for i in range(3)]
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 2),
                    axes_pad=1.2,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="edge",
                    cbar_pad=0.55,
                    cbar_size='5%',
                    label_mode='keep')
    column_count = 0
    labels = ["lower", "mean", "higher"]
    for i, axs in enumerate(axgr):
        #axs.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
        if i % 2 == 0:
            Zs = Z_list[0]
            to_plot = column_count % 3
            Z = Zs[to_plot]
        else:
            Zs = Z_list[1]
            to_plot = column_count % 3
            Z = Zs[to_plot]
        plot = axs.pcolormesh(Y, X, Z,  shading='gouraud',cmap=color_list[to_plot], vmin=Zvmin_list[to_plot], vmax=Zvmax_list[to_plot],offset_transform=ccrs.PlateCarree(central_longitude=180))
        axs.plot(XY_train[:,1], XY_train[:,0], "ok", marker="o", color="red",  markeredgecolor="black", markersize=9, label="sites point")
        axs.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
        axs.coastlines(resolution="10m")
        axs.add_feature(cfeature.BORDERS)
        axs.set_xticks(Y.flatten(), crs=ccrs.PlateCarree())
        axs.set_yticks(X.flatten(), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()

        if i == 0 or i == 1:
           axs.set_title(f"{model_names[i]}", fontdict={'fontsize':30})
        #plt.colorbar(plot,  orientation='horizontal')
        #axs.axis("equal")
        axs.xaxis.set_major_formatter(lon_formatter)
        axs.yaxis.set_major_formatter(lat_formatter) 
        if i % 2 == 1:
            axgr.cbar_axes[i//2].colorbar(plot).set_label(labels[to_plot], fontsize=20)
            #axgr.cbar_axes[i//2].set_label(labels[to_plot], fontsize=16)
            axgr.cbar_axes[i//2].tick_params(labelsize=20)
            column_count += 1
        axs.tick_params(axis="both", labelsize=18)
    if return_period is not None:
        fig.suptitle(f"Comparative {return_period}-year return level maps for {model_names[0]} and {model_names[1]} ", fontsize=30)
    else:
        fig.suptitle(f"Comparative GEV {gev_params[var_name]} parameter surface plots for {model_names[0]} and {model_names[1]} ", fontsize=30)
    fig.show();
