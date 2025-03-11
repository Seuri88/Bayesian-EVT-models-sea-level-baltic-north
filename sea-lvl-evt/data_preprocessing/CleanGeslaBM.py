import pandas as pd
import xarray as xr
import numpy as np
import warnings
from statsmodels.tsa.tsatools import detrend
from os import listdir
from os.path import isfile, join
from LoadingGesla import GeslaDataset
from datetime import datetime
from Cleaning import all_clean


#Module to load, clean (and detrend) GESLA data
#We make use of Pandas, the objective is to obtain a Block Maxima dataframe (matrix), where the indices corresponds 
#to years, with a year being defined as (staring from 1-July-YYYY and ending 31-June-YYY(Y+1)), 
#The columns are expressed in (lat,lon) coordinates, allowing for spatial dependency in models

#The GESLA data file contanins a metafile (csv) and a folder with tidle gauge station data
metafile = "/home/sm_seuba/Documents/GESLA3_ALL_2.csv"  #Change to yours
datapath = "/home/sm_seuba/Documents/GESLA3 DATA SWE/"  #Prepared so that it only containing Swedish tidal gauge stations 

#Function that takes in datapath and returns a list of all the filenames in the path
def swe_data(datapath): return [f for f in listdir(datapath) if isfile(join(datapath, f))]

#Function that takes in a list of swedish 
def sealvl_cleandata_and_lat_lon(sub_swe_sealvl, g3):
    '''Function that takes in a list of sealvl filenames and returns a dictionary
    containing the three first letters of the filename as key, with an added
    integer in case non-unique station name. The values of dict are clean (removed any
    flaged and interpolated values) sealvl pd dataframes, with longitued and latituted
    added to each observation. '''
    x = lambda x: x[0:3] + "_sealvl" #Lambda function generating names for file_to_pandas method
    #Swe_sealvl_dict: {keys=(lat,lon):values=pd.Seires(sea_level data)}
    #Station_names_dict = {keys=(lat,lon):values= "sitename_sealvl" }
    name_of_site = lambda x: x.rsplit("-")[0].capitalize()
    swe_sealvl_dict = {}
    station_names_dict = {}
    for site in sub_swe_sealvl:
        station = x(site)
        station_name = name_of_site(site)
        station, meta = g3.file_to_pandas(site)
        station = all_clean(station)
        station = station.astype("float64")
        coord = meta["latitude"], meta["longitude"]
        swe_sealvl_dict.update({coord:station})
        #station_names_dict.update({coord:x(site)})
        station_names_dict.update({coord:station_name})
    return swe_sealvl_dict, station_names_dict

def detrend_clean_sealvl_latlon(swe_sealvl_dict, trend=None, window=False, center=False):
    '''Function that creates a detrended(polynomial) pandas Series, for each of the 
    key,value with respect to the swe_sealvl_dict'''
    clean_detrend_sealvl = {}
    if window is True:
        window_size = trend*8760
        for site in swe_sealvl_dict:
            #detrended = swe_sealvl_dict[site].rolling(window_size, center=center).mean()
            window_MA_swe_sealvl_site = swe_sealvl_dict[site].rolling(window_size, center=center).mean()
            if center is False:
                window_MA_swe_sealvl_site.interpolate(method='linear' , inplace=True, limit_direction="backward")
            else:
                window_MA_swe_sealvl_site.interpolate(method='linear' , inplace=True, limit_direction="both")
            detrended = swe_sealvl_dict[site] - window_MA_swe_sealvl_site
            clean_detrend_sealvl.update({site:detrended})
    else:
        for site in swe_sealvl_dict:
            clean_detrend_sealvl.update({site:detrend(swe_sealvl_dict[site], order=trend)})
    return clean_detrend_sealvl   

#Function that produces a Block Maxima of annunal sea-level maxiam at each of the gauged sites
def year_maxima(sub_sealvl_dict, start=1850, end=2020, max_info=False, dropp_NaN_rows=True, mask_neg_values=True):
    '''Function that takes a sealvl dict, return a pandas dataframe with site coordinates as columns, 
        index as years starting at 1885 by default, ends at 2022 by deafult.
        The cells of the columns are the annual maximum sea level, where any given year starts at yyyy-07-01
        and ends in yyy(y+1)-06-30. 


        sub_sealvl_dict: type dict,
        start: type int default=1850,
        end: type int default=2022,

        if out of bin error, change start date to 1850!
        '''
    xy = list(sub_sealvl_dict.keys())
    obs_years = [str(i) for i in range(start-1, end)]
    starts = str(start) + "-07-01-00-00-00"
    ends = str(end) + "-06-30-23-59-00"
    pidx = pd.period_range(starts, ends, freq='min')
    idx = pidx.astype('datetime64[ns]')
    Ys = pd.DataFrame(columns=xy, index=obs_years)
    location = 0
    max_infos = {}
    for site in xy:
        #We the maximum at each site to be collected over uniformed defiened year
        #To do this we need to add missing values to obtain dataframe with uniform shape
        #From there we select max over the subsets of 8760 observation  
        sea_lvl_ = sub_sealvl_dict[site].reindex(idx)
        ys_max_n = sea_lvl_.groupby(pd.Grouper(freq='525600 min', origin='start')).max()
        if max_info is True:
            max_infos.update({site:sea_lvl_.groupby(pd.Grouper(freq='525600 min', origin='start')).idxmax()})
        years = len(ys_max_n.index.year)
        Ys.iloc[0:years, location] = ys_max_n.iloc[:,0].to_numpy(dtype="float", na_value=np.nan)
        location += 1
    Ys.rename_axis("year")
    if mask_neg_values is True:
        Ys.mask(Ys<0, np.nan, inplace=True)
    if max_info is True:
        if dropp_NaN_rows is True:
            Ys = Ys.dropna(how="all")
        return Ys, max_infos
    else:
        if dropp_NaN_rows is True:
            Ys = Ys.dropna(how="all")
        return Ys

def BlockMaximaRawTsData(metafile, datapath, detrend=None, window=False, center=False):
    '''Function that takes in a metafile and datapath, and returns a dictionary where keys are lat,lon coordinates
    and values are the raw time series tidal gauge data represented as a pandas Series.
    
    detrend: type int, default None
        If detrend is None, then return only the raw non-detrended data, else return the trend specific data.
    window: type boolen, deafult False
        For detrened is not None, if window is False, then data is detrened using a polynomial trend of
        degree=detrend, else detrending using a detrend-moving average window.
    center: type boolean, deafult False
        For detrened is not None and window is False, return a non-centered detrend-window moving average detrened
        data, else return centered detrend-window moving average detrended data.
    both: type boolen, default True
        For detrend is not None, if both is False, then return the trend specific data, else return both the
        detrended and non-detrended data
    '''
    g3 = GeslaDataset(metafile, datapath)
    sub_sealvl = swe_data(datapath)
    sealvl_dict, _ = sealvl_cleandata_and_lat_lon(sub_sealvl, g3)
    if detrend is not None:
        sealvl_dict = detrend_clean_sealvl_latlon(sealvl_dict, trend=detrend, window=window, center=center)
    return sealvl_dict

def BlockMaxima(metafile, datapath, start=1850, end=2022, detrend=None, window=False, center=False,max_info=False, dropp_NaN_rows=True, mask_neg_values=True):
    '''''
    
    Function that takes in a list of sea level filenames, corresponding to GESLA class instance, which allow for depicting the sea 
    level time series files using Pandas data frames, and returns the Block Maxima, annual sea level maxima, for each of the sites in the list, 
    as a pandas dataframe, where a year is 356 days starting at July 1 for some given starting year, along with a dictionary of site names and
    coordinates. The returned data frames is index by years, columnized by tuples of lat,lon coordinates, for each of the sites.
    
    If detrend is None, then the Block Maxima is obtianed for non detrened data, else the Blocka Maxima is obtained from
    detrended data. If detrend is note None, and window False, then the detrending using polynomial of degree=detrend. If 
    detrend is note None, and window=True, then the Block Maxima is obtained from the detrend-year rolling non-centred moving averages data, where
    center=True implies centred moving average. 
    
    If info is True, then return a dict where site is key and value is a pd.DataFrame where the column is the timestamp for each max is 
    in the block maxima, using idxmax()
    
    If detrend is True, and both=False, then only detrend Block Maxima and dict is returned. If detrend is True, and both=True, then
    both detrend, non-detrended and dict are returend. 

    
    sub_sealvl: type list, 
    start: type int default 1882,
    end: type inte default 2022,
    detrend: type int, default None,
    both: type boolen, default True,
    window: type boolen, deafult False,
    center: type boolean, deafult False,
    dropna_all_row: type boolean, deafult True
    mask_neg_values: type boolean, default True
    
    
    '''
    g3 = GeslaDataset(metafile, datapath)
    sub_sealvl = swe_data(datapath)
    sealvl_dict, station_names_dict = sealvl_cleandata_and_lat_lon(sub_sealvl, g3)
    if detrend is not None:
        sealvl_dict = detrend_clean_sealvl_latlon(sealvl_dict, trend=detrend, window=window, center=center)
    if max_info is False:
        return year_maxima(sealvl_dict, start, end, max_info=max_info, dropp_NaN_rows=dropp_NaN_rows, mask_neg_values=mask_neg_values), station_names_dict
    else:
        Ys, Ys_max_info = year_maxima(sealvl_dict, start, end, max_info=max_info, dropp_NaN_rows=dropp_NaN_rows)
        return Ys, Ys_max_info, station_names_dict





#sealvl_BM, sealevl_sites = BlockMaxima(swe_sealvl_data)


