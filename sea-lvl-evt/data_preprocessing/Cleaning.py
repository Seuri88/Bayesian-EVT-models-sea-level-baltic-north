import pandas as pd
#Functions to clean the data before EVA analysis
#The three first functions are called in the last function (all_clean)
def drop_cells(df, clm_str, value):
    '''Function that takes in a pandas dataframe, column name and a value,
    then drops any cell in the column with the specific value. 
    df : pandas.Dataframe
    clm_str : str 
    value : int'''
    drop_index = df.loc[df[clm_str] == value].index
    df.drop(drop_index, inplace = True)
def zero_intrpl_drop(df):
    '''Function returns frame with no 0 or interpolate points
    df: pandas.DataFrame ''' 
    #return drop_zero_flags(df), drop_interpolation(df)
    return drop_cells(df,"use_flag", 0) , drop_cells(df,"qc_flag", 2)
def no_flags(df):
    '''Function that returns a new dataframe with only data-time index and sea_lvl
    df: pandas.DataFrame '''
    df_red = df.iloc[:,0:1] 
    return df_red
def all_clean(df):
    '''Function that returns a dataframe contaning only value valid for analysis, and no 
    interpolated data, dropping the flags columns, and returning a new fram with only
    sea_level as a column and date-time as an index. 
    df: pandas.DataFrame'''
    zero_intrpl_drop(df)
    return no_flags(df)
