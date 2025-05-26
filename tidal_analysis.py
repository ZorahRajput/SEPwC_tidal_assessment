#!/usr/bin/env python3
"""
Created on Mon May  5 14:35:48 2025

@author: zorahrajput
"""
# Import modules
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import wget
import os
import numpy as np
import uptide
import pytz
import math
import glob
from scipy import stats

def read_tidal_data(tidal_file):
    try:
        # Use only relevant rows
        data = pd.read_csv(
            tidal_file,
            header=None,
            skiprows=11,  
            sep=r'\s+',
            usecols=[1, 2, 3],
            names=['DateStr', 'TimeStr', 'SeaLevelRaw'], 
            dtype={'SeaLevelRaw' : str}, 
            )
        
        # Replace invalid data with NaN 
        suffixes_to_remove = ['M', 'N', 'T']
        pattern = r'\s*(' + '|'.join(suffixes_to_remove) + r')$'

        data['Sea Level'] = pd.to_numeric(
            data['SeaLevelRaw'].str.replace(pattern, '', regex=True),
            errors='coerce' 
            )
        # Gemini - Suggestion to remove "bad numbers"
        data['Sea Level'] = data['Sea Level'].replace(-99.0000, np.nan) 

        # Use only relevant columns in the correct format 
        data['DateTime'] = data['DateStr'].str.replace('/', '-') + ' ' + data['TimeStr']
        data = data.rename(columns={'DateStr' : 'Date', 'TimeStr' : 'Time'})
        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y-%m-%d %H:%M:%S')
        data = data.set_index('DateTime')
        data = data.drop(columns=['SeaLevelRaw']) 
        return data
    
    # Handle potential errors
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {tidal_file}")
        
    except pd.errors.ParserError:
        print(f"Error parsing CSV file '{tidal_file}'")
        return pd.DataFrame()

    except KeyError:
        print(f"Error: Missing expected column in '{tidal_file}'")
        return pd.DataFrame()

    except ValueError:
        print(f"Error: Data conversion failed in '{tidal_file}'")
        return pd.DataFrame()
    
def extract_single_year_remove_mean(year, data):
    start_data = pd.to_datetime(f'{year}-01-01 00:00:00')
    end_data = pd.to_datetime(f'{year}-12-31 23:00:00')
    year_data = data.loc[start_data:end_data, ['Sea Level']]
  
    # Handle potential extraction errors
    if year_data.empty:
        print(f"Warning: No data found for '{year}'.")
        return pd.DataFrame()

        sea_level_series = year_data['Sea Level'].dropna()

        if sea_level_series.empty:
            print(f"Warning: No valid 'Sea Level' data found for {year} after dropping NaNs.")
            return pd.DataFrame()  
  
    # Calculate and remove mean
    mmm = np.mean(year_data['Sea Level'])
    year_data['Sea Level'] -= mmm

    return year_data


def extract_section_remove_mean(start, end, data):
    try:
        # Convert date strings to datetime objects
        section_start = pd.to_datetime(f'{start} 00:00:00')
        section_end = pd.to_datetime(f'{end} 23:59:59')

        section_data = data.loc[section_start:section_end, ['Sea Level']].copy()

        # Interpolate missing values
        section_data['Sea Level'] = section_data['Sea Level'].interpolate(
            method='linear', limit_direction='both', limit_area='inside'
        )

        # Ensure there's data after interpolation, then remove mean
        if not section_data['Sea Level'].empty and section_data['Sea Level'].notna().any():
            mmm = np.mean(section_data['Sea Level'])
            section_data['Sea Level'] -= mmm
        else: 
            print(f"Warning: No valid data in section {start} to {end}")

        return section_data

    # Error Handling
    except 'Sea Level' not in data.columns:
        print(f"Error: 'Sea Level' column not found when trying to extract section {start} to {end}.")
        return pd.DataFrame()

    except section_data.empty:
        print(f"Warning: No data found for section from {start} to {end}.")
        return pd.DataFrame()
    
    except ValueError:
        print(f"Error: Invalid date format in {start} or {end}")
        return pd.DataFrame()
    
    except Exception:
        print(f"An unexpected error occurred while extracting data from {start} to {end}")
        return pd.DataFrame()

def join_data(data1, data2):
    if not isinstance(data1, pd.DataFrame) or not isinstance(data2, pd.DataFrame):
        print(f"Error: Both inputs must be pandas DataFrames for join_data. Got types: {type(data1)}, {type(data2)}")
        return pd.DataFrame()

    try:
        combined_data = pd.concat([data1, data2])
        sorted_data = combined_data.sort_index(ascending=True)
        return sorted_data
    
    except Exception as e:
        print(f"An error occurred during data joining: {e}")
        return pd.DataFrame() 

def sea_level_rise(data):

                                                     
    return 

def tidal_analysis(data, constituents, start_datetime):
    # Convert data into an array
    tidal_elevations = data['Sea Level'].to_numpy()
    tide = uptide.Tides(constituents)

    tide.set_initial_time(start_datetime) # Already in UTC
    seconds_since = (data.index.astype('int64').to_numpy()/1e9) - start_datetime.timestamp()

    # Harmonic analysis
    amp, pha = uptide.harmonic_analysis(tide, tidal_elevations, seconds_since)
    return amp, pha

def get_longest_contiguous_data(data):


    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
                     epilog="Copyright 2025, Zorah Rajput"
                     )

    parser.add_argument("directory",
                    help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose
    


