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
   

    return 


def extract_section_remove_mean(start, end, data):


    return 


def join_data(data1, data2):

    return 



def sea_level_rise(data):

                                                     
    return 

def tidal_analysis(data, constituents, start_datetime):


    return 

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
    


