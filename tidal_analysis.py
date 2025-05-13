#!/usr/bin/env python3
"""
Created on Mon May  5 14:35:48 2025

@author: zorahrajput
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import wget
import os
import numpy as np
import uptide
import pytz
import math
import glob

def read_tidal_data(tidal_file):
    try:
        data = pd.read_csv(
            tidal_file,
            header=None,
            skiprows=11,  # Skip irrelevant lines
            sep=r'\s+',
            usecols=[1, 2, 3],
            names=['DateStr', 'TimeStr', 'SeaLevelRaw'], 
            dtype={'SeaLevelRaw' : str}, # Read as a string to make conversion easier - Gemini used 
            )
        
        data['Sea Level'] = data['SeaLevelRaw'].apply(
        lambda x: np.nan if any(suffix in x for suffix in ['M', 'N', 'T']) #Gemini
        else pd.to_numeric(x, errors='coerce')
        ) 
        data['DateTime'] = data['DateStr'].str.replace('/', '-') + ' ' + data['TimeStr']
        data = data.rename(columns={'DateStr' : 'Date', 'TimeStr' : 'Time'})
        data['DateTime'] = pd.to_datetime(data['DateTime'], format='%Y-%m-%d %H:%M:%S')
        data = data.set_index('DateTime')
        data = data.drop(columns=['SeaLevelRaw']) 
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {tidal_file}")

    
def extract_single_year_remove_mean(year, data):
    start_data = pd.to_datetime(f'{year}-01-01 00:00:00')
    end_data = pd.to_datetime(f'{year}-12-31 23:00:00')
    year_data = data.loc[start_data:end_data, ['Sea Level']]
  
    mmm = np.mean(year_data['Sea Level'])
    year_data['Sea Level'] -= mmm

    return year_data


def extract_section_remove_mean(start, end, data):
    section_start = pd.to_datetime(f'{start} 00:00:00')
    section_end = pd.to_datetime(f'{end} 23:00:00')
    section_data = data.loc[section_start:section_end, ['Sea Level']]
    
    mmm = np.mean(section_data['Sea Level'])
    section_data['Sea Level'] -= mmm

    return section_data 


def join_data(data1, data2):
    combined_data = pd.concat([data1, data2])
    sorted_data = combined_data.sort_index(ascending=True)

    return sorted_data



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
                     epilog="Copyright 2024, Jon Hill"
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
    


