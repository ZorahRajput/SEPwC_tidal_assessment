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
            skiprows=11,  # Skip irrelevant lines
            sep=r'\s+',
            usecols=[1, 2, 3],
            names=['DateStr', 'TimeStr', 'SeaLevelRaw'], 
            dtype={'SeaLevelRaw' : str}, # Read as a string to make conversion easier - Gemini used 
            )
        
        data['DateTime'] = data['DateStr'].str.replace('/', '-') + ' ' + data['TimeStr']
        data['Date'] = pd.to_datetime(data['DateTime'], format='%Y-%m-%d %H:%M:%S')
        data = data.set_index('Date')
        data['Sea Level'] = data['SeaLevelRaw'].apply(lambda x: np.nan if 'M' in x else pd.to_numeric(x, errors='coerce')) # Gemini used for lambda
        data = data[['Sea Level']]
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {tidal_file}")

    
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
    


