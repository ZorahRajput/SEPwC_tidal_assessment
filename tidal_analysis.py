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

def read_tidal_data(tidal_file):
    try:
        data = pd.read_csv(
            tidal_file,
            skiprows=10,  # Skip the two header lines
            delim_whitespace=True,  # Use whitespace as the delimiter
            usecols=[1, 2, 3],      # Select the Date, Time, and ASLVZZ01 columns
            names=['Date', 'Time', 'SeaLevel'], # Rename columns
            dtype={'SeaLevel' : str}, # Read as a string to make conversion easier - Gemini used here
            parse_dates={'Date': [0, 1]}, 
            date_format='%Y/%m/%d %H:%M:%S',
            converters={'Sea Level': lambda x: np.nan if 'M' in x else float(x)}
            )
        
        data = data.set_index('Date')
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
    


