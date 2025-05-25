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

def read_tidal_data(filename):

    return 0
    
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
    


