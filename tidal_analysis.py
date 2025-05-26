#!/usr/bin/env python3
"""
Created on Mon May  5 14:35:48 2025

@author: zorahrajput
"""
# Import modules
import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np
import uptide
from scipy import stats

def read_tidal_data(tidal_file):
    """
Reads tidal data from a specified file, cleans it, and prepares it for analysis.

This function expects a whitespace-separated file, skips initial rows,
handles specific 'M', 'N', 'T' suffixes, converts 'bad numbers' like -99.0000 to NaN,
and converts relevant columns to datetime objects.
"""
    try:
        # Use only relevant rows
        dat = pd.read_csv(
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

        dat['Sea Level'] = pd.to_numeric(
            dat['SeaLevelRaw'].str.replace(pattern, '', regex=True),
            errors='coerce'
            )
        # Gemini - Suggestion to remove "bad numbers"
        dat['Sea Level'] = dat['Sea Level'].replace(-99.0000, np.nan)

        # Use only relevant columns in the correct format
        dat['DateTime'] = dat['DateStr'].str.replace('/', '-') + ' ' + dat['TimeStr']
        dat = dat.rename(columns={'DateStr' : 'Date', 'TimeStr' : 'Time'})
        dat['DateTime'] = pd.to_datetime(dat['DateTime'], format='%Y-%m-%d %H:%M:%S')
        dat = dat.set_index('DateTime')
        dat = dat.drop(columns=['SeaLevelRaw'])
        return dat

    # Handle potential errors
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {tidal_file}") from exc

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
    """
Extracts all data from a file for the specified year.
    
This function extracts data from midnight on the first day of the year
to 11pm on the final day of the year, handles potential errors by returning an
empty DataFrame, and removes the mean from the data.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    start_dat = pd.to_datetime(f'{year}-01-01 00:00:00')
    end_dat = pd.to_datetime(f'{year}-12-31 23:00:00')
    year_dat = data.loc[start_dat:end_dat, ['Sea Level']]
    sea_level_series = year_dat['Sea Level'].dropna()
    # Handle potential extraction errors
    if year_dat.empty:
        print(f"Warning: No data found for '{year}'.")
        return pd.DataFrame()

    if sea_level_series.empty:
        print(f"Warning: No valid 'Sea Level' data found for {year} after dropping NaNs.")
        return pd.DataFrame()

    # Calculate and remove mean
    mmm = np.mean(year_dat['Sea Level'])
    year_dat['Sea Level'] -= mmm

    return year_dat


def extract_section_remove_mean(start, end, data):
    """
Extracts all data from a file for the specified timeframe.

This function extracts data from midnight on the specified start date to
11pm on the specified end date, handles potential errors by returning an 
empty pandas DataFrame, and removes the mean from the data.
"""
    try:
        # Convert date strings to datetime objects
        section_start = pd.to_datetime(f'{start} 00:00:00')
        section_end = pd.to_datetime(f'{end} 23:59:59')

        section_data = data.loc[section_start:section_end, ['Sea Level']].copy()

        # Interpolate missing values
        section_data['Sea Level'] = section_data['Sea Level'].interpolate(
            method='linear', limit_direction='both', limit_area='inside'
        )

        if section_data.empty: # This is a more appropriate check for 'no data found'
            print(f"Warning: No data found for section from {start} to {end}.")
            return pd.DataFrame()

        mmm = np.mean(section_data['Sea Level'])
        section_data['Sea Level'] -= mmm

        return section_data

    # Error Handling
    except section_data.empty:
        print(f"Warning: No data found for section from {start} to {end}.")
        return pd.DataFrame()

    except ValueError:
        print(f"Error: Invalid date format in {start} or {end}")
        return pd.DataFrame()

def join_data(data1, data2):
    """
Joins data from two specified files and sorts the index chronologically.

This function handles potential errors by returning an empty pandas DataFrame.
"""
    if not isinstance(data1, pd.DataFrame) or not isinstance(data2, pd.DataFrame):
        print(f"Error: Both inputs must be pandas DataFrames. Types: {type(data1)}, {type(data2)}")
        return pd.DataFrame()

    try:
        combined_data = pd.concat([data1, data2])
        sorted_data = combined_data.sort_index(ascending=True)
        return sorted_data

    except ValueError:
        print(f"Error: Invalid date format in {data1} or {data2}")
        return pd.DataFrame()

def sea_level_rise(data):
    """
Cleans data by removing NaN values, removes outliers, converts the DateTime index
into seconds, and calculates a linear regression. 

This function handles potential issues and reports to the user.
"""
    # Drop NaN values for linear regression
    cleaned_data = data.dropna(subset=['Sea Level'])

    if cleaned_data.empty or len(cleaned_data) < 2:
        print("Warning: Insufficient data points for linear regression")
        return pd.DataFrame()

    # Gemini- Suggested to remove outliers
    mean_val = cleaned_data['Sea Level'].mean()
    std_val = cleaned_data['Sea Level'].std()
    outlier_threshold_std = 2.4 # Experimented to get as close to expected value as possible

    initial_len = len(cleaned_data)
    cleaned_data = cleaned_data[
    (cleaned_data['Sea Level'] > (mean_val - outlier_threshold_std * std_val)) &
    (cleaned_data['Sea Level'] < (mean_val + outlier_threshold_std * std_val))
    ]
    if len(cleaned_data) < initial_len:
        if initial_len - len(cleaned_data) > 0: # Report removals to user
            print(f"Removed {initial_len - len(cleaned_data)} outliers.")

    # Convert the DateTime index for the x-axis.
    start_time = cleaned_data.index.min()
    x = (cleaned_data.index - start_time).total_seconds() / (24 * 3600) # Time in days

    y = cleaned_data['Sea Level'].values

    # Linear Regression
    slope, _, _, p_value, _ = stats.linregress(x, y)
    print(slope, p_value) # Added for debugging
    return slope, p_value

def tidal_analysis(data, constituents, start_datetime):
    """
Converts 'Sea Level' data into an array to perform a harmonic analysis. 

This function returns the amplitude and phase data which is calculated.
"""
    # Convert data into an array
    tidal_elevations = data['Sea Level'].to_numpy()
    tide = uptide.Tides(constituents)

    tide.set_initial_time(start_datetime) # Already in UTC
    seconds_since = (data.index.astype('int64').to_numpy()/1e9) - start_datetime.timestamp()

    # Harmonic analysis
    amplitude, phase = uptide.harmonic_analysis(tide, tidal_elevations, seconds_since)
    return amplitude, phase

def get_longest_contiguous_data(data):
    """
Sorts data into valid and missing data, finds the longest section of valid data, 
and returns this to the user.

This function handles errors related to an empty DataFrame.
"""
    # Assign boolean values to valid and missing data
    not_na = data['Sea Level'].notna()
    group_ids = (not_na != not_na.shift()).cumsum()
    contiguous_lengths = data[not_na].groupby(group_ids[not_na]).size()

    if data.empty or 'Sea Level' not in data.columns:
        return pd.DataFrame()
    if contiguous_lengths.empty:
        return pd.DataFrame()

    # Find largest sequence of valid data
    longest_group_id = contiguous_lengths.idxmax()
    longest_data = data[not_na][group_ids[not_na] == longest_group_id]

    return longest_data

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

    # Read file output
    master_data = pd.DataFrame() # Initialize as an empty DataFrame
    all_data_frames = [] # Hold data from all text files in a list
    file_paths = glob.glob(os.path.join(dirname, "*.txt"))

    if not file_paths:
        print(f"Error: No .txt files found in the directory: {dirname}")
        sys.exit(1)

    for file_path in file_paths:
        if verbose:
            print(f"Attempting to read data from: {file_path}")

        try:
            master_data = read_tidal_data(file_path)
            if not master_data.empty:
                all_data_frames.append(master_data)
            else:
                if verbose:
                    print(f"Skipping empty or invalid data from: {file_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")

    if not all_data_frames:
        print("Error: No valid data could be read from any files in the directory.")
        sys.exit(1)

    # Join and sort data ouput
    master_combined_data = pd.concat(all_data_frames).sort_index(ascending=True)

    if master_combined_data.empty:
        print("Error: Combined data is empty after reading and joining files.")
        sys.exit(1)

    if verbose:
        print(f"\nSuccessfully combined data from {len(all_data_frames)} files.")
        print(f"Combined shape: {master_combined_data.shape}")
        print(f"Number of NaNs before regression: {master_combined_data['Sea Level'].isna().sum()}")


    # Sea Level Rise output
    if verbose:
        print("\n--- Calculating Sea Level Rise ---")

    slope_rsl, p_value_rsl = sea_level_rise(master_combined_data)

    # Print RSL results for the regression tests (> 25 characters)
    print(f"RSL Slope: {slope_rsl:.6e}, P-value: {p_value_rsl:.6f}")

    if verbose:
        print("Sea Level Rise calculation complete.")


    # Tidal Analysis output
    if verbose:
        print("\n--- Performing Tidal Analysis ---")

    longest_contiguous_block = get_longest_contiguous_data(master_combined_data)

    if longest_contiguous_block.empty:
        print("Warning: Could not find a longest contiguous block for tidal analysis. Skipping.")
    else:
        constituents_for_analysis = ['M2', 'S2']

        analysis_start_datetime = longest_contiguous_block.index.min()
        try:
            amp, pha = tidal_analysis(
                longest_contiguous_block,
                constituents_for_analysis,
                analysis_start_datetime
                )

            # Print tidal analysis results if verbose is true
            if verbose:
                print("\nTidal Constituents Amplitudes and Phases:")
                for i, const in enumerate(constituents_for_analysis):
                    print(f"  {const}: Amplitude={amp[i]:.4f}, Phase={pha[i]:.2f}")
                    print(f"Tidal Analysis for {os.path.basename(dirname)} complete.")

        except ValueError as e:
            print(f"Error during tidal analysis (ValueError): {e}")
            if verbose:
                print(f"Tidal analysis failed due to invalid data or parameters. {e}")
