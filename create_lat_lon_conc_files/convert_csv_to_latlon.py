"""
convert_csv_to_latlon.py

Convert a .csv file from NSIDC-0790 from:
  EASEx EASEy Conc
to 
  Lat Lon Conc

Last tested with NSIDC-0790 v1.1 files

Sample usage:
    python convert_csv_to_latlon.py nsidc0790_Oct-imparcels_19991001_20001001_v1.1.csv
  where the CSV file was downloaded from NSIDC-0790 v1.1

Sample output:
    parcels_lats_19991001_20001001.csv
    parcels_lons_19991001_20001001.csv
    parcels_conc_19991001_20001001.csv

See the local directories:
    ./sample_output/:  Contains one (only!) of the three csv files created
                       by this code on the contributor's system when this
                       file was originally written.  The file is gzipped in
                       this repository to save space.  This file can serve
                       as a check to see how close to the original function
                       the code performs on another system.  The other output
                       files are *not* included here.
"""

import datetime as dt
import os
import sys
import numpy as np
import pyproj


def get_usage_string():
    """Return the usage string."""

    usage_string = f"""
    Usage:
        python {sys.argv[0]} <nsidc0790_filename>
      eg
        python {sys.argv[0]} nsidc0790_Oct-imparcels_19991001_20001001_v1.1.csv
    """

    return usage_string


def parse_0790_filename(fn):
    """Parse the nsidc0790 file name.
    By convention, the filename is expected to be of the form:
        nsidc0790_imparcels_<startymd>_<endymd>_v<ver>.csv.gz
      eg:
        nsidc0790_imparcels_20000801_20010801_v1.0.csv.gz"""
    fn_parts = os.path.basename(fn).split('_')
    try:
        startymd = dt.datetime.strptime(fn_parts[2], '%Y%m%d').strftime('%Y%m%d')
        endymd = dt.datetime.strptime(fn_parts[3], '%Y%m%d').strftime('%Y%m%d')
    except ValueError:
        raise RuntimeError(f'Could not determine dates from filename: {fn}')

    return startymd, endymd


def read_csv_header(fn, header_line=14):
    """Read only the header from a csv file."""
    import gzip
    import csv

    if fn[-3:] == '.gz':
        with gzip.open(fn) as f:
            for _ in range(header_line - 1):
                f.readline()
            orig_csv_header = f.readline()
            orig_csv_header = orig_csv_header.decode('utf-8').rstrip()
    else:
        with open(fn) as f:
            for _ in range(header_line - 1):
                f.readline()
            orig_csv_header = f.readline()

    print(f'orig_csv_header: {orig_csv_header}')
    list_of_dates = [val[2:] for val in orig_csv_header.split(',') if val[:2] == 'i_']
    csv_header = ','.join(list_of_dates)
    return csv_header


def gen_latlonconc_from_csv(ifn, num_header_lines=14, verbose=False):
    startymd, endymd = parse_0790_filename(ifn)

    csv_header = read_csv_header(ifn, header_line=num_header_lines)

    ofn_lats = f'parcels_lats_{startymd}_{endymd}.csv'
    ofn_lons = f'parcels_lons_{startymd}_{endymd}.csv'
    ofn_conc = f'parcels_conc_{startymd}_{endymd}.csv'
    
    if verbose:
        print('Output file names:')
        print(f'   lats: {ofn_lats}')
        print(f'   lons: {ofn_lons}')
        print(f'   conc: {ofn_conc}', flush=True)
    
    if verbose:
        print(f'Assuming {num_header_lines} in the uncompressed csv file.')

    arr = np.loadtxt(
        ifn,
        delimiter=',',
        skiprows=num_header_lines,
        dtype=np.float32,
    )

    if verbose:
        print(f'shape of arr: {arr.shape}', flush=True)
    
    missing_locs_prior = arr == -999
    missing_locs_post = arr == 999
    arr[missing_locs_prior] = 0
    arr[missing_locs_post] = 0
    
    ivals = arr[:, 0::3]
    jvals = arr[:, 1::3]
    cvals = arr[:, 2::3]
    
    # Convert from ivals, jvals (grid indices) to xvals, yvals (meters)
    # Grid is 361 x 361 with 1 grid cell = 200.5402km/8 = 25067.525m
    # Middle of (i=180, j=180) is at (x=0, y=0)
    # Middle of (i=179, j=179) is at (x=-25067.525, y=25067.525)
    #   x(0) = (i - 180) * 25067.525
    #   y(0) = (180 - j) * 25067.525
    grid_resolution = 25067.525
    xvals = (ivals - 180) * grid_resolution
    yvals = (180 - jvals) * grid_resolution
    
    if verbose:
        print(f'range of ivals:  {np.nanmin(ivals)} to {np.nanmax(ivals)}')
        print(f'range of jvals:  {np.nanmin(jvals)} to {np.nanmax(jvals)}')
        print(f'range of cvals:  {np.nanmin(cvals)} to {np.nanmax(cvals)}')
        print(' ')
        print(f'range of xvals:  {np.nanmin(xvals)} to {np.nanmax(xvals)}')
        print(f'range of yvals:  {np.nanmin(yvals)} to {np.nanmax(yvals)}')
        print(' ', flush=True)
    
    # New pyproj style
    transformer = pyproj.Transformer.from_crs('epsg:3408', 'epsg:4326')
    lats, lons = transformer.transform(xvals, yvals)
    if verbose:
        print(f'range of lats:  {np.nanmin(lats)} to {np.nanmax(lats)}')
        print(f'range of lons:  {np.nanmin(lons)} to {np.nanmax(lons)}')
        print(' ', flush=True)
    
    ll_arr = arr.copy()
    ll_arr[:, 0::3] = lats[:, :]
    ll_arr[:, 1::3] = lons[:, :]
    
    ll_arr[missing_locs_prior] = -999
    ll_arr[missing_locs_post] = 999
    
    lats = ll_arr[:, 0::3]
    lons = ll_arr[:, 1::3]
    cvals = ll_arr[:, 2::3]
    
    # To save the new array, you could do the following
    # np.savetxt(ofn, ll_arr, fmt='%.3f', delimiter=',')
    # print(f'Wrote: {ofn}')
    
    np.savetxt(ofn_lats, lats, fmt='%.3f', delimiter=',', header=csv_header, comments='')
    if verbose:
        print(f'Wrote: {ofn_lats}', flush=True)

    np.savetxt(ofn_lons, lons, fmt='%.3f', delimiter=',', header=csv_header, comments='')
    if verbose:
        print(f'Wrote: {ofn_lons}', flush=True)

    np.savetxt(ofn_conc, cvals, fmt='%.3f', delimiter=',', header=csv_header, comments='')
    if verbose:
        print(f'Wrote: {ofn_conc}', flush=True)
    
    if verbose:
        print(f' ')
        
        is_not_missing_lats = (lats > -900) & (lats < 900)
        print(f'range of lats: {np.min(lats[is_not_missing_lats])} to {np.max(lats[is_not_missing_lats])}  (shape: {lats.shape})')
        
        is_not_missing_lons = (lons > -900) & (lons < 900)
        print(f'range of lons: {np.min(lons[is_not_missing_lons])} to {np.max(lons[is_not_missing_lons])}  (shape: {lons.shape})')
        
        is_not_missing_cvals = (cvals > -900) & (cvals < 900)
        print(f'range of cvals: {np.min(cvals[is_not_missing_cvals])} to {np.max(cvals[is_not_missing_cvals])}  (shape: {cvals.shape})')


if __name__ == '__main__':
    try:
        ifn = sys.argv[1]
        assert os.path.isfile(ifn)
    except IndexError:
        raise SystemExit(f'\nNo input file given\n{get_usage_string()}')
    except AssertionError:
        raise RuntimeError(f'Specified input file does not exist')

    gen_latlonconc_from_csv(ifn, verbose=True)
