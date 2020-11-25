import gdal as gdal
import time
import sys
import csv
import numpy as np
import math
import pyproj
import numpy as np
from scipy import stats
from scipy.stats import norm
import pandas as pd
# import modin.pandas as pd #experimental library to speed up pandas --> not working well
import glob
import os
# from ipynb.fs.full.my_functions_MOVES import VSP, vVSP, VSP_rg, vVSP_rg, Pneuma_to_MOVESID, OperatingMode,
# EmissionRate
import affine
from osgeo import gdal, ogr, osr
import time
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


def elev(lon, lat, demarray, inverse_transform):
    # demarray is the DEM in array format
    # inverse transfor to convert from lon,lat to pixel coordinates
    px, py = [int(round(f)) for f in inverse_transform * (lon, lat)]
    return demarray[py, px]  # elevation in m


# import pandas as pd

all_df = pd.read_pickle('/Users/txu81/Desktop/epfl/6rev/trajectories_with_lanes.pickle')

# import elevation info
filename = "/Users/txu81/Desktop/epfl/4grade/Athens_All_STRM90_3x.tif"
demdata = gdal.Open(filename)
demarray = np.array(demdata.GetRasterBand(1).ReadAsArray())
velev = np.vectorize(elev, excluded=['demarray', 'inverse_transform'])

affine_transform = affine.Affine.from_gdal(*demdata.GetGeoTransform())
inverse_transform = ~affine_transform

# df = all_df[0]
# temp_class = df['Type'][0]
# print([str(temp_class)] * len(df))
for i in range(len(all_df)):
    df = all_df[i]
    temp_id = df['Tracked Vehicle'][0]
    temp_class = df['Type'][0]
    df = df.iloc[1:]
    df['Elevation'] = velev(lon=df['Longitude [deg]'].to_numpy(), lat=df['Latitude [deg]'].to_numpy(),
                            demarray=demarray, inverse_transform=inverse_transform)
    df["id"] = [temp_id] * len(df)
    df["class"] = [str(temp_class)] * len(df)

    if len(df[df["lane_actual"] == 1]) > 0:
        (df[df["lane_actual"] == 1])[["Time[ms]", "CumSum", "Elevation", "id", "class"]].to_csv(
            '/Users/txu81/Desktop/epfl/6rev/1/'
            + str(i) + '.csv', index=False)

    if len(df[df["lane_actual"] == 2]) > 0:
        (df[df["lane_actual"] == 2])[["Time[ms]", "CumSum", "Elevation", "id", "class"]].to_csv(
            '/Users/txu81/Desktop/epfl/6rev/2/'
            + str(i) + '.csv', index=False)

    if len(df[df["lane_actual"] == 3]) > 0:
        (df[df["lane_actual"] == 3])[["Time[ms]", "CumSum", "Elevation", "id", "class"]].to_csv(
            '/Users/txu81/Desktop/epfl/6rev/3/'
            + str(i) + '.csv', index=False)

    if len(df[df["lane_actual"] == 4]) > 0:
        (df[df["lane_actual"] == 4])[["Time[ms]", "CumSum", "Elevation", "id", "class"]].to_csv(
            '/Users/txu81/Desktop/epfl/6rev/4/'
            + str(i) + '.csv', index=False)

    if len(df[df["lane_actual"] == 5]) > 0:
        (df[df["lane_actual"] == 5])[["Time[ms]", "CumSum", "Elevation", "id", "class"]].to_csv(
            '/Users/txu81/Desktop/epfl/6rev/5/'
            + str(i) + '.csv', index=False)

    if len(df[df["lane_actual"] == 6]) > 0:
        (df[df["lane_actual"] == 6])[["Time[ms]", "CumSum", "Elevation", "id", "class"]].to_csv(
            '/Users/txu81/Desktop/epfl/6rev/6/'
            + str(i) + '.csv', index=False)
