import pandas as pd
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
# from ipynb.fs.full.my_functions_MOVES import VSP, vVSP, VSP_rg, vVSP_rg, Pneuma_to_MOVESID, OperatingMode, EmissionRate
import affine
from osgeo import gdal, ogr, osr
import time
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
# import pandas as pd


def data_parser(path_trajectory_data):
    csv.field_size_limit(1000000)  # This might need to be changed if the script does not run in Windows
    data_file = open(path_trajectory_data, 'r')
    data_reader = csv.reader(data_file)
    data = []
    for row in data_reader:
        data.append([elem for elem in row[0].split("; ")])
    return data


def create_new_trajectory_data_frame():
    headings_names = ['track_id', 'type', 'entry_gate', 'entry_time', 'exit_gate', 'exit_time', 'traveled_distance', 'avg_speed', 'lat', 'lon', 'speed', 'tan_accel', 'lat_accel', 'time']
    new_df = pd.DataFrame(columns=headings_names)
    return new_df, headings_names


# import elevation info
filename = "/Users/txu81/Desktop/epfl/4grade/Athens_All_STRM90_3x.tif"
demdata = gdal.Open(filename)
demarray = np.array(demdata.GetRasterBand(1).ReadAsArray())

affine_transform = affine.Affine.from_gdal(*demdata.GetGeoTransform())
inverse_transform = ~affine_transform

def elev(lon, lat, demarray, inverse_transform):
    # demarray is the DEM in array format
    # inverse transfor to convert from lon,lat to pixel coordinates
    px, py = [int(round(f)) for f in inverse_transform * (lon, lat)]  # write in a more efficient way ?

    return demarray[py, px]  # elevation in m


velev = np.vectorize(elev, excluded=['demarray', 'inverse_transform'])


# generate trajectories

maxInt = sys.maxsize  # to read "problematic" csv files with various numbers of columns
decrement = True
start = time.time()
name = 'aka'
path_input_trajectory_data = "/Users/txu81/Desktop/epfl/newdata/" + name + ".csv"  # csv file
path_to_export = "/Users/txu81/Desktop/epfl/newdata/" + name + "_trajectories/"  # folder
print("path:" + path_input_trajectory_data)
trajectory_data_array = data_parser(path_input_trajectory_data)
print(len(trajectory_data_array))
new_trajectory, column_names = create_new_trajectory_data_frame()
tracked_vehicle_id = 0
geodesic = pyproj.Geod(ellps='WGS84')
#
#
for tracked_vehicle_id in range(1, len(trajectory_data_array)):
# for tracked_vehicle_id in range(1, 12):
    print('Tracked Vehicle: ', tracked_vehicle_id)
    new_trajectory.at[0, column_names[0]] = trajectory_data_array[tracked_vehicle_id][0]  # 0: Tracked Vehicle
    new_trajectory.at[0, column_names[1]] = trajectory_data_array[tracked_vehicle_id][1]  # 1: Type
    new_trajectory.at[0, column_names[2]] = trajectory_data_array[tracked_vehicle_id][2]  # 2: Entry Gate
    new_trajectory.at[0, column_names[3]] = trajectory_data_array[tracked_vehicle_id][3]  # 3: Entry Time [ms]
    new_trajectory.at[0, column_names[4]] = trajectory_data_array[tracked_vehicle_id][4]  # 4: Exit Gate
    new_trajectory.at[0, column_names[5]] = trajectory_data_array[tracked_vehicle_id][5]  # 5: Exit Time [ms]
    new_trajectory.at[0, column_names[6]] = trajectory_data_array[tracked_vehicle_id][6]  # 6: Traveled Dist. [m]
    new_trajectory.at[0, column_names[7]] = trajectory_data_array[tracked_vehicle_id][7]  # 7: Avg. Speed [km/h]
    for j in range(8, len(trajectory_data_array[tracked_vehicle_id]), 6):
        try:
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8]] = float(trajectory_data_array[tracked_vehicle_id][j])          # 8: Latitude [deg]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 1]] = float(trajectory_data_array[tracked_vehicle_id][j + 1])  # 9: Longitude [deg]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 2]] = float(trajectory_data_array[tracked_vehicle_id][j + 2])  # 10: Speed [km/h]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 3]] = float(trajectory_data_array[tracked_vehicle_id][j + 3])  # 11: Tan. Accel. [ms-2]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 4]] = float(trajectory_data_array[tracked_vehicle_id][j + 4])  # 12: Lat. Accel. [ms-2]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 5]] = float(trajectory_data_array[tracked_vehicle_id][j + 5])  # 13: Time [ms]
        except IndexError:
            continue
        except ValueError:
            continue

    new_trajectory['distance'] = np.nan
    starting_lat = float(new_trajectory['lat'][0])
    starting_lon = float(new_trajectory['lon'][0])
    for row in range(0, new_trajectory.shape[0]):
        fwd_azimuth, back_azimuth, distance = geodesic.inv(starting_lon, starting_lat, float(new_trajectory['lon'][row]), float(new_trajectory['lat'][row]))
        new_trajectory.loc[row, 'distance'] = distance
    #     ### calculate Lane number
        if float(new_trajectory['lon'][row]) > - 0.8135526505400843 * float(new_trajectory['lat'][row]) + 54.63434783016084:
            lane = 1
        elif float(new_trajectory['lon'][row]) > - 0.8132094447923875 * float(new_trajectory['lat'][row]) + 54.621281864323706:
            lane = 2
        elif float(new_trajectory['lon'][row]) > - 0.8129186207919953 * float(new_trajectory['lat'][row]) + 54.610198412246:
            lane = 3
        else:
            lane = 4
        new_trajectory.loc[row, 'lane'] = lane
    new_trajectory['Elevation'] = velev(lon=new_trajectory['lon'].to_numpy(), lat=new_trajectory['lat'].to_numpy(), demarray=demarray, inverse_transform=inverse_transform)
    new_trajectory.to_csv(path_to_export + 'trajectory' + str(tracked_vehicle_id) + '.csv', index=False)
    new_trajectory, column_names = create_new_trajectory_data_frame()

end = time.time()
print()
print()
print('Time for extracting ' + str(tracked_vehicle_id) + ' vehicles was ' + str(int(divmod(end - start, 60)[0])) + ' minutes and ' +
      str(int(divmod(end - start, 60)[1])) + ' seconds.')
print()
