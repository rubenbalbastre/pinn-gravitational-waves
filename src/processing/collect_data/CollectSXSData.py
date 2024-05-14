"""
Objective:
* obtain data previously selected
* save its masses and spins
* save its waveform
"""

import sxs
import os
import qgrid
import time
import pandas as pd
import numpy as np
from sys import path

path_to_root_project = "../../../"
path.append(path_to_root_project)

from src.utils.collect_data.simulation_data import Simulation_Data, Sim_IDs
from src.utils.collect_data.utils import create_directory_if_does_not_exist


# path parameters
output_folder = path_to_root_project + 'data/input/case_2/'
master_data_file_name = '/master_data.csv'

start = time.time()

# For more infor see "02-Catalog.ipynb" in "https://mybinder.org/v2/gh/moble/sxs_notebooks/master"
catalog = sxs.load("catalog")
dataframe = catalog.table
qgrid.show_grid(dataframe, precision=8, show_toolbar=True, grid_options={"forceFitColumns": False})

# dict to save identification data of each waveform
info_dict = {'ID':[], 'initial_separation': [], 'mA':[],'mB':[], 'spinA':[], 'spinB':[], 'q':[], 'e':[]}

# -------------------------------------------------
# SELECTION 1.
# select n waveform in a space range of effective spin and ratio of masses

# q_min = 1
# q_max = 10
# eff_spin_min = -1e-4 #1e-8
# eff_spin_max = 1e-4 #1e-1
# IDs_list = Sim_IDs(dataframe, q_min, q_max, eff_spin_min, eff_spin_max)
# print("Número de muestras", len(IDs_list) )


# -------------------------------------------------
# SELECTION 2.
# select n random waveforms from previous selection

# num_waveforms = 1
# indexs = np.random.randint(0,len(IDs_list), size=num_waveforms)
# print(pd.DataFrame(IDs_list, index=range(0,len(IDs_list))).iloc[indexs])
# IDs_list = pd.DataFrame(IDs_list, index=range(0,len(IDs_list)), columns=['names']).iloc[indexs].names
# print("Número de muestras seleccionadas", len(IDs_list) )


# -------------------------------------------------
# SELECTION 3.
# generate specific dataset

IDs_list = [
    # "SXS:BBH:0063", "SXS:BBH:0055", "SXS:BBH:0166", "SXS:BBH:0168", "SXS:BBH:0169", "SXS:BBH:0187", "SXS:BBH:0186", "SXS:BBH:0188", "SXS:BBH:0198"

    # "SXS:BBH:1356",
    # "SXS:BBH:1357",
    # "SXS:BBH:1358",
    # "SXS:BBH:1359",
    # "SXS:BBH:1360",
    "SXS:BBH:0211",
    "SXS:BBH:0217"
]

create_directory_if_does_not_exist(output_folder)

for number_of_loop, sim_ID in enumerate(IDs_list):

    if not os.path.exists(output_folder+sim_ID):

        print(f"Processing Waveform {sim_ID}")
        # Define simulation object
        sim = Simulation_Data(sim_ID)

        # Easily get useful parameters
        mA  = sim.mA
        mB  = sim.mB
        q   = sim.q
        e   = sim.e
        spinA = sim.spinA_ini
        spinB = sim.spinB_ini
        initial_sep = sim.initial_separation

        # For parameters/data not in "__init__" you can directly type "data_name" as in the metadata file
        # eg.
        Num_Orbits = sim.Metadata('number_of_orbits')

        # Horizons data (Centers of BH's in inertial frames)
        xyz_A,   times = sim.Horizons_center_in_and_times('A')                # "in" for inertial frame"
        xyz_B,   times = sim.Horizons_center_in_and_times('B')
        xyz_com, times = sim.xyz_COM()

        # Waveform data for all (l,m)
        W_obect  = sim.W()
        W_all_lm = W_obect.data
        t        = W_obect.t

        # Waveform data for a particular (l,m)
        l = 2
        m = 2
        W_lm, t = sim.W_lm_and_times(l,m)

        show = False
        save = True

        create_directory_if_does_not_exist(output_folder+sim_ID)

        # save waveform details
        info_dict['ID'].append(sim_ID); info_dict['q'].append(q); info_dict['e'].append(e); info_dict['initial_separation'].append(initial_sep)
        info_dict['mA'].append(mA); info_dict['mB'].append(mB); info_dict['spinA'].append(spinA); info_dict['spinB'].append(spinB)

        # Saving waveform data in .txt file
        datvec = np.zeros((len(t),2))
        datvec[:,0] = t
        datvec[:,1] = np.real(W_lm)
        np.savetxt(output_folder + sim_ID + '/waveform_real.txt',datvec) if save else None
        datvec[:,1] = np.imag(W_lm)
        np.savetxt(output_folder + sim_ID + '/waveform_imag.txt',datvec) if save else None

        # Saving COM-corrected trajectory data in .txt file
        datvec = np.zeros((len(times),3))
        datvec[:,0] = times
        datvec[:,1] = xyz_A[:,0]
        datvec[:,2] = xyz_A[:,1]
        np.savetxt(output_folder + sim_ID + '/trajectoryA.txt',datvec) if save else None
        datvec[:,0] = times
        datvec[:,1] = xyz_B[:,0]
        datvec[:,2] = xyz_B[:,1]
        np.savetxt(output_folder + sim_ID + '/trajectoryB.txt',datvec) if save else None

    # evaluation of time
    print(number_of_loop + 1, " out of ", len(IDs_list))

# master data
info = pd.DataFrame(info_dict)

# if info.csv exist, append downloaded master data information
if os.path.exists(output_folder + master_data_file_name):
    readed_info_df = pd.read_csv(output_folder + master_data_file_name)
    info = readed_info_df.append(info)

print("Saving Master Data table")
info.to_csv(output_folder + master_data_file_name, index=False)

print("Collect Data Process finished in ", time.time()-start, " s")
