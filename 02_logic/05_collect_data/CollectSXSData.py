############################
# Given the "Simulation_ID" for SpEC,
# the following code can be used to easily get parameters, horizon_centers, and waveforms


# For more information on the sxs package check: https://github.com/sxs-collaboration/sxs
# some noteboks with info on how to use sxs package are also available for more info on : https://mybinder.org/v2/gh/moble/sxs_notebooks/master
# Mike Boyl's tutorial given at ICERM's first workshop regarding this is available on : https://icerm.brown.edu/video_archive/?play=2352

# https://data.black-holes.org/waveforms/catalog.html

############################

import sxs
import os
import numpy as np
import pandas as pd
import corner
import qgrid
import time
#___________________________________________________________
import matplotlib.pyplot as plt

start = time.time()

#___________________________________________________________

class Simulation_Data:
    def __init__(self,sim_ID):

        self.sim_ID   = sim_ID
        self.metadata = sxs.load(self.sim_ID + '/Lev/metadata.json')

        self.e        = self.metadata['reference_eccentricity']
        self.q        = self.metadata['reference_mass_ratio']
        self.mA       = self.metadata['reference_mass1']
        self.mB       = self.metadata['reference_mass2']
        self.mC       = self.metadata['remnant_mass']
        self.spinA_ini    = self.metadata['reference_dimensionless_spin1']
        self.spinB_ini    = self.metadata['reference_dimensionless_spin2']
        self.mean_anamoly = self.metadata['reference_mean_anomaly']
        self.initial_separation = self.metadata['initial_separation']

    def Metadata(self, param_name):
        param_val = self.metadata[param_name]
        return param_val

    def Horizons_center_in_and_times(self,BH_name, COM_correction = True): # COM corrected - Centers of BH's in inertial frames
        horizons = sxs.load(self.sim_ID + '/Lev/Horizons.h5')

        if   BH_name == 'A' : xyz = horizons.A.coord_center_inertial
        elif BH_name == 'B' : xyz = horizons.B.coord_center_inertial

        xyz_COM  = horizons.newtonian_com # Eq 4 (https://arxiv.org/pdf/1904.04842.pdf)
        if COM_correction : xyz = xyz - xyz_COM

        times = horizons.A.time
        return xyz, times

    def xyz_t_AhC(self):
        horizons = sxs.load(self.sim_ID + '/Lev/Horizons.h5')
        xyz = horizons.C.coord_center_inertial
        t = horizons.C.time
        return xyz,t

    def xyz_COM(self):
        xyz = sxs.load(self.sim_ID + '/Lev/Horizons.h5').newtonian_com # Eq 4 (https://arxiv.org/pdf/1904.04842.pdf)
        return xyz, xyz.t

    def W(self, extrapolation_order = 4, Lev = 'Lev'):
        W = sxs.load(self.sim_ID +'/'+Lev+'/rhOverM_Asymptotic_GeometricUnits_CoM' , extrapolation_order=extrapolation_order)
        return W

    def W_lm_and_times(self,l,m):
        W       = self.W()
        lm_Ind  = lm_Index(W.LM,l,m)         # W.LM gives (l,m) list
        return W.data[:,lm_Ind] , W.time



#___________________________________________________________

def lm_Index(lm_List,l,m):
    cols    = np.where(lm_List[:,0]==l)[0]
    row_ind = np.where(lm_List[cols,1]==m)[0][0]
    return cols[row_ind]


#___________________________________________________________


#___________________________________________________________
# For more infor see "02-Catalog.ipynb" in "https://mybinder.org/v2/gh/moble/sxs_notebooks/master"
#___________________________________________________________
catalog = sxs.load("catalog")
dataframe = catalog.table
qgrid.show_grid(dataframe, precision=8, show_toolbar=True, grid_options={"forceFitColumns": False})


def Sim_IDs(q_min, q_max, eff_spin_min, eff_spin_max):
    a = dataframe[  (dataframe['reference_mass_ratio'] >= q_min        ) & \
                    (dataframe['reference_mass_ratio'] <= q_max        ) & \
                    (dataframe['reference_chi_eff']    >= eff_spin_min ) & \
                    (dataframe['reference_chi_eff']    <= eff_spin_max ) ]

    return list(a.index)

# seleccionamos 10 al azar
# num_ondas = 10
# indexs = np.random.randint(0,len(IDs_list), size=num_ondas)
# print(pd.DataFrame(IDs_list, index=range(0,len(IDs_list))).iloc[indexs])
# IDs_list = pd.DataFrame(IDs_list, index=range(0,len(IDs_list)), columns=['names']).iloc[indexs].names
# print("Número de muestras seleccionadas", len(IDs_list) )

#___________________________________________________________
# All available keys names are :
'''
'object_types', 'initial_separation', 'initial_orbital_frequency',
       'initial_adot', 'initial_ADM_energy', 'initial_ADM_linear_momentum',
       'initial_ADM_linear_momentum_mag', 'initial_ADM_angular_momentum',
       'initial_ADM_angular_momentum_mag', 'initial_mass1', 'initial_mass2',
       'initial_mass_ratio', 'initial_dimensionless_spin1',
       'initial_dimensionless_spin1_mag', 'initial_dimensionless_spin2',
       'initial_dimensionless_spin2_mag', 'initial_position1',
       'initial_position2', 'com_correction_space_translation',
       'com_correction_space_translation_mag', 'com_correction_boost_velocity',
       'com_correction_boost_velocity_mag', 'reference_time',
       'reference_separation', 'reference_orbital_frequency_mag',
       'reference_mass_ratio', 'reference_chi1_mag', 'reference_chi2_mag',
       'reference_chi_eff', 'reference_chi1_perp', 'reference_chi2_perp',
       'reference_eccentricity', 'reference_eccentricity_bound',
       'reference_mean_anomaly', 'reference_mass1', 'reference_mass2',
       'reference_dimensionless_spin1', 'reference_dimensionless_spin1_mag',
       'reference_dimensionless_spin2', 'reference_dimensionless_spin2_mag',
       'reference_orbital_frequency', 'reference_position1',
       'reference_position2', 'relaxation_time', 'common_horizon_time',
       'remnant_mass', 'remnant_dimensionless_spin',
       'remnant_dimensionless_spin_mag', 'remnant_velocity',
       'remnant_velocity_mag', 'eos', 'initial_data_type', 'disk_mass',
       'ejecta_mass', 'url', 'metadata_path'
'''
#___________________________________________________________
#___________________________________________________________


"""
Objective:
* obtain data previously selected
* save its masses and spins
* save its waveform
"""

# Example:

# Simulation ID
# sim_ID = 'SXS:BBH:0217' # e≈0
# sim_ID = 'SXS:BBH:1356' # e>0

# búsqueda global para tener un pequeño dataset
q_min = 1
q_max = 10
eff_spin_min = -1e-4 #1e-8
eff_spin_max = 1e-4 #1e-1
# IDs_list = Sim_IDs(q_min, q_max, eff_spin_min, eff_spin_max)
# print("Número de muestras", len(IDs_list) )


# dict to save identification data of each waveform
info_dict = {'ID':[], 'initial_separation': [], 'mA':[],'mB':[], 'spinA':[], 'spinB':[], 'q':[], 'e':[]}

# loop to download all data
i = 0

# generate a dataset for training case 2
IDs_list = [
    # "SXS:BBH:0063", "SXS:BBH:0055", "SXS:BBH:0166", "SXS:BBH:0168", "SXS:BBH:0169", "SXS:BBH:0187", "SXS:BBH:0186", "SXS:BBH:0188", "SXS:BBH:0198"

    "SXS:BBH:1356",
    "SXS:BBH:1357",
    "SXS:BBH:1358",
    "SXS:BBH:1359",
    "SXS:BBH:1360",
    "SXS:BBH:1361",
]
output_folder = '../../01_data/01_input/03_case_3/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for sim_ID in IDs_list:

    if not os.path.exists(output_folder+sim_ID):

        i += 1
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



        #___________________________________________________________
        #___________________________________________________________
        show = False
        save = True

        if os.path.exists(output_folder+sim_ID):
            pass
        else:
            os.mkdir(output_folder+sim_ID)


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
    else:
        pass

    # evaluation of time
    print(i, " out of ",len(IDs_list))

# save info data
info = pd.DataFrame(info_dict)

if os.path.exists(output_folder + '/info.csv'):
    readed_info_df = pd.read_csv(output_folder + '/info.csv')
    info = readed_info_df.append(info)
    info.to_csv(output_folder + '/info.csv')
else:
    info.to_csv(output_folder + '/info.csv')


print("Importing finished")
print(time.time()-start, " s")