############################
# Given the "Simulation_ID" for SpEC,
# the following code can be used to easily get parameters, horizon_centers, and waveforms


# For more information on the sxs package check: https://github.com/sxs-collaboration/sxs
# some noteboks with info on how to use sxs package are also available for more info on : https://mybinder.org/v2/gh/moble/sxs_notebooks/master
# Mike Boyl's tutorial given at ICERM's first workshop regarding this is available on : https://icerm.brown.edu/video_archive/?play=2352

# https://data.black-holes.org/waveforms/catalog.html

import sxs
import numpy as np
from pandas import DataFrame
from typing import List


def Sim_IDs(catalog_table_dataframe: DataFrame, q_min: float, q_max: float, eff_spin_min: float, eff_spin_max: float) -> List[str]:
    """
    Filter waveforms according to some values of effective spin and ratio of masses q
    """

    catalog_table_dataframe_selected = catalog_table_dataframe[  
        (catalog_table_dataframe['reference_mass_ratio'] >= q_min) 
        &
        (catalog_table_dataframe['reference_mass_ratio'] <= q_max) 
        &
        (catalog_table_dataframe['reference_chi_eff'] >= eff_spin_min ) 
        & 
        (catalog_table_dataframe['reference_chi_eff'] <= eff_spin_max ) 
    ]

    return list(catalog_table_dataframe_selected.index)



class Simulation_Data:

    def __init__(self, sim_ID: str):

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


def lm_Index(lm_List,l,m):
    cols    = np.where(lm_List[:,0]==l)[0]
    row_ind = np.where(lm_List[cols,1]==m)[0][0]
    return cols[row_ind]


all_available_keys_names = '''
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
