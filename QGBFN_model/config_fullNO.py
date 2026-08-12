#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:20:42 2021

@author: leguillou
"""

#name_experiment = '06-08_018x037_tau_3_K_0.5' # name of the experiment
name_experiment = 'fullNO_NK_dt300_fine_V23' # name of the experiment
#name_experiment = 'NorKyst trial' # name of the experiment
dt=300
#################################################################################################################################
# Global libraries     
#################################################################################################################################

from datetime import datetime, timedelta
 
#################################################################################################################################
# EXPERIMENTAL PARAMETERS
#################################################################################################################################
EXP = dict(

    name_experiment = name_experiment, # name of the experiment

    saveoutputs = True, # save outputs flag (True or False)

    name_exp_save = name_experiment, # name of output files

    path_save = f'outputs/{name_experiment}/', # path of output files

    tmp_DA_path = f"scratch/{name_experiment}", # temporary data assimilation directory path,

    init_date = datetime(2025,6,1,0), # initial date (yyyy,mm,dd,hh) 

    final_date = datetime(2025,8,2,0),  # final date (yyyy,mm,dd,hh) 

    assimilation_time_step = timedelta(hours=3),  

    saveoutput_time_step = timedelta(hours=3),  # time step at which the states are saved 

    flag_plot = 0, # 0= no plots, 1 = only back and forward iterations, 2 = observations and 1.

    write_obs = True

)
    
#################################################################################################################################
# GRID parameters
#################################################################################################################################
NAME_GRID = 'myGRID'

myGRID = dict(

    super = 'GRID_GEO',

    # Zone 1
    #lon_min = 2,                                        # domain min longitude
    #lon_max = 13.9,                                        # domain max longitude
    #lat_min = 61,                                         # domain min latitude
    #lat_max = 68.5,                                         # domain max latitude

    # Zone 2
    #lon_min = 2,                                        # domain min longitude
    #lon_max = 13.9,                                        # domain max longitude
    #lat_min = 57.5,                                         # domain min latitude
    #lat_max = 68.5,                                         # domain max latitude

    # fullNO
    lon_min = 2,                                          # domain min longitude
    lon_max = 13.9,                                        # domain max longitude
    lat_min = 57.5,                                         # domain min latitude
    lat_max = 68.5,                                         # domain max latitude

    dlat = 0.018*2,                                            # zonal grid spatial step (in degree)

    dlon = 0.037*2,                                            # meridional grid spatial step (in degree)

    #dlat = 0.1,                                            # zonal grid spatial step (in degree)

    #dlon = 0.1,   

    name_init_mask = './data/NK/norkyst_landmask_regular_clean.nc',

    #name_init_mask = './data/NK/norkyst_landmask_better.nc',

    name_var_mask = {'lon':'lon','lat':'lat','var':'landmask'}
)


# Regular cartesian grid 
GRID_CAR = dict(

    super = 'GRID_CAR',

    lon_min = 0.1,                                        # domain min longitude

    lon_max = 8,                                           # domain max longitude

    lat_min = 58,                                         # domain min latitude
 
    lat_max = 65,                                          # domain max latitude

    dx = 2,                                              # grid spacing in km

    nx = None,                                             # If not None, use nx to compute dx 

    ny = None,                                             #

    name_init_mask = './data/NK/LandMask_NK_fullNO.nc',

    name_var_mask = {'lon':'lon','lat':'lat','var':'mask'}
)


#################################################################################################################################
# Model parameters
#################################################################################################################################
NAME_MOD = 'myMOD'

myMOD = dict(
    
    super = 'MOD_QG1L_NP',

    name_var = {'SSH':"ssh"},

    init_from_bc = True,

    dist_sponge_bc = 4, # Width (in km) of the band where boundary conditions are applied to edges of the domain and to coastal aeras

    dtmodel = dt, # model timestep

    name_init_var = {},

    dir_model = None,

    var_to_save = None,

    upwind = 3,

    upwind_adj = None,

    Reynolds = False,

    qgiter = 20,

    qgiter_adj = None,

    c0 = 0.6, # If not None, fixed value for phase velocity 

    filec_aux = '../auxx/aux_first_baroclinic_speed.nc', # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    #filec_aux = None, # if c0==None, auxilliary file to be used as phase velocity field (the spatial interpolation is handled inline)

    name_var_c = {'lon':'lon','lat':'lat','var':'c1'}, # Variable names for the phase velocity auxilliary file 

    cmin = None,

    cmax = None,

    only_diffusion = False,

    path_mdt = None,

    name_var_mdt = {'lon':'','lat':'','mdt':'','mdu':'','mdv':''},

    g = 9.81

)

#################################################################################################################################
# Boundary conditions
#################################################################################################################################

NAME_BC = 'myBC' # For now, only BC_EXT is available

myBC = dict(

    super = 'BC_EXT',

    file = './data/NK/NK_fullNO.nc', # netcdf file(s) in whihch the boundary conditions fields are stored

    name_lon = 'lon',

    name_lat = 'lat',

    name_time = 'time',

    name_var = {'SSH':'adt'}, # name of the boundary conditions variable

    name_mod_var = {'SSH':'adt'},

    dist_sponge = 2, # Peripherical band width (km) on which the boundary conditions are applied

    coast_bc = False, # True= BD applies,  Apply boundary conditions on coastal areas as well (True or None)

)

#################################################################################################################################
# Observation parameters
#################################################################################################################################
NAME_OBS = ['june2025',
            'july2025',
            'august2025',
            #'norkyst_L4'
            ]


norkyst_L4 = dict(

    super = 'OBS_L4',

    path = './data/NK/NK_fullNO.nc',

    name_time = 'time',

    name_lon = 'lon',

    name_lat = 'lat',

    name_var = {
        'SSH': 'adt'
    },

    name_err = {},

    subsampling = None,

    sigma_noise = None
)


june2025 = dict(

    super = 'OBS_SSH_SWATH',

    path = './data/NK/june_geoid_nativ_NK_MDT2025_filtered.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_xac = 'x_ac',

    name_var = {'SSH':'ssha'},
    
    add_mdt = False,

    path_mdt = '../auxx/aux_mdt_hybrid_cnes_cls22_cmems2020_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

    nudging_params_ssh = {'sigma':0,'K':0.1,'Tau':timedelta(days=1)},
    
    #nudging_params_relvort = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},
)

july2025 = dict(

    super = 'OBS_SSH_SWATH',

    path = './data/NK/july_geoid_nativ_NK_MDT2025_filtered.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_xac = 'x_ac',

    name_var = {'SSH':'ssha'},
    
    add_mdt = False,

    path_mdt = '../auxx/aux_mdt_hybrid_cnes_cls22_cmems2020_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

    nudging_params_ssh = {'sigma':0,'K':0.1,'Tau':timedelta(days=1)},
    
    #nudging_params_relvort = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},
)


august2025 = dict(

    super = 'OBS_SSH_SWATH',

    path = './data/NK/august_geoid_nativ_NK_MDT2025_filtered.nc',

    name_time = 'time',
    
    name_lon = 'lon',

    name_lat = 'lat',
    
    name_xac = 'x_ac',

    name_var = {'SSH':'ssha'},
    
    add_mdt = False,

    path_mdt = '../auxx/aux_mdt_hybrid_cnes_cls22_cmems2020_global.nc',

    name_var_mdt = {'lon':'longitude','lat':'latitude','mdt':'mdt'},

    nudging_params_ssh = {'sigma':0,'K':0.1,'Tau':timedelta(days=1)},
    
    #nudging_params_relvort = {'sigma':0,'K':0.7,'Tau':timedelta(days=1)},
)



#################################################################################################################################
# INVERSION
#################################################################################################################################
NAME_INV = 'myINV'

myINV = dict(
    
    super = 'INV_BFN',

    window_size = timedelta(days=10), # length of the bfn time window

    window_output = timedelta(days=3), # length of the output time window, in the middle of the bfn window. (need to be smaller than *bfn_window_size*)

    propagation_timestep = timedelta(hours=3), # propagation time step of the BFN, corresponding to the time step at which the nudging term is computed

    max_iteration = 10, # maximal number of iterations if *bfn_criterion* is not met

    window_overlap = True, # overlap the BFN windows

    criterion = 0.01, # convergence criterion. typical value: 0.01
    
    dist_scale = 10, # distance scale (in km) for the localization of the BFN nudging term
    
    use_bc_as_init = True,
    
    save_obs_proj = True, # save or not the projected observation as pickle format. Set to True to maximize the speed of the algorithm.

    path_save_proj = f'./observations/{name_experiment}/', # path to save projected observations

)


#################################################################################################################################
# Diagnostics
#################################################################################################################################
NAME_DIAG = 'myDIAG'

# Observatory System Experiment (validation with real data)
myDIAG = dict(

    dir_output = None,

    time_min = None,

    time_max = None,

    lon_min = None,

    lon_max = None,

    lat_min = None,

    lat_max = None,

    bin_lon_step = 1,

    bin_lat_step = 1,

    bin_time_step = '1D',

    name_ref = '',

    name_ref_time = '',

    name_ref_lon = '',

    name_ref_lat = '',

    name_ref_var = '',

    options_ref =  {},

    add_mdt_to_ref = False,

    path_mdt = None,

    name_var_mdt = None,
    
    delta_t_ref = 0.9434, # s

    velocity_ref = 6.77, # km/s

    lenght_scale = 1000, # km

    nb_min_obs = 10,

    name_exp_var = '',

    compare_to_baseline = False,

    name_bas = None,

    name_bas_time = None,

    name_bas_lon = None,

    name_bas_lat = None,

    name_bas_var = None

)
