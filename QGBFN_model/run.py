# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:13:30 2024

@author: Sara

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

path_config = './config_fullNO.py' # path of the configuration file

import sys
#sys.path.append('./.')

from src import exp
config = exp.Exp(path_config)

from src import state as state
State = state.State(config)

from src import obs as obs
dict_obs = obs.Obs(config,State)

from src import mod as mod
Model = mod.Model(config,State)

from src import bc as bc
Bc = bc.Bc(config)

from src import inv as inv
inv.Inv(config,State,Model,dict_obs=dict_obs,Bc=Bc)

