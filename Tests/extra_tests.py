


from ReferenceModification.Planners.OraclePlanner import Oracle 
from ReferenceModification.Planners.NavigationPlanner import NavTrainVehicle, NavTestVehicle 
from ReferenceModification.Planners.ModificationPlanner import ModVehicleTrain, ModVehicleTest
from ReferenceModification.Planners.FollowTheGap import ForestFGM 

from ReferenceModification import LibFunctions as lib 
from Tests.train_loops import * 
from Tests.testing_fcns import *

import numpy as np

from ReferenceModification.Simulator.ForestSimulator import ForestSim


map_name_forest = "forest2"
train_test_n = 4
nav_name_forest = f"Navforest_{train_test_n}"
# train_test_n = 2
mod_name_forest = f"ModForest_{train_test_n}"



def run_fgm_forest():
    sim_conf = lib.load_conf("fgm_config")
    env = ForestSim(map_name_forest, sim_conf)
    vehicle = ForestFGM()

    test_single_vehicle(env, vehicle, False, 100, add_obs=True)

def run_mod_forest():
    sim_conf = lib.load_conf("std_config")
    env = ForestSim(map_name_forest, sim_conf)
    vehicle = ModVehicleTest(mod_name_forest, map_name_forest, env.sim_conf)

    test_single_vehicle(env, vehicle, True, 100, add_obs=True, wait=False)

def run_nav_forest():

    sim_conf = lib.load_conf("std_config")
    env = ForestSim(map_name_forest, sim_conf)
    vehicle = NavTestVehicle(nav_name_forest, sim_conf)

    test_single_vehicle(env, vehicle, True, 100, add_obs=True, wait=False)

def run_oracle_forest_test():
    env = ForestSim(map_name_forest)
    vehicle = Oracle(env.sim_conf)

    test_single_vehicle(env, vehicle, False, 100, add_obs=True, wait=False)



if __name__ == "__main__":
    
    # run_oracle_forest_test()
    # run_fgm_forest()
    run_mod_forest()
    # run_nav_forest()
