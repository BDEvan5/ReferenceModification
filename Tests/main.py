

from ReferenceModification.Planners.OraclePlanner import Oracle 
from ReferenceModification.Planners.NavigationPlanner import NavTrainVehicle, NavTestVehicle 
from ReferenceModification.Planners.ModificationPlanner import ModVehicleTrain, ModVehicleTest
from ReferenceModification.Planners.FollowTheGap import ForestFGM, TrackFGM 

from ReferenceModification import LibFunctions as lib 
from Tests.train_loops import * 
from Tests.tests import *

import numpy as np

from ReferenceModification.Simulator.ForestSimulator import ForestSim
from ReferenceModification.Simulator.TrackSimulator import TrackSim


map_name_forest = "forest2"
train_test_n = 2
nav_name_forest = f"Navforest_{train_test_n}"
mod_name_forest = f"ModForest_{train_test_n}"

repeat_name = f"RepeatTest_{train_test_n}"
eval_name_f= f"BigTest{train_test_n}"



"""
Training Functions
"""
def train_nav_forest():
    env = ForestSim(map_name_forest)
    vehicle = NavTrainVehicle(nav_name_forest, env.sim_conf, h_size=200)

    # train_vehicle(env, vehicle, 100)
    # train_vehicle(env, vehicle, 500000)


def train_mod_forest():
    env = ForestSim(map_name_forest)

    vehicle = ModVehicleTrain(mod_name_forest, map_name_forest, env.sim_conf, load=False, h_size=200)
    # train_vehicle(env, vehicle, 100)
    train_vehicle(env, vehicle, 200000)


def train_repeatability():
    env = ForestSim(map_name_forest)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"

        vehicle = ModVehicleTrain(train_name, map_name_forest, env.sim_conf, load=False)

        train_vehicle(env, vehicle, 100)
        # train_vehicle(env, vehicle, 200000)

# test forest
def run_comparison_forest():
    env = ForestSim(map_name_forest)
    test = TestVehicles(env.sim_conf, eval_name_f)

    # vehicle = NavTestVehicle(nav_name_forest, env.sim_conf)
    # test.add_vehicle(vehicle)

    vehicle = ForestFGM()
    test.add_vehicle(vehicle)

    vehicle = Oracle(env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name_forest, map_name_forest, env.sim_conf)
    # vehicle = ModVehicleTest("ModForest_nr6", map_name_forest, env.sim_conf)
    test.add_vehicle(vehicle)

    # test.run_eval(env, 1, True)
    test.run_eval(env, 100, False, wait=False)

def test_repeat():
    env = ForestSim(map_name_forest)
    test = TestVehicles(env.sim_conf, repeat_name)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"
        vehicle = ModVehicleTest(train_name, map_name_forest, env.sim_conf)
        test.add_vehicle(vehicle)

    # test.run_eval(env, 1000, False)
    test.run_eval(env, 1000, False)


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


def run_oracle_forest_test():
    env = ForestSim(map_name_forest)
    vehicle = Oracle(env.sim_conf)

    test_single_vehicle(env, vehicle, False, 100, add_obs=True, wait=False)




if __name__ == "__main__":
    
    # train_mod_forest()
    # train_nav_forest()

    # train_repeatability()
    # test_repeat()

    # run_oracle_forest_test()
    run_fgm_forest()
    # run_mod_forest()



