# import vehicles
from ReferenceModification.Planners.OraclePlanner import Oracle 
from ReferenceModification.Planners.NavigationPlanner import NavTrainVehicle, NavTestVehicle 
from ReferenceModification.Planners.ModificationPlanner import ModVehicleTrain, ModVehicleTest
from ReferenceModification.Planners.FollowTheGap import ForestFGM

# import utils
from ReferenceModification import LibFunctions as lib 
from Tests.train_loops import * 
from Tests.testing_fcns import *
from ReferenceModification.Simulator.ForestSimulator import ForestSim

# Set name variables
map_name_forest = "forest2"
train_test_n = 7
nav_name_forest = f"Navforest_{train_test_n}"
# train_test_n = 2
mod_name_forest = f"ModForest_{train_test_n}"

repeat_name = f"RepeatTest_{train_test_n}"
eval_name_f= f"ComparisonTest_{train_test_n}"



"""
Training Functions
"""
def train_nav_forest():
    env = ForestSim(map_name_forest)
    vehicle = NavTrainVehicle(nav_name_forest, env.sim_conf, h_size=200)

    # train_vehicle(env, vehicle, 100)
    train_vehicle(env, vehicle, 200000)


def train_mod_forest():
    env = ForestSim(map_name_forest)

    vehicle = ModVehicleTrain(mod_name_forest, map_name_forest, env.sim_conf, load=False, h_size=200)
    # train_vehicle(env, vehicle, 1000)
    train_vehicle(env, vehicle, 200000)


def train_repeatability():
    env = ForestSim(map_name_forest)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"

        vehicle = ModVehicleTrain(train_name, map_name_forest, env.sim_conf, load=False)

        # train_vehicle(env, vehicle, 100)
        train_vehicle(env, vehicle, 200000)

# test forest
def run_comparison_forest():
    env = ForestSim(map_name_forest)
    test = TestVehicles(env.sim_conf, eval_name_f)

    vehicle = NavTestVehicle(nav_name_forest, env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ForestFGM()
    test.add_vehicle(vehicle)

    vehicle = Oracle(env.sim_conf)
    test.add_vehicle(vehicle)

    vehicle = ModVehicleTest(mod_name_forest, map_name_forest, env.sim_conf)
    test.add_vehicle(vehicle)

    test.run_eval(env, 100, True, wait=False)

def test_repeat():
    env = ForestSim(map_name_forest)
    test = TestVehicles(env.sim_conf, repeat_name)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"
        vehicle = ModVehicleTest(train_name, map_name_forest, env.sim_conf)
        test.add_vehicle(vehicle)

    # test.run_eval(env, 1000, False)
    test.run_eval(env, 100, False)





if __name__ == "__main__":
    
    # train_nav_forest()
    # train_mod_forest()

    # train_repeatability()
    
    run_comparison_forest()

    # test_repeat()

