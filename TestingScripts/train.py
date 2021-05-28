

from ReferenceModification.NavAgents.AgentNav import NavTrainVehicle
from ReferenceModification.NavAgents.AgentMod import ModVehicleTrain

from toy_f110 import ForestSim




def train_vehicle(env, vehicle, steps):
    done = False
    state = env.reset()

    print(f"Starting Training: {vehicle.name}")
    for n in range(steps):
        a = vehicle.plan_act(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.agent.train(2)
        
        # env.render(False)
        
        if done:
            vehicle.done_entry(s_prime)
            # vehicle.show_vehicle_history()
            # env.history.show_history()
            # env.render(wait=False, name=vehicle.name)

            vehicle.reset_lap()
            state = env.reset()

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()

    print(f"Finished Training: {vehicle.name}")



map_name = "forest2"
n = 1
nav_name = f"Navforest_{n}"
mod_name = f"ModForest_{n}"
repeat_name = f"RepeatTest_{n}"
eval_name = f"CompareTest_{n}"

"""
Training Functions
"""
def train_nav():
    env = ForestSim(map_name)
    vehicle = NavTrainVehicle(nav_name, env.sim_conf, h_size=200)

    train_vehicle(env, vehicle, 1000)
    # train_vehicle(env, vehicle, 200000)


def train_mod():
    env = ForestSim(map_name)

    vehicle = ModVehicleTrain(mod_name, map_name, env.sim_conf, load=False, h_size=200)
    train_vehicle(env, vehicle, 1000)
    # train_vehicle(env, vehicle, 200000)



def train_repeatability():
    env = ForestSim(map_name)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"

        vehicle = ModVehicleTrain(train_name, map_name, env.sim_conf, load=False)

        train_vehicle(env, vehicle, 1000)
        # train_vehicle(env, vehicle, 200000)


if __name__ == "__main__":
    train_mod()
    train_nav()
    train_repeatability()
