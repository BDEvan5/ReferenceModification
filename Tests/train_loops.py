
from numba.core.decorators import njit
from toy_auto_race.ImitationLearning import BufferIL
import numpy as np
import csv
from matplotlib import pyplot as plt


import ReferenceModification.LibFunctions as lib




def train_vehicle(env, vehicle, steps):
    done = False
    state = env.reset()

    print(f"Starting Training: {vehicle.name}")
    for n in range(steps):
        a = vehicle.plan_act(state)
        s_prime, r, done, _ = env.step(a)

        state = s_prime
        vehicle.agent.train(2)
        
        if done:
            vehicle.done_entry(s_prime)
            env.render(wait=False, name=vehicle.name)

            # vehicle.reset_lap()
            state = env.reset()

    vehicle.print_update(True)
    vehicle.save_csv_data()

    print(f"Finished Training: {vehicle.name}")

if __name__ == "__main__":
    pass
