

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
            # env.render(wait=False, name=vehicle.name)

            state = env.reset()

    vehicle.print_update(True)
    vehicle.save_csv_data()

    print(f"Finished Training: {vehicle.name}")

if __name__ == "__main__":
    pass
