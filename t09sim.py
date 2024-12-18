from sim_class import Simulation
import random

# Initialize the simulation with a specified number of agents
sim = Simulation(num_agents=2, render=False)  # For two robots

# Run the simulation for a specified number of steps
states = []
for i in range(100):
    # Example action: Move joints with specific velocities
    velocity_x = random.uniform(-0.5, 0.5)
    velocity_y = random.uniform(-0.5, 0.5)
    velocity_z = random.uniform(-0.5, 0.5)
    drop_command = random.randint(0, 1)

    actions = [
        [velocity_x, velocity_y, velocity_z, drop_command],
        [velocity_x, velocity_y, velocity_z, drop_command]
    ]

    state = sim.run(actions)
    states.append(state)

print(states[0])

