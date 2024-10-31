import random

# Set up parameters
target_position = 10
num_episodes = 50
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.2

# Initialize Q-table
q_table = [[0, 0] for _ in range(target_position + 1)]

# Define action selection using epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < exploration_rate:
        return random.choice([0, 1])
    else:
        return q_table[state].index(max(q_table[state]))

# Game loop for multiple episodes
for episode in range(num_episodes):
    state = 0  # Start at position 0
    steps = 0

    # Initial action choice based on the starting state
    action = choose_action(state)

    while state != target_position:
        # Take action and observe next state
        next_state = state - 1 if action == 0 else state + 1
        next_state = max(0, min(target_position, next_state))  # Ensure within bounds

        # Reward system
        if next_state == target_position:
            reward = 10  # Reached the goal
        else:
            reward = -1  # Penalty for each move

        # Choose next action based on the new state
        next_action = choose_action(next_state)

        # SARSA update rule
        old_value = q_table[state][action]
        next_value = q_table[next_state][next_action]
        q_table[state][action] = old_value + learning_rate * (reward + discount_factor * next_value - old_value)

        # Update state and action
        state = next_state
        action = next_action
        steps += 1

    print(f"Episode {episode + 1}: Goal reached in {steps} steps.")

# Display Q-values for each position on the grid
print("\nQ-table:")
for i, values in enumerate(q_table):
    print(f"Position {i}: Left = {values[0]:.2f}, Right = {values[1]:.2f}")
