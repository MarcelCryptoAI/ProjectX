import numpy as np

class QLearningBot:
    def __init__(self, session, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.session = session  # Pass in the trading session (environment)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Initialize Q-table

    def get_state_key(self, state):
        """Convert state into a tuple (hashable format for Q-table)."""
        return tuple(state)

    def choose_action(self, state):
        """Choose an action using the epsilon-greedy strategy."""
        state_key = self.get_state_key(state)

        # If state is not in Q-table, initialize it
        if state_key not in self.q_table:
            self.q_table[state_key] = { "buy": 0, "sell": 0 }

        # Exploration vs Exploitation
        if np.random.uniform(0, 1) < self.epsilon:
            # Exploration: Choose a random action
            action = np.random.choice(["buy", "sell"])
        else:
            # Exploitation: Choose the action with the highest Q-value
            action = max(self.q_table[state_key], key=self.q_table[state_key].get)

        return action

    def update_q_table(self, current_state, action, reward, next_state):
        """Update Q-table using the Q-learning update rule."""
        current_state_key = self.get_state_key(current_state)
        next_state_key = self.get_state_key(next_state)

        # If next state is not in the Q-table, initialize it
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = { "buy": 0, "sell": 0 }

        # Q-learning update rule
        best_next_action = max(self.q_table[next_state_key], key=self.q_table[next_state_key].get)
        q_value_update = self.alpha * (reward + self.gamma * self.q_table[next_state_key][best_next_action] - self.q_table[current_state_key][action])

        # Update Q-table
        self.q_table[current_state_key][action] += q_value_update

    def calculate_reward(self, price, action, slippage, transaction_fee, volatility):
        """
        Reward function based on trade outcome, slippage, transaction fee, and volatility.
        """
        if action == "buy":
            return price - slippage - transaction_fee - volatility
        elif action == "sell":
            return slippage + transaction_fee + volatility - price
        return 0
