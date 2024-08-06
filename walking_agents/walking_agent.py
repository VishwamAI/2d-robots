import importlib


def lazy_import(module_name, class_name=None):
    module = importlib.import_module(module_name)
    return getattr(module, class_name) if class_name else module


tf = lazy_import('tensorflow')
layers = lazy_import('tensorflow.keras.layers')
np = lazy_import('numpy')
random = lazy_import('random')
gym = lazy_import('gym')


# Define the neural network model for the agent
def create_model(input_shape, action_space):
    model = tf.keras.Sequential()
    model.add(layers.Dense(24, input_shape=(np.prod(input_shape),),
                           activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_space, activation='linear'))
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model


# Define the agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = np.prod(state_size)  # Flatten the state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = create_model((self.state_size,), action_size)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, size=(6,))
        act_values = self.model.predict(state.reshape(1, -1))
        return np.squeeze(act_values).reshape(6,)


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state.reshape(1, -1))[0]))
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def initialize_environment_and_agent():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    return env, agent, state_size


def train_agent(episodes=1000, batch_size=32):
    env, agent, state_size = initialize_environment_and_agent()

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, "
                      f"e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


if __name__ == "__main__":
    train_agent()
