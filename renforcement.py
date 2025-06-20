import numpy as np
import gymnasium as gym
from gymnasium import spaces
from errorcalc import calculate_error_graph, calculate_graph_connectivity, error_linear, error_square
from visual import create_graph_visualizer



class GraphEnv(gym.Env):
    def __init__(self, n_nodes=4, dist_threshold=100, nodes=None, targets=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.dist_threshold = dist_threshold
        self.action_space = spaces.Box(low=-100, high=100, shape=(n_nodes, 2), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=600, shape=(n_nodes*2 + n_nodes*2,), dtype=np.float32)
        if nodes is not None and targets is not None:
            self.set_env(nodes, targets)
        else:
            self.reset()

    def reset(self, seed=None, options=None):
        self.nodes = np.random.uniform(100, 500, (self.n_nodes, 2)).astype(np.float32)
        self.targets = np.random.uniform(0, 600, (self.n_nodes, 2)).astype(np.float32)
        return self._get_obs(), {}
    
    def set_env(self, nodes, targets):
        self.nodes = nodes
        self.targets = targets
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.nodes.flatten(), self.targets.flatten()])

    def step(self, action):
        self.nodes += action
        # clip les positions pour rester dans une zone raisonnable
        self.nodes = np.clip(self.nodes, 0, 600)
        error = calculate_error_graph(self.nodes, self.targets, error_linear)
        connectivity = calculate_graph_connectivity(self.nodes, self.dist_threshold)
        reward = -error
        if connectivity < 1:
            reward -= 10  # pénalité si non connexe
        terminated = error < 1e-2 
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

# Pour tester l’environnement
if __name__ == "__main__":

    # create a graph with 4 nodes and 4 targets
    nodes = np.array([[0, 0], [100, 100], [200, 200], [300, 300]]).astype(np.float32)
    targets = np.array([[100, 100], [200, 200], [300, 300], [400, 400]]).astype(np.float32)
    dist_threshold = 100


    # create a graph with 4 nodes and 4 targets
    env = GraphEnv(n_nodes=4, dist_threshold=dist_threshold, nodes=nodes, targets=targets)


    # test the environnement
    obs, _ = env.set_env(nodes.copy(), targets.copy())
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if i%10 == 0:   
            print(f"Reward: {reward}, Terminated: {terminated}")
        if terminated:
            break



    from stable_baselines3 import PPO

    env = GraphEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=30000)
    model.save("ppo_graph_env")

    # testing the model on a graph and visualizing the graph using the visual.py file

    model = PPO.load("ppo_graph_env")

    # test the model on a graph and visualizing the graph using the visual.py file
    env = GraphEnv(n_nodes=4, dist_threshold=dist_threshold, nodes=nodes, targets=targets)
    obs, _ = env.set_env(nodes.copy(), targets.copy())

    # test the model on a graph and visualizing the graph using the visual.py file
    model = PPO.load("ppo_graph_env")
    for i in range(100):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if i%10 == 0:   
            print(f"Reward: {reward}, Terminated: {terminated}")
        if terminated:
            break


    # obs, _ = env.set_env(nodes.copy(), targets.copy())
    # for i in range(100):
    #     action, _ = model.predict(obs)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if i%10 == 0:   
    #         print(f"Reward: {reward}, Terminated: {terminated}")
    #     if terminated:
    #         break



    
