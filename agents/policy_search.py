import numpy as np
from task import Task

class PolicySearch_Agent():

    def __init__(self, task):
        # environment information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  
            scale=(self.action_range / (2 * self.state_size))) 
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        
        self.total_reward += reward
        self.count += 1

        if done:
            self.learn()

    def act(self, state):
        # action based on given state and policy
        action = np.dot(state, self.w) 
        return action

    def learn(self):
        # random policy search using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  
        
