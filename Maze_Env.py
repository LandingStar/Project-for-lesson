import gym
from gym import spaces
import random

class MazeEnv(gym.Env):
    def __init__(self, size=(100, 100)):
        super(MazeEnv, self).__init__()
        self.size = size
        self.start = (0, 0)
        self.goal = (size[0] - 1, size[1] - 1)
        self.state = self.start

        # 动作空间：上、下、左、右四个动作
        self.action_space = spaces.Discrete(4)
        
        # 状态空间：二维网格的每个位置
        self.observation_space = spaces.Discrete(self.size[0] * self.size[1])

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # 上
            x = max(0, x - 1)
        elif action == 1:  # 下
            x = min(self.size[0] - 1, x + 1)
        elif action == 2:  # 左
            y = max(0, y - 1)
        elif action == 3:  # 右
            y = min(self.size[1] - 1, y + 1)

        self.state = (x, y)

        # 奖励设计：到达目标时奖励 +1，否则 -0.1
        if self.state == self.goal:
            reward = 1
            done = True
        else:
            reward = -0.1
            done = False
        
        return self.state, reward, done, {}
