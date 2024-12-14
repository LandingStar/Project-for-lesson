
import numpy as np
import random


class DeepQLearning:
    def __init__(self, learning_rate, discount_factor, replay_buffer_size, batch_size, target_update_frequency):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.iteration_count = 0
        # 这里假设main_network和target_network是已经定义好的神经网络模型
        self.main_network = None
        self.target_network = None

    def store_experience(self, state, action, reward, next_state):
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从回放缓冲区中均匀采样一个小批量
        mini_batch = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states = zip(*mini_batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        # 计算目标值
        target_values = rewards + self.discount_factor * np.max(
            self.target_network.predict(next_states), axis=1)

        # 使用小批量更新主网络
        self.main_network.train(states, actions, target_values)

        self.iteration_count += 1
        if self.iteration_count % self.target_update_frequency == 0:
            # 每C次迭代更新目标网络
            self.target_network.set_weights(self.main_network.get_weights())