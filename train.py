import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def train_dqn(env, main_network, target_network, optimizer, replay_buffer, gamma, epsilon, epsilon_min, epsilon_decay, batch_size, max_episodes, target_update_freq, writer):
    for episode in range(max_episodes):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        total_reward = 0

        for t in range(1000):  # 每个回合最大步数
            # ε-greedy 策略
            if random.random() < epsilon:
                action = env.action_space.sample()  # 随机选择动作
            else:
                q_values = main_network(state_tensor)  # 根据当前状态预测 Q 值
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            # 存储经验
            replay_buffer.push(state, action, reward, next_state, done)

            # 更新状态
            state = next_state
            state_tensor = next_state_tensor
            total_reward += reward

            # 经验回放和网络更新
            if replay_buffer.size() >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones)

                # 计算目标 Q 值
                with torch.no_grad():
                    target_q_values = target_network(next_states)
                    target_q_value = rewards + gamma * torch.max(target_q_values, dim=1)[0] * (1 - dones.float())

                # 计算当前 Q 值
                q_values = main_network(states)
                current_q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # 计算损失
                loss = F.mse_loss(current_q_value, target_q_value)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # 更新目标网络
        if episode % target_update_freq == 0:
            target_network.load_state_dict(main_network.state_dict())

        # ε 衰减
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 记录每一回合的奖励和损失
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Epsilon', epsilon, episode)

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")
