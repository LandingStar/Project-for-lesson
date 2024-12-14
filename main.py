import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Maze_Env import MazeEnv
from Replay_Buffer import ReplayBuffer
from Dqn import DQN
from train import train_dqn

if __name__ == "__main__":
    # 设置迷宫环境
    env = MazeEnv(size=(100, 100))

    # 设置网络和优化器
    state_dim = 2  # 机器人位置（x, y）
    action_dim = 4  # 四个动作（上、下、左、右）
    main_network = DQN(state_dim, action_dim)
    target_network = DQN(state_dim, action_dim)
    target_network.load_state_dict(main_network.state_dict())  # 目标网络初始时和主网络一样
    optimizer = optim.Adam(main_network.parameters(), lr=0.001)

    # 设置训练参数
    replay_buffer = ReplayBuffer(buffer_size=10000)
    gamma = 0.99  # 折扣因子
    epsilon = 1.0  # 初始ε值
    epsilon_min = 0.1
    epsilon_decay = 0.995
    batch_size = 32
    max_episodes = 1000
    target_update_freq = 1000  # 每1000回合更新一次目标网络

    # 设置 TensorBoard 记录
    writer = SummaryWriter(log_dir='./runs/maze_dqn')

    # 训练 DQN
    train_dqn(env, main_network, target_network, optimizer, replay_buffer, gamma, epsilon, epsilon_min, epsilon_decay, batch_size, max_episodes, target_update_freq, writer)
    
    # 关闭 TensorBoard writer
    writer.close()
