import numpy as np
import args
import classes
import sys
import matplotlib.pyplot as plt
import math
import random

def import_mazz(path: str) -> classes.set:
    """
    从给定文件路径中导入迷宫数据，并构建相关的状态空间等信息
    """
    global max_column, max_row
    file = open(path, "r")
    n, m = tuple(map(lambda x: int(x), file.readline()[:-1].split(",")))
    max_row, max_column = n, m
    blocks = [classes.block([]) for _ in range(m * n)]
    value = []

    def give_actions(pos: int) -> list:
        r, c = [pos // m, pos % m]
        a = [-1, 0, 1]
        ret = [classes.action(np.array([(r + i) % n * m + (c + j) % m,]), np.array([1])) for i in a for j in a]
        tmp=[(i,j)for i in a for j in a]
        cnt = 0
        for i in a:
            for j in a:
                ret[cnt].action_reward = value[(r + i) % n * m + (c + j) % m]
                ret[cnt].action_value = value[(r + i) % n * m + (c + j) % m]
                ret[cnt].visit_cnt = 1
                ret[cnt].direction = (i, j)
                cnt += 1

        return ret

    for i in range(n):
        lin = file.readline()[:-1].split("\t")
        value += tuple(map(lambda x: np.float64(x), lin))

    for i in range(n * m):
        blocks[i].actions = give_actions(i)
        blocks[i].state_reward = value[i]

    maze_space = classes.set(blocks, n, m)
    maze_space.maxlen = [n, m]
    return maze_space


def plot_maze():
    """
    绘制迷宫的可视化图形，根据要求展示不同元素及方向线段，并添加每个方向的action value显示
    """
    global max_column, max_row
    fig, ax = plt.subplots(figsize=(max_column, max_row))
    for i in range(len(classes.space)):
        y = i // max_column
        x = i % max_column
        state = classes.space[i]
        action_values = [action.action_value for action in state.actions]
        actions = [i for i in state.actions]
        # 找到最大动作值的索引
        max_value_index = np.argmax(action_values)
        a = [-1, 0, 1]
        directions = [(i, j) for i in a for j in a]  # [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1)]
        for j in range(len(directions)):
            # dx, dy = directions[j]
            #k = next(actions[j])
            #_y = k // max_column
            #_x = k % max_column
            #_dx, _dy = _x - x, _y - y
            #if _dx == _dy == 0:
            #    continue
            dx, dy=actions[j].direction
            if dx == dy == 0:
                continue
            dy,dx = dx / math.sqrt(dx * dx + dy * dy), dy / math.sqrt(dx * dx + dy * dy)
            length = (1 + math.tanh(action_values[j])) * 0.1
            # 根据是否是最大动作值来确定颜色
            color = 'red' if j == max_value_index else 'green'
            ax.plot([x + 0.5, x + 0.5 + dx * length], [y + 0.5, y + 0.5 + dy * length], color=color)

            # 添加显示action value的文本，位置稍偏离线段起点一点，方便查看且不重叠
            #text_x = x + 0.5 + 0.05 * dx
            #text_y = y + 0.5 + 0.05 * dy
            #ax.text(text_x, text_y, f"{action_values[j]:.2f}", fontsize=8, color='black')

        if state.state_reward > 0:
            circle = plt.Circle((x + 0.5, y + 0.5), radius=0.1, color='blue')
            ax.add_patch(circle)
        elif state.state_reward < 0:
            rect = plt.Rectangle((x, y), width=1, height=1, alpha=0.8, color='yellow')
            ax.add_patch(rect)
        else:
            rect = plt.Rectangle((x, y), width=1, height=1, alpha=0.8, color='white')
            ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig('actions taken.png')
    return


if __name__ == "__main__":
    max_column = 0
    max_row = 0
    sys.stdout = open("out.txt", "w")
    classes.space = import_mazz("mazz.txt")
    for cnt_round in range(args.run_round):
        epsilon_policy = classes.policy(args.epsilon_greedy(1))

        # epsilon_policy=classes.policy(args.random_choice(5))

        init_state = random.randint(0, len(classes.space) - 1)
        eps = classes.episode(init_state, epsilon_policy, args.episode_lenth)
        # sample_return=classes.space[eps.track[-1][0]].actions[eps.track[-1][1]].action_value
        for ind in range(len(eps.track) - 2, 0, -1):
            s, a = eps.track[ind]

            # sample_return=classes.space[s].actions[a].action_reward+args.gamma*sample_return
            # classes.space[s].actions[a].update(sample_return,lambda x:0.001)
            classes.space[s].actions[a].update(classes.space[s].actions[a].action_reward + args.gamma * max(
                list(map(lambda x: x.action_value, classes.space[eps.track[ind + 1][0]].actions))))
    plot_maze()
    sys.stdout = sys.__stdout__
