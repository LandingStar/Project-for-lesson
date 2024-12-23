import random
import sys

def two_dimensional_maze(width, height):
    maze = [[-1] * width for i in range(height)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def dfs(x, y):
        maze[y][x] = 0
        random.shuffle(directions)
        for dx, dy in directions:
            new_x, new_y = (x + 2 * dx + width) % width, (y + 2 * dy + height) % height
            if maze[new_y][new_x] == -1 and maze[(y + dy) % height][(x + dx) % width] != 1:
                maze[(y + dy) % height][(x + dx) % width] = 0
                dfs(new_x, new_y)

    target_x, target_y = random.randint(0, width - 1), random.randint(0, height - 1)
    maze[target_y][target_x] = 1
    random.shuffle(directions)
    dx_, dy_ = directions[0]
    new_x_, new_y_ = (target_x + dx_ + width) % width, (target_y + dy_ + height) % height
    dfs(new_x_, new_y_)
    return maze, (target_x, target_y)

r,h=int(input('width:')), int(input('height:'))
maze_, target = two_dimensional_maze(r,h)
sys.stdout=open("C:\\Users\\16329\\Source\\Repos\\LandingStar\\CST-Project\\maze game\\epsilon-greedy\\maze.txt","w")
print(str(h)+","+str(r))
for row in maze_:
    print('\t'.join(map(str, row)))
#print(target)
