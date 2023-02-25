import numpy as np
import pylab as pl
import sys
sys.path.append('osr_examples/scripts/')
import environment_2d
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

def check_line_collision(env, x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            if env.check_collision(x, y):
                return False
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            if env.check_collision(x, y):
                return False
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    return True


def path_shortcutting(q, M):
    for rep in range(M):
        t1 = np.random.randint(len(q))
        t2 = np.random.randint(len(q))
        if t1 == t2:
            continue
        if t1 > t2:
            t1, t2 = t2, t1
        if not any(env.check_collision(x, y) for x, y in q[t1:t2+1]):
            #q = q[:t2] + [q[t1]] + q[t1+1:t2] + [q[t2]] + q[t2+1:]
            q = q[:t1+1] + [q[t2]] + q[t2+1:]
    return q

pl.ion()
np.random.seed(4)

env = environment_2d.Environment(10, 6, 5)
pl.clf()
env.plot()

q = env.random_query()
if q is not None:
  x_start, y_start, x_goal, y_goal = q
  env.plot_query(x_start, y_start, x_goal, y_goal)

# Define parameters for PRM algorithm and path shortcutting algorithm
K = 30  # number of nearest neighbors to consider
N = 200 # number of nodes to generate
R = 1.5  # maximum radius to connect neighbors
M = 50

# Initialize arrays to store nodes and edges
nodes = []
edges = []

# Generate N nodes and add them to the environment

nodes.append((x_start, y_start))
nodes.append((x_goal, y_goal))

for i in range(N):
    x = np.random.uniform(0, 10)
    y = np.random.uniform(0, 6)
    if not env.check_collision(x, y):
        nodes.append([x, y])
        plt.plot(x,y,'o')

# Build a KDTree for efficient nearest neighbor search
tree = KDTree(nodes)

# Connect nodes within a certain radius
for i in range(N):
    if i >= len(nodes):
        break
    x1, y1 = nodes[i]
    neighbors = tree.query(np.array([x1, y1]).reshape(1,2), k=K)[1][0]
    for j in neighbors:
        if i != j:
            x2, y2 = nodes[j]
            # calculate distance between nodes
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # check if nodes are within radius and not already connected
            if dist < R and (i, j) not in edges and (j, i) not in edges and not any(env.check_collision(x, y) for x, y in nodes[i:j+1]):
                edges.append((i, j))

# Find path from start to goal using Dijkstra's algorithm
start = np.array([x_start, y_start])
goal = np.array([x_goal, y_goal])
dist = np.full((N,), np.inf)
prev = np.full((N,), -1)
visited = np.full((N,), False)
distances = np.zeros((N, N))

for i, (x1, y1) in enumerate(nodes):
    for j, (x2, y2) in enumerate(nodes):
        if (i, j) in edges or (j, i) in edges:
            distances[i, j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

start_idx = np.where(start)[0][0]
goal_idx = np.where(goal)[0][0]

dist[start_idx] = 0

for i in range(N):
    u = np.argmin(dist)
    visited[u] = True
    if u == goal_idx:
        break
    for j in range(N):
        if not visited[j] and distances[u, j] > 0 and dist[u] + distances[u, j] < dist[j]:
            dist[j] = dist[u] + distances[u, j]
            prev[j] = u
    dist[u] = np.inf

# Extract path from start to goal
path = []
u = goal_idx
while u != -1:
    path.append(nodes[u])
    u = prev[u]

path.reverse()
path_shortcutting(path,M)

# Plot path
for i in range(len(path) - 1):
    x1, y1 = path[i]
    x2, y2 = path[i + 1]
    pl.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

# Plot start and goal points
pl.plot(start[0], start[1], 'bo', markersize=10)
pl.plot(goal[0], goal[1], 'ro', markersize=10)


pl.plot(path)
pl.pause(1000)
