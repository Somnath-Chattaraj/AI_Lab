import time
import tracemalloc
import copy
import os
import heapq
import csv
import pandas as pd
import matplotlib.pyplot as plt

class TicTacToe:
    def __init__(self, board, player='X', path=None, depth=0, cost=0):
        self.board = board
        self.player = player
        self.path = path or []
        self.depth = depth
        self.cost = cost

    def get_opponent(self):
        return 'O' if self.player == 'X' else 'X'

    def is_goal(self, target_board):
        return self.board == target_board

    def get_successors(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '':
                    new_board = copy.deepcopy(self.board)
                    new_board[i][j] = self.player
                    next_state = TicTacToe(
                        new_board,
                        self.get_opponent(),
                        self.path + [(i, j)],
                        self.depth + 1,
                        self.cost + 1
                    )
                    moves.append(next_state)
        return moves

    def __eq__(self, other):
        return self.board == other.board and self.player == other.player

    def __hash__(self):
        return hash(str(self.board) + self.player)

    def __lt__(self, other):  
        return self.cost < other.cost

def format_result(node, start_time, peak, nodes_expanded, start_player):
    boards = []
    board = [['' for _ in range(3)] for _ in range(3)]
    player = start_player
    for move in node.path:
        i, j = move
        board[i][j] = player
        boards.append(copy.deepcopy(board))
        player = 'O' if player == 'X' else 'X'

    return {
        'solution': node.path,
        'intermediate_boards': boards,
        'depth': node.depth,
        'time': time.time() - start_time,
        'memory': peak / 1024,
        'nodes': nodes_expanded
    }
def save_stats(result, algo_name, filename):
    dirname = os.path.dirname(filename)
    if dirname: 
        os.makedirs(dirname, exist_ok=True)
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Algorithm", "Time (s)", "Memory (KB)", "Depth", "Nodes Expanded"])
        if result:
            writer.writerow([
                algo_name.upper(),
                f"{result['time']:.4f}",
                f"{result['memory']:.2f}",
                result['depth'],
                result['nodes']
            ])
        else:
            writer.writerow([algo_name.upper(), "N/A", "N/A", "N/A", "N/A"])


def load_input(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    start_board = [line.strip().split() for line in lines[:3]]

    target_board = [line.strip().split() for line in lines[4:7]]

    player = 'X'

    for i in range(3):
        for j in range(3):
            if start_board[i][j] in ['.', '-']:
                start_board[i][j] = ''
            if target_board[i][j] in ['.', '-']:
                target_board[i][j] = ''

    return TicTacToe(start_board, player), target_board


def save_output(result, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        if result:
            f.write("Intermediate Steps:\n")
            for step, board in enumerate(result['intermediate_boards']):
                f.write(f"Step {step+1}:\n")
                for row in board:
                    f.write(" ".join(cell if cell else '.' for cell in row) + "\n")
                f.write("\n")

            f.write(f"Path to reach final board: {result['solution']}\n")
            f.write(f"Depth: {result['depth']}\n")
            f.write(f"Time: {result['time']:.4f} sec\n")
            f.write(f"Memory: {result['memory']:.2f} KB\n")
            f.write(f"Nodes Expanded: {result['nodes']}\n")
        else:
            f.write("No solution found.\n")

def bfs(start, target_board):
    return graph_search(start, target_board, mode="bfs")

def dfs(start, target_board):
    return graph_search(start, target_board, mode="dfs")

def dls(start, target_board, limit=9):
    start_time = time.time()
    tracemalloc.start()

    stack = [start]
    nodes_expanded = 0

    while stack:
        node = stack.pop()
        nodes_expanded += 1

        if node.is_goal(target_board):
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return format_result(node, start_time, peak, nodes_expanded, start.player)

        if node.depth < limit:
            for child in reversed(node.get_successors()):
                stack.append(child)

    tracemalloc.stop()
    return None

def ids(start, target_board, max_depth=10):
    for limit in range(max_depth+1):
        result = dls(copy.deepcopy(start), target_board, limit)
        if result:
            return result
    return None


def ils(start, target_board, max_cost=10):
    for cost_limit in range(1, max_cost+1):
        result = ucs(copy.deepcopy(start), target_board, cost_limit)
        if result:
            return result
    return None


def ucs(start, target_board, cost_limit=float('inf')):
    start_time = time.time()
    tracemalloc.start()

    queue = [(0, start)]
    explored = set()
    nodes_expanded = 0

    while queue:
        cost, node = heapq.heappop(queue)
        if node in explored:
            continue
        explored.add(node)
        nodes_expanded += 1

        if node.is_goal(target_board):
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return format_result(node, start_time, peak, nodes_expanded, start.player)

        if cost <= cost_limit:
            for child in node.get_successors():
                heapq.heappush(queue, (child.cost, child))

    tracemalloc.stop()
    return None


def graph_search(start, target_board, mode="bfs"):
    start_time = time.time()
    tracemalloc.start()

    queue = [start]
    explored = set()
    nodes_expanded = 0

    while queue:
        node = queue.pop(0) if mode == "bfs" else queue.pop()
        if node in explored:
            continue
        explored.add(node)
        nodes_expanded += 1

        if node.is_goal(target_board):
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return format_result(node, start_time, peak, nodes_expanded, start.player)

        successors = node.get_successors()
        queue.extend(successors if mode == "bfs" else reversed(successors))

    tracemalloc.stop()
    return None

def main():
    algorithms = {
        'bfs': bfs,
        'dfs': dfs,
        'dls-3': lambda s, t: dls(s, t, limit=3),
        'dls-6': lambda s, t: dls(s, t, limit=6),
        'dls-9': lambda s, t: dls(s, t, limit=9),
        'ids-3': lambda s, t: ids(s, t, max_depth=3),
        'ids-6': lambda s, t: ids(s, t, max_depth=6),
        'ids-9': lambda s, t: ids(s, t, max_depth=9),
        'ils': lambda s, t: ils(s, t, max_cost=9),
        'ucs': ucs,
    }

    start_state, target_board = load_input('021_input.txt')

    for name, func in algorithms.items():
        print(f"\nRunning {name.upper()}")
        result = func(copy.deepcopy(start_state), target_board)
        save_output(result, f'021_output/021_{name}_output.txt')
        save_stats(result, name, "021_stats.csv")
        if result:
            print(f"{name.upper()} completed. Time={result['time']:.4f}s")
        else:
            print(f"{name.upper()} found no solution.")


if __name__ == '__main__':
    main()


df = pd.read_csv("021_stats.csv")

plt.figure(figsize=(6,4))
plt.bar(df["Algorithm"], df["Time (s)"], color="skyblue")
plt.title("Execution Time - Bar Chart")
plt.ylabel("Time (s)")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(df["Algorithm"], df["Time (s)"], marker="o", color="green")
plt.title("Execution Time - Line Chart")
plt.ylabel("Time (s)")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["Nodes Expanded"], df["Time (s)"], color="red")
for i, txt in enumerate(df["Algorithm"]):
    plt.annotate(txt, (df["Nodes Expanded"][i], df["Time (s)"][i]))
plt.title("Time vs Nodes Expanded - Scatter Plot")
plt.xlabel("Nodes Expanded")
plt.ylabel("Time (s)")
plt.show()

plt.figure(figsize=(6,4))
plt.bar(df["Algorithm"], df["Memory (KB)"], color="orange")
plt.title("Memory Usage - Bar Chart")
plt.ylabel("Memory (KB)")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(df["Algorithm"], df["Memory (KB)"], marker="o", color="purple")
plt.title("Memory Usage - Line Chart")
plt.ylabel("Memory (KB)")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["Nodes Expanded"], df["Memory (KB)"], color="blue")
for i, txt in enumerate(df["Algorithm"]):
    plt.annotate(txt, (df["Nodes Expanded"][i], df["Memory (KB)"][i]))
plt.title("Memory vs Nodes Expanded - Scatter Plot")
plt.xlabel("Nodes Expanded")
plt.ylabel("Memory (KB)")
plt.show()