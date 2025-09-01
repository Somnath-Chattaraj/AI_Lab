import time
import os
import sys
import resource
import heapq
from collections import deque
import matplotlib.pyplot as plt

class State:
    def __init__(self, m, c, boat, parent=None, depth=0, cost=0):
        self.m = m  
        self.c = c  
        self.boat = boat  
        self.parent = parent  
        self.depth = depth  
        self.cost = cost  

    def __eq__(self, other):
        return (self.m == other.m and self.c == other.c and self.boat == other.boat)

    def __hash__(self):
        return hash((self.m, self.c, self.boat))

    def __str__(self):
        return f"({self.m}, {self.c}, {'Left' if self.boat == 0 else 'Right'})"


    def __lt__(self, other):
        return (self.cost, self.m, self.c, self.boat) < (other.cost, other.m, other.c, other.boat)


    def is_goal(self):
        return self.m == 0 and self.c == 0 and self.boat == 1

    def get_successors(self):
        successors = []
        if self.boat == 0:  
            possible_moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
            possible_moves = [(-1, 0), (-2, 0), (0, -1), (0, -2), (-1, -1)]
        else:  
            possible_moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

        for m_delta, c_delta in possible_moves:
            new_m = self.m + m_delta
            new_c = self.c + c_delta


            if 0 <= new_m <= 3 and 0 <= new_c <= 3:
                if (new_m == 0 or new_m >= new_c) and ((3 - new_m) == 0 or (3 - new_m) >= (3 - new_c)):
                    new_state = State(new_m, new_c, 1 - self.boat, self, self.depth + 1, self.depth + 1)
                    successors.append(new_state)
        return successors


    def get_heuristic(self):
        return self.m + self.c


def bfs_with_memory(initial_state):
    frontier = deque([initial_state])
    explored = set()
    max_memory = sys.getsizeof(frontier) + sys.getsizeof(explored)
    while frontier:
        state = frontier.popleft()
        if state.is_goal():
            return state, max_memory, len(explored)
        explored.add(state)
        max_memory = max(max_memory, sys.getsizeof(frontier) + sys.getsizeof(explored))
        for successor in state.get_successors():
            if successor not in explored and successor not in frontier:
                frontier.append(successor)
                max_memory = max(max_memory, sys.getsizeof(frontier) + sys.getsizeof(explored))
    return None, max_memory, len(explored)

def dfs_with_memory(initial_state, depth_limit=float('inf')):
    frontier = [initial_state]
    explored = set()
    max_memory = sys.getsizeof(frontier) + sys.getsizeof(explored)
    while frontier:
        state = frontier.pop()
        if state.is_goal():
            return state, max_memory, len(explored)
        explored.add(state)
        max_memory = max(max_memory, sys.getsizeof(frontier) + sys.getsizeof(explored))
        if state.depth < depth_limit:
            for successor in reversed(state.get_successors()):  
                if successor not in explored and successor not in frontier:
                    frontier.append(successor)
                    max_memory = max(max_memory, sys.getsizeof(frontier) + sys.getsizeof(explored))
    return None, max_memory, len(explored)

def dls_with_memory(initial_state, depth_limit):
    return dfs_with_memory(initial_state, depth_limit)

def ids_with_memory(initial_state, max_depth):
    max_memory = 0
    total_explored = 0
    for depth_limit in range(max_depth + 1):
        result, current_memory, explored_count = dls_with_memory(initial_state, depth_limit)
        max_memory = max(max_memory, current_memory)
        total_explored += explored_count 
        if result:
            return result, max_memory, total_explored
    return None, max_memory, total_explored

def ucs_with_memory(initial_state):
    frontier = [(0, initial_state)] 
    explored = set()
    heapq.heapify(frontier)
    max_memory = sys.getsizeof(frontier) + sys.getsizeof(explored)
    while frontier:
        cost, state = heapq.heappop(frontier)
        if state in explored:
            continue
        if state.is_goal():
            return state, max_memory, len(explored)
        explored.add(state)
        max_memory = max(max_memory, sys.getsizeof(frontier) + sys.getsizeof(explored))
        for successor in state.get_successors():
            if successor not in explored:
                heapq.heappush(frontier, (successor.cost, successor))
                max_memory = max(max_memory, sys.getsizeof(frontier) + sys.getsizeof(explored))
    return None, max_memory, len(explored)

def ils_with_memory(initial_state, max_iterations):
    return ids_with_memory(initial_state, max_iterations)


def get_time_usage(start_time):
    return time.time() - start_time

def read_input_file(file_path):
    initial_state_tuple = (3, 3, 0)
    goal_state_tuple = (0, 0, 1)
    dls_depth_limit = 10
    ids_max_depth = 20
    ils_max_iterations = 20

    if not os.path.exists(file_path):
        print(f"Warning: Input file '{file_path}' not found. Using default values.")
        with open(file_path, 'w') as f:
            f.write("initial_state=(3,3,0)\n")
            f.write("goal_state=(0,0,1)\n")
            f.write("dls_depth_limit=10\n")
            f.write("ids_max_depth=20\n")
            f.write("ils_max_iterations=20\n")
        return initial_state_tuple, goal_state_tuple, dls_depth_limit, ids_max_depth, ils_max_iterations


    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("initial_state"):
            initial_state_tuple = tuple(map(int, line.split("=")[1].strip()[1:-1].split(',')))
        elif line.startswith("goal_state"):
            goal_state_tuple = tuple(map(int, line.split("=")[1].strip()[1:-1].split(',')))
        elif line.startswith("dls_depth_limit"):
            dls_depth_limit = int(line.split("=")[1].strip())
        elif line.startswith("ids_max_depth"):
            ids_max_depth = int(line.split("=")[1].strip())
        elif line.startswith("ils_max_iterations"):
            ils_max_iterations = int(line.split("=")[1].strip())

    return initial_state_tuple, goal_state_tuple, dls_depth_limit, ids_max_depth, ils_max_iterations



def run_algorithm(algorithm, initial_state, *args):
    start_time = time.time()
    result = None
    memory_used = 0
    explored_count = 0

    if algorithm == 'bfs':
        result, memory_used, explored_count = bfs_with_memory(initial_state)
    elif algorithm == 'dfs':
        result, memory_used, explored_count = dfs_with_memory(initial_state, *args)
    elif algorithm == 'dls':
        result, memory_used, explored_count = dls_with_memory(initial_state, *args)
    elif algorithm == 'ids':
        result, memory_used, explored_count = ids_with_memory(initial_state, *args)
    elif algorithm == 'ucs':
        result, memory_used, explored_count = ucs_with_memory(initial_state)
    elif algorithm == 'ils':
        result, memory_used, explored_count = ils_with_memory(initial_state, *args)

    time_taken = get_time_usage(start_time)

    return result, time_taken, memory_used, explored_count

def write_results(algorithm, result, time_taken, memory_used, explored_count, output_file):
    with open(output_file, 'a') as f:
        f.write(f"Algorithm: {algorithm.upper()}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Time taken: {time_taken:.6f} seconds\n")
        f.write(f"Memory used: {memory_used} bytes (estimated from data structures)\n")
        f.write(f"Nodes explored: {explored_count}\n")
        f.write("-" * 30 + "\n")

        if result:
            f.write("Path to goal state:\n")
            path = []
            state = result
            while state:
                path.append(str(state))
                state = state.parent
            for i, step in enumerate(reversed(path)):
                f.write(f"Step {i}: {step}\n")
            f.write(f"\nTotal steps: {len(path) - 1}\n")
            f.write("-" * 30 + "\n")
        else:
            f.write("No solution found.\n")
    print(f"Results for {algorithm.upper()} written to {output_file}")


def plot_time_results(results):
    algorithms = [r['algorithm'].upper() for r in results]
    times = [r['time'] for r in results]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, times, color='skyblue')
    plt.xlabel('Algorithms', fontweight='bold')
    plt.ylabel('Time (seconds)', fontweight='bold')
    plt.title('Time Comparison of Search Algorithms', fontweight='bold', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.6f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig("time_comparison.png")
    plt.show()

def plot_memory_results(results):
    algorithms = [r['algorithm'].upper() for r in results]
    memory = [r['memory'] for r in results]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, memory, color='lightgreen')
    plt.xlabel('Algorithms', fontweight='bold')
    plt.ylabel('Memory (bytes)', fontweight='bold')
    plt.title('Memory Comparison of Search Algorithms (Estimated)', fontweight='bold', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig("memory_comparison.png")
    plt.show()


if __name__ == "__main__":
    initial_state_tuple, goal_state_tuple, dls_depth_limit, ids_max_depth, ils_max_iterations = read_input_file('input.txt')

    initial_state = State(*initial_state_tuple)
    results = []
    algorithms = ['bfs', 'dfs', 'dls', 'ids', 'ucs', 'ils']

    for algorithm in algorithms:
        print(f"Running {algorithm.upper()}...")
        args = ()
        if algorithm == 'dfs':
            args = (30,)
        elif algorithm == 'dls':
            args = (dls_depth_limit,)
        elif algorithm == 'ids':
            args = (ids_max_depth,)
        elif algorithm == 'ils':
            args = (ils_max_iterations,)

        result, time_taken, memory_used, explored_count = run_algorithm(algorithm, initial_state, *args)

        output_file = f"log.txt"

        write_results(algorithm, result, time_taken, memory_used, explored_count, output_file)

        results.append({'algorithm': algorithm, 'time': time_taken, 'memory': memory_used, 'nodes': explored_count})
        print("-" * 40)

    plot_time_results(results)
    plot_memory_results(results)