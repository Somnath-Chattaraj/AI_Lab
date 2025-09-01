from collections import deque

def bfs(graph, start_node):
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)

    while queue:
        node = queue.popleft()
        print(node, end=" ")
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def create_graph():
    graph = {}
    num_nodes = int(input("Enter the number of nodes in the graph: "))


    for _ in range(num_nodes):
        node = input("Enter a node: ")
        graph[node] = []

    num_edges = int(input("Enter the number of edges: "))

    for _ in range(num_edges):
        node1, node2 = input("Enter an edge (format: node1 node2): ").split()
        graph[node1].append(node2)
        graph[node2].append(node1)

    return graph


def print_graph(graph):
    print("\nGraph Representation:")
    for node in graph:
        print(f"{node}: {', '.join(graph[node])}")

