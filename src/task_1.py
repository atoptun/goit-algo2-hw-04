import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from tabulate import tabulate


def create_graph() -> nx.DiGraph:
    """ 
    Create a directed graph for the logistics network. 
    Returns:
        Directed graph
    """
    G = nx.DiGraph()

    G.add_nodes_from(['Термінал 1', 'Термінал 2', 'Склад 1', 'Склад 2', 'Склад 3', 'Склад 4'])
    
    edges = [
        ('Термінал 1', 'Склад 1', 25),
        ('Термінал 1', 'Склад 2', 20),
        ('Термінал 1', 'Склад 3', 15),
        ('Термінал 2', 'Склад 3', 15),
        ('Термінал 2', 'Склад 4', 30),
        ('Термінал 2', 'Склад 2', 10),
        ('Склад 1', 'Магазин 1', 15),
        ('Склад 1', 'Магазин 2', 10),
        ('Склад 1', 'Магазин 3', 20),
        ('Склад 2', 'Магазин 4', 15),
        ('Склад 2', 'Магазин 5', 10),
        ('Склад 2', 'Магазин 6', 25),
        ('Склад 3', 'Магазин 7', 20),
        ('Склад 3', 'Магазин 8', 15),
        ('Склад 3', 'Магазин 9', 10),
        ('Склад 4', 'Магазин 10', 20),
        ('Склад 4', 'Магазин 11', 10),
        ('Склад 4', 'Магазин 12', 15),
        ('Склад 4', 'Магазин 13', 5),
        ('Склад 4', 'Магазин 14', 10),
    ]
    G.add_weighted_edges_from(edges)
    return G


def show_graph(G: nx.DiGraph):
    """
    Show graph
    Args:
        G: Directed graph
    """
    pos = {
        'Термінал 1': (3, 5),
        'Термінал 2': (11, 5),

        'Склад 1': (5, 7),
        'Склад 2': (9, 7),

        'Склад 3': (5, 3),
        'Склад 4': (9, 3),

        'Магазин 1': (1, 10),
        'Магазин 2': (3, 10),
        'Магазин 3': (5, 10),

        'Магазин 4': (7, 10),
        'Магазин 5': (9, 10),
        'Магазин 6': (11, 10),

        'Магазин 7': (1, 0),
        'Магазин 8': (3, 0),
        'Магазин 9': (5, 0),

        'Магазин 10': (7, 0),
        'Магазин 11': (9, 0),
        'Магазин 12': (11, 0),
        'Магазин 13': (13, 0),
        'Магазин 14': (15, 0),
    }

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

    plt.show()


def bfs(capacity_matrix, flow_matrix, source, destination, parent):
    """
    Function to find an augmenting path from source to destination in the residual graph using Breadth-First Search (BFS).

    Args:
        capacity_matrix: The matrix representing edge capacities (C).
        flow_matrix: The matrix representing the current flow on edges (F).
        source: The index of the source node.
        destination: The index of the destination node.
        parent: An array/list to store the path (parent[v] stores the node preceding v in the path).

    Returns:
        True if an augmenting path is found, False otherwise.
    """
    visited = [False] * len(capacity_matrix)
    queue = deque([source])
    visited[source] = True

    while queue:
        current_node = queue.popleft()
        
        for neighbor in range(len(capacity_matrix)):
            # Перевірка, чи є залишкова пропускна здатність у каналі
            if not visited[neighbor] and capacity_matrix[current_node][neighbor] - flow_matrix[current_node][neighbor] > 0:
                parent[neighbor] = current_node
                visited[neighbor] = True
                if neighbor == destination:
                    return True
                queue.append(neighbor)
    
    return False


def edmonds_karp(capacity_matrix, source, destination):
    """
    Main function to compute the maximum flow from source to destination using the Edmonds-Karp algorithm.

    Args:
        capacity_matrix: The matrix representing edge capacities.
        source: The index of the source node.
        destination: The index of the destination node.
        parent: An array/list to store the path (parent[v] stores the node preceding v in the path).

    Returns:
        The maximum flow from source to destination.
    """
    num_nodes = len(capacity_matrix)
    flow_matrix = [[0] * num_nodes for _ in range(num_nodes)]  # Init flow matrix to zero
    parent = [-1] * num_nodes
    max_flow = 0
    route = []

    # While there is an augmenting path, add flow
    while bfs(capacity_matrix, flow_matrix, source, destination, parent):
        # Find the minimum capacity along the found path (bottleneck)
        path_flow = float('Inf')
        current_node = destination

        while current_node != source:
            previous_node = parent[current_node]
            path_flow = min(path_flow, capacity_matrix[previous_node][current_node] - flow_matrix[previous_node][current_node])
            current_node = previous_node

        # Update flow along the path, considering reverse flow
        current_node = destination
        while current_node != source:
            route.append(current_node)
            previous_node = parent[current_node]
            flow_matrix[previous_node][current_node] += path_flow
            flow_matrix[current_node][previous_node] -= path_flow
            current_node = previous_node
        route.append(source)

        # Збільшуємо максимальний потік
        max_flow += path_flow

    return max_flow, route[::-1]


def main():
    G = create_graph()

    nodes: list[str] = list(G.nodes)
    terminals = [i for i, node in enumerate(nodes) if node.startswith('Термінал')]
    stores = [i for i, node in enumerate(nodes) if node.startswith('Магазин')]

    # capacity_matrix = nx.to_numpy_array(G, nodelist=nodes, weight='weight')
    adj_matrix = nx.adjacency_matrix(G, nodelist=nodes, weight='weight', dtype=int)
    capacity_matrix = adj_matrix.todense()

    results = []
    result_terminals = []

    for t in terminals:
        t_sum = 0
        for s in stores:
            flow, route = edmonds_karp(capacity_matrix, t, s)
            route = [nodes[i] for i in route]
            results.append((nodes[t], nodes[s], flow, ' -> '.join(route[1:-1])))
            t_sum += flow
        result_terminals.append((nodes[t], t_sum))

    print('Nodes:', ', '.join([f'{i} - {node}' for i, node in enumerate(nodes)]), '\n')
    print('Capacity Matrix:\n', capacity_matrix, '\n')

    headers = ['Термінал', 'Магазин', 'Макс. потік', 'Шлях через']
    print(tabulate(results, headers=headers, numalign='decimal', tablefmt='pipe')) 
    print()
    print(tabulate(result_terminals, headers=['Термінал', 'Всього'], numalign='decimal', tablefmt='pipe'))

    show_graph(G)


if __name__ == '__main__':
    main()
