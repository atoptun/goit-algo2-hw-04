import networkx as nx
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from tabulate import tabulate
from pprint import pprint


def create_graph() -> 'nx.DiGraph[str]':
    """ 
    Create a directed graph for the logistics network.

    Returns:
        Directed graph
    """
    G = nx.DiGraph()
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


def bfs(
        capacity_matrix: npt.NDArray[np.float64],
        flow_matrix: npt.NDArray[np.float64],
        source: int,
        destination: int,
        parent: list[int]
    ) -> bool:
    """
    Function to find an augmenting path from source to destination in the residual graph using BFS.

    Args:
        capacity_matrix: The matrix representing edge capacities.
        flow_matrix: The matrix representing the current flow on edges.
        source: The index of the source node.
        destination: The index of the destination node.
        parent: A list to store the path.

    Returns:
        True if an augmenting path is found, False otherwise.
    """

    visited = [False] * len(capacity_matrix)
    queue = deque([source])
    visited[source] = True
    parent[source] = -1

    while queue:
        current_node = queue.popleft()
        for neighbor in range(len(capacity_matrix)):
            residual_capacity = capacity_matrix[current_node][neighbor] - \
                flow_matrix[current_node][neighbor]
            if not visited[neighbor] and residual_capacity > 0:
                parent[neighbor] = current_node
                visited[neighbor] = True
                if neighbor == destination:
                    return True
                queue.append(neighbor)

    return False


def edmonds_karp(
        capacity_matrix: npt.NDArray[np.float64],
        source: int, 
        destination: int
    ) -> tuple[float, npt.NDArray[np.float64]]:
    """
    Main function to compute the maximum flow from source to destination using the Edmonds-Karp algorithm.

    Args:
        capacity_matrix: The matrix representing edge capacities.
        source: The index of the source node.
        destination: The index of the destination node.
        parent: A list to store the path.

    Returns:
        The maximum flow from source to destination.
        The flow matrix representing the flow on each edge.
    """
    num_nodes = len(capacity_matrix)
    flow_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    parent = [-1] * num_nodes
    max_flow = 0

    # While there is an augmenting path, add flow
    while bfs(capacity_matrix, flow_matrix, source, destination, parent):
        # Find the minimum capacity along the found path (bottleneck)
        path_flow = float('Inf')
        current_node = destination

        while current_node != source:
            previous_node = parent[current_node]
            residual_capacity = capacity_matrix[previous_node][current_node] - \
                flow_matrix[previous_node][current_node]
            path_flow = min(path_flow, residual_capacity)
            current_node = previous_node

        # Update flow along the path, considering reverse flow
        current_node = destination
        while current_node != source:
            previous_node = parent[current_node]
            flow_matrix[previous_node][current_node] += path_flow
            flow_matrix[current_node][previous_node] -= path_flow
            current_node = previous_node
        max_flow += path_flow

    return max_flow, flow_matrix


def find_path_dfs(
        source: int, 
        destinations: set[int], 
        flow_matrix: npt.NDArray[np.float64], 
        visited: set[int]
    ) -> list[int] | None:
    """
    Find a path with flow > 0 from source to one of destinations.
    Uses DFS.

    Args:
        source: Source node index.
        destinations: Set of destination node indices.
        flow_matrix: The flow matrix.
        visited: Set of visited node indices.
    
    Returns:
        A list of node indices representing the path, or None if no path is found.
    """
    visited.add(source)

    if source in destinations:
        return [source]

    for node_idx in range(len(flow_matrix)):
        if flow_matrix[source][node_idx] > 0 and node_idx not in visited:
            path_suffix = find_path_dfs(node_idx, destinations, flow_matrix, visited)
            if path_suffix:
                return [source] + path_suffix

    return None


def decompose_flow(
        flow_matrix: npt.NDArray[np.float64], 
        sources: set[int],
        destinations: set[int]
    ) -> dict[tuple[int, int], float]:
    """
    Decompose the overall flow into Sources -> Destinations paths.

    Args:
        flow_matrix: The flow matrix obtained from the Edmonds-Karp algorithm.
        sources: Set of source node indices.
        destinations: Set of destination node indices.

    Returns:
        A dictionary with keys as (source, destination) and values as the total flow between them.
    """
    flow_copy = flow_matrix.copy()
    decomposed_paths = []

    while True:
        path_inds = None

        # Looking for any path from any source to any destination
        for s_idx in sources:
            visited = set()
            path_inds = find_path_dfs(s_idx, destinations, flow_copy, visited)
            if path_inds:
                break

        # No more flows, finish decomposition
        if not path_inds:
            break  

        # Find the bottleneck on the found path
        bottleneck = float('Inf')
        for i in range(len(path_inds) - 1):
            u = path_inds[i]
            v = path_inds[i+1]
            bottleneck = min(bottleneck, flow_copy[u][v])

        # Decrease the flow along the path by the bottleneck
        for i in range(len(path_inds) - 1):
            u = path_inds[i]
            v = path_inds[i+1]
            flow_copy[u][v] -= bottleneck

        # Save the decomposed path
        decomposed_paths.append((path_inds[0], path_inds[-1], bottleneck))

    # Aggregate results
    aggregated_flows = defaultdict(float)
    for s, d, flow in decomposed_paths:
        aggregated_flows[(s, d)] += flow

    return aggregated_flows


def main():
    G_orig = create_graph()
    nodes_orig = list(G_orig.nodes)

    node_start = 'START'
    node_end = 'END'
    G_ek = G_orig.copy()
    G_ek.add_node(node_start)
    G_ek.add_node(node_end)

    terminals = [n for n in nodes_orig if n.startswith('Термінал')]
    stores = [n for n in nodes_orig if n.startswith('Магазин')]

    for t in terminals:
        G_ek.add_edge(node_start, t, weight=1000)
    for s in stores:
        G_ek.add_edge(s, node_end, weight=1000)

    nodes_ek: list[str] = list(G_ek.nodes)
    node_map_ek = {i: name for i, name in enumerate(nodes_ek)}
    print("Node Map:")
    pprint(node_map_ek)

    capacity_matrix = nx.to_numpy_array(G_ek, nodelist=nodes_ek, weight='weight', dtype=int) # type: ignore
    print("\nCapacity Matrix:\n")
    pprint(capacity_matrix)

    start_idx = [k for k, v in node_map_ek.items() if v == node_start][0]
    end_idx = [k for k, v in node_map_ek.items() if v == node_end][0]
    terminal_inds = set(k for k, v in node_map_ek.items() if v in terminals)
    store_inds = set(k for k, v in node_map_ek.items() if v in stores)

    max_flow, flow_matrix = edmonds_karp(capacity_matrix, start_idx, end_idx)
    print("\nFlow Matrix:\n")
    pprint(flow_matrix)

    dec_flows = decompose_flow(flow_matrix, terminal_inds, store_inds)

    # Prepare table data for display
    table_data = []
    terminal_totals = defaultdict(float)
    store_totals = defaultdict(float)
    for t in terminal_inds:
        for s in store_inds:
            flow = dec_flows[(t, s)]
            table_data.append([node_map_ek[t], node_map_ek[s], flow])
            terminal_totals[node_map_ek[t]] += flow
            store_totals[node_map_ek[s]] += flow

    print("-"*40)
    print(f"Загальний потік мережі: {max_flow} одиниць\n")

    print("Розподіл потоків 'Термінал -> Магазин':\n")
    headers = ['Термінал', 'Магазин', 'Фактичний Потік (одиниць)']
    print(tabulate(table_data, headers=headers, numalign='decimal', tablefmt='pipe'))

    print("\nПідсумок по терміналах:\n")
    headers = ['Термінал', 'Відправлено (одиниць)']
    print(tabulate([[t, total] for t, total in terminal_totals.items()], 
                   headers=headers, numalign='decimal', tablefmt='pipe'))
    print("-"*40)

    print("\nПідсумок по магазинах:\n")
    headers = ['Магазин', 'Отримано (одиниць)']
    store_totals = dict(sorted(store_totals.items(), key=lambda item: item[1]))
    print(tabulate([[s, total] for s, total in store_totals.items()], 
                   headers=headers, numalign='decimal', tablefmt='pipe'))
    print("-"*40)

    show_graph(G_orig)


if __name__ == '__main__':
    main()
