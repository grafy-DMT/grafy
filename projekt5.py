import networkx as nx
import matplotlib.pyplot as plt
from graph import *
from graph import random_flow_network


def ford_fulkerson(graph, source, sink):
    flow, path = 0, True

    while path:
        path, reserve = bfs(graph, source, sink)
        flow += reserve

        # increase flow along the path
        for v, u in zip(path, path[1:]):
            if graph.has_edge(v, u):
                graph[v][u]['flow'] += reserve
            else:
                graph[u][v]['flow'] -= reserve

    print(f"Maxiumum flow: {flow}")


def bfs(graph, source, sink):
    undirected = graph.to_undirected()
    explored = {source}
    stack = [(source, 0, dict(undirected[source]))]

    while stack:
        v, _, neighbours = stack[-1]
        if v == sink:
            break

        # search the next neighbour
        while neighbours:
            u, e = neighbours.popitem()
            if u not in explored:
                break
        else:
            stack.pop()
            continue

        # current flow and capacity
        in_direction = graph.has_edge(v, u)
        capacity = e['capacity']
        flow = e['flow']
        neighbours = dict(undirected[u])

        # increase or redirect flow at the edge
        if in_direction and flow < capacity:
            stack.append((u, capacity - flow, neighbours))
            explored.add(u)
        elif not in_direction and flow:
            stack.append((u, flow, neighbours))
            explored.add(u)

    reserve = min((f for _, f, _ in stack[1:]), default=0)
    path = [v for v, _, _ in stack]

    return path, reserve


def set_draw_position_of_flow_network(layers):
    n = len(layers)
    actual_node_pos = 2
    pos = {'1': [0, 0]}
    actual_layer = 0
    distance_between_nodes_in_layer = 1 / (n - 1)
    for vertex in layers:
        actual_layer += 1
        position_at_layer = 0.5
        for i in range(actual_node_pos, actual_node_pos + vertex):
            pos[str(i)] = [actual_layer, position_at_layer]
            position_at_layer -= distance_between_nodes_in_layer
        actual_node_pos += vertex
    actual_layer += 1
    pos[str(actual_node_pos)] = [actual_layer, 0]
    return pos


def draw_graph(graph, layers):
    plt.axis('off')

    layout = set_draw_position_of_flow_network(layers)
    nx.draw_networkx_nodes(graph, layout)
    nx.draw_networkx_edges(graph, layout)
    nx.draw_networkx_labels(graph, layout)

    for u, v, e in graph.edges(data=True):
        label = '{}/{}'.format(e['flow'], e['capacity'])
        x = layout[u][0] * .6 + layout[v][0] * .4
        y = layout[u][1] * .6 + layout[v][1] * .4
        t = plt.text(x, y, label, size=13, horizontalalignment='center', verticalalignment='center')
    plt.show()


def main():
    # AD1 Create random flow network
    number_of_layer = 2
    my_graph, layers = random_flow_network(number_of_layer)
    nodes = [str(x) for x in range(1, my_graph.vertex_count + 1)]
    edges = []
    for i, neighbours in enumerate(my_graph.neighbours_lists):
        for node in neighbours:
            edges.append((str(i + 1), str(node + 1),
                          {'capacity': my_graph.capacity_matrix[i][node], 'flow': my_graph.flow_matrix[i][node]}))

    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(nodes)
    flow_graph.add_edges_from(edges)

    draw_graph(flow_graph, layers)

    # AD2 Ford_Fulkerson Algorithm
    s = '1'
    t = str(len(nodes))
    ford_fulkerson(flow_graph, s, t)
    draw_graph(flow_graph, layers)


if __name__ == "__main__":
    main()
