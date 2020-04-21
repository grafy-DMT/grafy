import numpy as np
from graph import *
from graph import matrix_to_string, draw_graph

d = 0
p = 0
BIGINT = 922337203


def is_connected(graph, v=1):
    visited = np.zeros(graph.vertex_count, bool)
    stack = np.array([], int)
    stack = np.append(stack, v)
    vertex_count = 0
    visited[v - 1] = True

    while stack.shape[0] != 0:
        vertex = stack[stack.shape[0] - 1]
        stack = np.delete(stack, stack.shape[0] - 1)
        vertex_count += 1
        for neighbour in graph.neighbours_lists[vertex - 1]:
            if not visited[neighbour]:
                visited[neighbour] = True
                stack = np.append(stack, neighbour + 1)

    if vertex_count == graph.vertex_count:
        return True
    else:
        return False


def random_connected_graph():
    connected = False

    while not connected:
        graph = random_graph(7, 10)
        graph = convert(graph, AdjacencyList)
        if is_connected(graph):
            connected = True

    return graph


def relax(u, v, graph):
    global d, p
    if d[v] > (d[u] + graph.weights_matrix[u][v]):
        d[v] = d[u] + graph.weights_matrix[u][v]
        p[v] = u


def dijkstra(graph, v=1):
    global d, p
    n = graph.vertex_count
    d = np.zeros(n, int)
    p = np.zeros(n, int)
    Q = np.zeros(n, bool)
    for vertex in range(n):
        d[vertex] = BIGINT
        p[vertex] = -1
        Q[vertex] = False
    d[v - 1] = 0

    S = np.array([], int)

    while S.shape[0] != n:
        j = 0
        while Q[j]:
            j += 1
        u = j
        while j < n:
            if not Q[j] and (d[j] < d[u]):
                u = j
            j += 1
        Q[u] = True
        S = np.append(S, u)
        for neighbour in graph.neighbours_lists[u]:
            if neighbour not in S:
                relax(u, neighbour, graph)

    print("START: s = " + str(v))
    ptr = 0
    for i in range(n):
        row = ""
        row += "d(" + str(i + 1) + ") = " + str(d[i]) + " ==> "
        row += "["
        j = i
        while j > -1:
            S[ptr] = j
            ptr += 1
            j = p[j]
        while ptr > 0:
            ptr -= 1
            row += str(S[ptr] + 1)
            if ptr != 0:
                row += " - "
        row += "]"
        print(row)


def main():
    print("PROJEKCIK 3 GRAFY")

    graph = random_connected_graph()
    graph = WeightedGraph(graph)
    dijkstra(graph)
    draw_graph(graph)


if __name__ == "__main__":
    main()
