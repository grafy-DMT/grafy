import graph as gp
import numpy as np


def hamilton_cycle(v, in_graph, n):
    global stack, visited, test, cycles_found
    stack = np.append(stack, v)

    if stack.shape[0] < n:
        visited[v - 1] = True
        for neighbour in in_graph.neighbours_lists[v - 1]:
            if not visited[neighbour]:
                hamilton_cycle(neighbour + 1, in_graph, n)
        visited[v - 1] = False
    else:
        test = False
        for neighbour in in_graph.neighbours_lists[v - 1]:
            if neighbour == 0:
                test = True
                break
        if test:
            cycles_found += 1
            cycle = "["
            for i in range(stack.shape[0]):
                cycle += str(stack[i]) + " - "
            cycle += str(stack[0]) + "]"
            print(cycle)
    stack = np.delete(stack, stack.shape[0] - 1)


print("PROJEKCIK 2 GRAFY")

# Ad1 Check if sequence
sequence1 = [4, 2, 2, 3, 2, 1, 4, 2, 2, 2, 2]
sequence2 = [4, 4, 3, 1, 2]

print("Sequence1: " + str(gp.is_graph_seq(sequence1)))
print("Sequence2: " + str(gp.is_graph_seq(sequence2)))

# Ad1 Get graph from sequence
graph_from_sequence1 = gp.graph_from_degree_seq(sequence1)
gp.draw_graph(graph_from_sequence1)

# Ad2 Randomize graph
randomized_graph = gp.randomize_graph(graph_from_sequence1, 10)
gp.draw_graph(randomized_graph)

# Ad3 Components
gp.print_components_and_max_component(graph_from_sequence1)

# Ad4 Eulerian Path
eulerian_graph = gp.read_graph_from_file('projekt2/eulerian_graph.txt')
gp.draw_graph(eulerian_graph)
gp.print_eulerian_path(eulerian_graph)

# Ad5 random regular graph
rnd_graph = gp.random_regular_graph(6,4)
gp.draw_graph(rnd_graph)

# Ad6 Hamilton Cycle
graph_in_file = gp.read_graph_from_file('projekt2/ham_ex2.txt')
graph = gp.convert(graph_in_file, gp.AdjacencyList)

# number of hamilton cycles found in a graph
cycles_found = 0
# array of visited vertices
visited = np.zeros(graph.vertex_count, bool)
# storage for hamilton cycle
stack = np.array([], int)
# test if graph is hamiltonian
test = False
print("SZUKANIE CYKLU HAMILTONA")
hamilton_cycle(1, graph, graph.vertex_count)
if cycles_found == 0:
    print("Nie znaleziono cyklu Hamiltona. Podany graf nie jest grafem hamiltonowskim.")
else:
    print("Podany graf jest grafem hamiltonowskim.")
gp.draw_graph(graph)
