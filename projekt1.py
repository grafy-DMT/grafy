import graph as gp

print("PROJEKCIK 1 GRAFY")
graph_in_file = gp.read_graph_from_file('projekt1/graph_examples.txt')
graph_type = type(graph_in_file)

if graph_type == gp.AdjacencyMatrix:
    print(gp.convert(graph_in_file, gp.AdjacencyList))
    print(gp.convert(graph_in_file, gp.IncidenceMatrix))
elif graph_type == gp.IncidenceMatrix:
    print(gp.convert(graph_in_file, gp.AdjacencyMatrix))
    print(gp.convert(graph_in_file, gp.AdjacencyList))
elif graph_type == gp.AdjacencyList:
    print(gp.convert(graph_in_file, gp.AdjacencyMatrix))
    print(gp.convert(graph_in_file, gp.IncidenceMatrix))

gp.draw_graph(graph_in_file)

rnd_graph = gp.random_graph(7, 10)
gp.draw_graph(rnd_graph)
rnd1_graph = gp.random_graph(7, edge_probability=0.5)
gp.draw_graph(rnd1_graph)
