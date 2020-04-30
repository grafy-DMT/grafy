#AD5
def random_graph_degree_count(
    vertex_count,
    degree,
    graph_type=AdjacencyMatrix):
    vertices = int(degree*vertex_count/2)
    all_pairs = []
    for idx1 in range(vertex_count):
        for idx2 in range(idx1):
            all_pairs.append((idx1, idx2))

    table_of_degrees = []
    for i in range(vertex_count):
        table_of_degrees.append(degree)

    matrix = np.zeros((vertex_count, vertex_count), dtype=int)
    v=vertices
    i=0
    j=0
    while i < vertices:
        if(j%60==59):
            i=0
            table_of_degrees.clear()
            all_pairs.clear()
            for x in range(vertex_count):
                table_of_degrees.append(degree)
            for idx1 in range(vertex_count):
                for idx2 in range(idx1):
                    all_pairs.append((idx1, idx2))
            matrix = np.zeros((vertex_count, vertex_count), dtype=int)
           
        random.shuffle(all_pairs)
        #print(all_pairs)           POMAGAJĄ ZROZUMIEC DZIALANIE FUNKCJI
        #print(table_of_degrees)    SKUTECZNOŚC 100% ALE TROCHĘ OSZUKANA
        j=j+1
        
        if ((table_of_degrees[all_pairs[0][0]]>0) and (table_of_degrees[all_pairs[0][1]]>0)):
            matrix[all_pairs[0][0]][all_pairs[0][1]] = 1+matrix[all_pairs[0][0]][all_pairs[0][1]]
            matrix[all_pairs[0][1]][all_pairs[0][0]] = 1+matrix[all_pairs[0][1]][all_pairs[0][0]]
            table_of_degrees[all_pairs[0][0]] = table_of_degrees[all_pairs[0][0]]-1 
            table_of_degrees[all_pairs[0][1]] = table_of_degrees[all_pairs[0][1]]-1
            all_pairs.remove(all_pairs[0])
            i=i+1
             
        else:
            random.shuffle(all_pairs)
                

    result = AdjacencyMatrix(matrix)
    return convert(result, graph_type)


def random_regular_graph(
        vertex_count,
        degree,
        graph_type=AdjacencyMatrix):

    if degree*vertex_count%2==1:
        raise ValueError("degree and vertex count must be even")
    if not 0 <= degree < vertex_count:
        raise ValueError("0 <= degree < vertex count must be satisfied")
               
    return random_graph_degree_count(
        vertex_count,
        degree,
        graph_type)


#proj2

print("Losowy graf k-regularny.")
rnd_graph = gp.random_regular_graph(6,4)
gp.draw_graph(rnd_graph)