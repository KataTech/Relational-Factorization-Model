import numpy as np
import os 
from dev import util
from collections import defaultdict

def gen_cycle_graph(n: int, permute = False): 
        """
        Generates a matrix representing a cycle graph with "n" nodes
        """
        adj_matrix = np.zeros((n, n))
        for i in range(n - 1): 
                adj_matrix[i, i + 1] = 1
                adj_matrix[i + 1, i] = 1
        return adj_matrix 

def gen_star_graph(n: int, permute = False, verbose = False): 
        """
        Generates a matrix representing a star graph with "n" nodes and "source" 
        as the center of the star graph 
        """
        # Determine a sourse node randomly
        source = np.random.randint(n)
        if verbose: 
                print(f"Source Node is {source}\n")
        
        adj_matrix = np.zeros((n, n))
        adj_matrix[:, source] = 1
        adj_matrix[source, :] = 1
        adj_matrix[source, source] = 0
        return adj_matrix 

def random_generate(gen_funcs, max_nodes, num_graphs, permute = False): 
        """
        Randomly generates <num_graphs> with each generator function in <gen_funcs>
        and populate it in a dictionary where the keys are the ID of the generator 
        function and the values is a list containing the matrices 

        Args: 
                - gen_funcs []list(func)]: a list containing K generating functions 
                - max_nodes [int]: the maximum number of nodes any generated graph may contain
                - num_graphs [int]: the total number of graphs to be generated per class 
                - permute [boolean]: whether the generator functions should permute the adj_matrices
        """
        graphs = defaultdict(list)

        # detrermine the number of classes 
        num_classes = len(gen_funcs)
        # pre-generate the number of nodes every graph should have 
        num_nodes = np.random.randint(1, max_nodes, num_graphs * num_classes)
        i = 0 
        # for every class, populate the dictionary at slot value <class_id>
        # with <num_graphs> number of graphs using its associated generator
        for class_id in range(num_classes): 
                generator = gen_funcs[class_id]
                for _ in range(num_graphs): 
                        graphs[class_id].append(generator(num_nodes[i], permute))
                        i += 1
        return graphs 

def graph2text(graph_dict, name, del_existing = False): 
        """
        Given a graph dictionary, an output of random_generate(), writes text files corresponding to 
        the graphs described in the dictionary with format outlined by TUDataset. 
        Link: https://chrsmrrs.github.io/datasets/docs/format/

        Args: 
                - graph_dict [dict(list(numpy arrays))]: a graph dictionary where the keys correspond 
                  to the class labels of the graphs, and the values correspond to a list storing 
                  adjacency matrices that represent graphs of the specified class 
                - name [str]: the dataset name. 
                - del_existing [bool]: determines if we delete existing files
        """
        
        # Set the file names 
        filename_edges = os.path.join(util.DATA_GRAPH_DIR, name, '{}_A.txt'.format(name))
        filename_graphs = os.path.join(util.DATA_GRAPH_DIR, name, '{}_graph_indicator.txt'.format(name))
        filename_labels = os.path.join(util.DATA_GRAPH_DIR, name, '{}_graph_labels.txt'.format(name))

        # Check if the text files with the same filename already exists. If so, print a warning msg and 
        # do nothing for this program. 
        file_exist = os.path.exists(filename_edges) or os.path.exists(filename_graphs) or os.path.exists(filename_labels)
        if file_exist: 
                if not del_existing: 
                        print("WARNING: Detecting at least one file corresponding to input name!")
                        return 
                os.remove(filename_edges); os.remove(filename_graphs); os.remove(filename_labels)
        
        # Open the files for write mode
        edge_f = open(filename_edges, "w"); graph_f = open(filename_graphs, "w"); label_f = open(filename_labels, "w")

        graph_idx = 0
        for label, graphs in graph_dict.items(): 
                for adj_mat in graphs: 
                        # Document the label
                        label_f.write(f"{label}\n")
                        # Extract the edges from the adjacency matrix 
                        for i in range(adj_mat.shape[0]): 
                                for j in range(i + 1, adj_mat.shape[0]):
                                        # Check if (i, j) is an edge 
                                        if adj_mat[i, j] == 1: 
                                                # document edges (both direction b/c undirected) 
                                                edge_f.write(f"{i + 1}, {j + 1}\n")
                                                edge_f.write(f"{j + 1}, {i + 1}\n")
                                                # document the graph type for both edges
                                                graph_f.write(f"{graph_idx + 1}\n") # NOTE: indexing begin at 1 
                                                graph_f.write(f"{graph_idx + 1}\n") # NOTE: indexing begin at 1 
                        # Increment graph index
                        graph_idx += 1
        
        # Close the files 
        edge_f.close(); graph_f.close(); label_f.close()
                                                

                        
                        

