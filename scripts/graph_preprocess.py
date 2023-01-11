# setting the path of this notebook to the root directory
from constants import ROOT_DIR
import sys
sys.path.append(ROOT_DIR)

import dev.util as util
import os
import pickle

# Process text files representing graphs without node attributes in the format 
# outlined by https://chrsmrrs.github.io/datasets/docs/format/.

# TODO: specify the folder of the dataset to use. 
name_datasets = [util.TEST_NAME] 

for name in name_datasets:
    # <DATA>_A.txt: sparse adjacency matrix representation. Every line represents an edge 
    # within a graph in the form of (node_id, node_id). 
    # Line count: m, the total number of edges from  all the graphs in the dataset. 
    filename_edges = os.path.join(util.DATA_GRAPH_DIR, name, '{}_A.txt'.format(name))
    # <DATA>_graph_indicator.txt: graph_id for every node_id. Indicates what graph a particular node_ID 
    # represents. The numbering of node_ID does not restart for new graphs. 
    # Line count: n, the total number of nodes from all the graphs in the dataset. 
    filename_graphs = os.path.join(util.DATA_GRAPH_DIR, name, '{}_graph_indicator.txt'.format(name))
    # <DATA>_graph_labels.txt: graph_label for every graph. Indicates what class a graph belongs to. 
    # Line count: N, the total number of graphs in the dataset.
    filename_labels = os.path.join(util.DATA_GRAPH_DIR, name, '{}_graph_labels.txt'.format(name))

    # populate a m-sized list of edges indicated by node-ID
    with open(filename_edges) as f:
        edges = f.readlines()
    for n in range(len(edges)):
        edge = edges[n].strip("\n")
        edges[n] = [float(node) for node in edge.split(',')]
    print("Edges:", edges[0:5])

    # populate a n-sized list corresponding to graph-ID of nodes
    with open(filename_graphs) as f:
        graphs = f.readlines()
    for n in range(len(graphs)):
        graph = int(graphs[n].strip("\n"))
        graphs[n] = graph
    print("Graphs:", graphs[0:5])

    # populate a N-sized list corresponding to class-ID of graphs 
    with open(filename_labels) as f:
        labels = f.readlines()
    for n in range(len(labels)):
        label = int(labels[n].strip("\n"))
        labels[n] = label
    print("Labels:", labels[0:5])

    # populate a mapping from label to the graph idx
    label2idx = {}
    idx = 0
    for n in range(len(labels)):
        if labels[n] not in label2idx.keys():
            label2idx[labels[n]] = idx
            idx += 1
    num_class = len(label2idx)

    # populate a mapping from graph ID to node ID, and vice versa
    graph2node = {}
    node2graph = {}
    # iterate over the list of nodes with their corresponding graph
    for i in range(len(graphs)):
        # correct for node_id starting index = 1 instead of 0
        node_id = i + 1 
        graph_id = graphs[i]
        node2graph[node_id] = graph_id
        if graph_id not in graph2node.keys():
            graph2node[graph_id] = {}
        # node-indexing is specific to graph, re-order 
        # the node-ids for every graph from 0, 1, ..., n_G
        idx = len(graph2node[graph_id])
        graph2node[graph_id][node_id] = idx

    # populate a mapping of graph to the size of the node and 
    # calculate the average node size 
    graph2size = {}
    ave_node_size = 0
    for graph_id in graph2node.keys():
        graph2size[graph_id] = len(graph2node[graph_id])
        ave_node_size += len(graph2node[graph_id])
    ave_node_size /= len(graph2node)

    # populate a mapping of graph to edges based on the 
    # re-ordered node-ids 
    graph2edge = {}
    for m in range(len(edges)):
        src = edges[m][0]
        dst = edges[m][1]
        graph_id = node2graph[src]
        src_id = graph2node[graph_id][src]
        dst_id = graph2node[graph_id][dst]
        if graph_id not in graph2edge.keys():
            graph2edge[graph_id] = [[src_id, dst_id]]
        else:
            graph2edge[graph_id].append([src_id, dst_id])

    # compute the average number of edges per graph
    ave_edge_size = 0
    for graph_id in graph2edge.keys():
        ave_edge_size += len(graph2edge[graph_id])
    ave_edge_size /= len(graph2edge)

    # populate a mapping from the graph_ID to class_ID
    graph2label = {}
    for n in range(len(labels)):
        graph_id = n + 1
        graph2label[graph_id] = label2idx[labels[n]]

    # display summary message of the dataset
    # NOTE: graph2edge, graph2label, graph2size should have equal lengths
    print('{}: {}/{}/{} graphs, {} classes, {:.2f} nodes + {:.2f} edges per graph'.format(
        name, len(graph2edge), len(graph2label), len(graph2size), num_class, ave_node_size, ave_edge_size))

    # save dataset in the following format: 4-length array where the first 3 slots contain 
    # N-sized arrays corresponding to the edges, labels, and sizes of every graph. 
    # the last slot contains the number of classes for this dataset. 
    graph_edges = []
    graph_labels = []
    graph_sizes = []
    for key in graph2edge:
        graph_edges.append(graph2edge[key])
        graph_labels.append(graph2label[key])
        graph_sizes.append(graph2size[key])

    filename_pkl = os.path.join(util.DATA_GRAPH_DIR, name, 'processed_data.pkl')
    with open(filename_pkl, 'wb') as f:
        pickle.dump([graph_edges, graph_sizes, graph_labels, num_class], f)







