# setting the path of this notebook to the root directory
from constants import ROOT_DIR
import sys
sys.path.append(ROOT_DIR)

import dev.util as util
import methods.FusedGromovWassersteinFactorization as FGWF
import numpy as np
import os
import pickle
import argparse
from methods.DataIO import StructuralDataSampler, structural_data_list
from sklearn.cluster import KMeans

# data settings
names = [util.TEST_NAME]

# model params
num_atoms = 20
size_atoms = num_atoms * [np.random.randint(20, 100)]
ot_method = 'ppa' # either 'ppa' or 'b-admm'
gamma = 1e-1
gwb_layers = 5
ot_layers = 50

# alg. params
size_batch = 250
epochs = 5
lr = 0.25 # learning rate
weight_decay = 0 
shuffle_data = True
zeta = None  # the weight of diversity regularizer
mode = 'fit'
ssl = [0] # percentage of labeled data; ssl = "semi-supervised learning"

# parse command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Verbose", type=bool)
parser.add_argument("-t", "--test", help="Test File")
parser.add_argument("-s", "--save", help="Save Model")
args = parser.parse_args()
verbose = args.verbose
save = args.save
if args.test: 
    names = [args.test]

best_acc = np.zeros((len(ssl), ))
# iterate for all input datasets 
for name in names:
    # read in the dataset from the appropriate pickle files 
    filename_pkl = os.path.join(util.DATA_GRAPH_DIR, name, 'processed_data.pkl')
    graph_data, num_classes = structural_data_list(filename_pkl)
    # verify whether the graph contain node attributes
    if len(graph_data[0]) == 4:
        # if there are no node attributes, then the dimension of embedding 
        # correspond to the dimension of the class labels
        dim_embedding = graph_data[0][2].shape[1]
    else:
        dim_embedding = 1
        
    # sample data for mini-batching 
    data_sampler = StructuralDataSampler(graph_data)
    labels = []
    for sample in graph_data:
        labels.append(sample[-1])
    labels = np.asarray(labels)

    # output debug messages 
    if verbose: 
        print(f"Processing \" {name} \"...")
        print(f"Dimension of Embedding: {dim_embedding} \nNumber of Classes: {num_classes}")

    for i in range(len(ssl)):
        p = ssl[i]
        # if the task is unsupervised 
        if p == 0:
            # set model parameters 
            model = FGWF.FGWF(num_samples=len(graph_data),
                              num_classes=num_classes,
                              size_atoms=size_atoms,
                              dim_embedding=dim_embedding,
                              ot_method=ot_method,
                              gamma=gamma,
                              gwb_layers=gwb_layers,
                              ot_layers=ot_layers,
                              prior=data_sampler)
            # unsupervised training
            model = FGWF.train_usl(model, graph_data,
                                   size_batch=size_batch,
                                   epochs=epochs,
                                   lr=lr,
                                   weight_decay=weight_decay,
                                   shuffle_data=shuffle_data,
                                   zeta=zeta,
                                   mode=mode,
                                   visualize_prefix=os.path.join(util.RESULT_DIR, name))
            model.eval()
            features = model.weights.cpu().data.numpy()
            embeddings = features.T
            # perform k-means clustering based on each graph's learned 
            # embeddings (weights to the graph factors)
            kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=10)
            pred = kmeans.fit_predict(embeddings)
            best_acc[i] = max([1 - np.sum(np.abs(pred - labels)) / len(graph_data),
                               1 - np.sum(np.abs((1 - pred) - labels)) / len(graph_data)])
            if verbose: 
                print(f"Best Accuracy: {best_acc[i]}")
        # if a percentage of the data is labeled... semi-supervised! 
        else:
            model = FGWF.FGWF(num_samples=len(graph_data),
                              num_classes=num_classes,
                              size_atoms=size_atoms,
                              dim_embedding=dim_embedding,
                              ot_method=ot_method,
                              gamma=gamma,
                              gwb_layers=gwb_layers,
                              ot_layers=ot_layers,
                              prior=data_sampler)
            # semi-supervised training 
            model, predictor, best_acc[i] = FGWF.train_ssl(model, graph_data,
                                                           size_batch=size_batch,
                                                           epochs=epochs,
                                                           lr=lr,
                                                           weight_decay=weight_decay,
                                                           shuffle_data=shuffle_data,
                                                           zeta=zeta,
                                                           mode=mode,
                                                           ssl=p,
                                                           visualize_prefix=os.path.join(util.RESULT_DIR, name))
            # save the best_predictor
            if save: 
                FGWF.save_model(predictor, os.path.join(util.MODEL_DIR, 
                                                        '{}_{}_{}_predictor.pkl'.format(name, mode, i)))
            if verbose: 
                print(f"Best Accuracy: {best_acc[i]}")
        # save the FGWF model
        if save: 
            FGWF.save_model(model, os.path.join(util.MODEL_DIR, '{}_{}_{}_fgwf.pkl'.format(name, mode, i)))

    # output and/or save the best accuracy
    if verbose: 
        print(f"Best Accuracy Overall: {best_acc}")
    if save: 
        filename_pkl = os.path.join(util.RESULT_DIR, 'classification_acc_{}_{}.pkl'.format(name, mode))
        with open(filename_pkl, 'wb') as f:
            pickle.dump(best_acc, f)
