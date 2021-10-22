import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
import os
import os.path as osp

def load_npz(filepath):
    filepath = osp.abspath(osp.expanduser(filepath))

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    if osp.isfile(filepath):
        with np.load(filepath, allow_pickle=True) as loader:
            loader = dict(loader)
            for k, v in loader.items():
                if v.dtype.kind in {'O', 'U'}:
                    loader[k] = v.tolist()
            return loader
    else:
        raise ValueError(f"{filepath} doesn't exist.")

def load_graph(dataset):
    datasets_links = {'lastfm':'https://www.dropbox.com/s/cslv0z7f3lkrbse/lastfm.npz',
    'cora':'https://www.dropbox.com/s/9u1m5sqrvn60u4k/cora.npz',
    'blogcatalog':'https://www.dropbox.com/s/t829qjpkvwzc7v2/blogcatalog.npz',
    'flickr':'https://www.dropbox.com/s/p5k8a5rpj5bhbtz/flickr.npz'}
    if dataset not in datasets_links:
        raise Exception("Unkonwn dataset")
    else:
        os.system(f'wget {datasets_links[dataset]}')

    print(f'Loading {dataset} dataset...')
    data = load_npz(f'{dataset}.npz')
    graph = nx.from_scipy_sparse_matrix(data['adj'])
    feat = csr_matrix.todense(data['feat'])
    labels = data['label']
    return graph, feat, labels

def similarity_matrix(graph, feat, sim_measures, weights):
    nodes_num = len(graph.nodes)
    sim_matrices = []
    for measure, weight in zip(sim_measures, weights):
        if weight == 0.0:
            continue
        if measure == 'shortest_path':
            sim_matrix = nx.floyd_warshall_numpy(graph)
            sim_matrix = 1 - np.divide(1, sim_matrix, out=np.zeros_like(sim_matrix), where=sim_matrix != 0)
        elif measure == 'sim_rank':
            sim_matrix = 1 - nx.simrank_similarity_numpy(graph, importance_factor=0.9, max_iterations=100, tolerance=0.01)
        elif measure == 'features':
            sim_matrix = pairwise_distances(feat, metric='cosine')
        else:
            raise Exception('Unkonwn similarity measure')
        sim_matrices.append(weight * sim_matrix)
    return sum(sim_matrices)
