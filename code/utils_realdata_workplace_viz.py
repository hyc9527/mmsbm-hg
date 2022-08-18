#!/usr/bin/env python
# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false, reportMissingImports=false

import random
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from palettable.colorbrewer.qualitative import Set2_7
from formatted_logger import formatted_logger
import vizCM as viz


colors = Set2_7.mpl_colors
log = formatted_logger('Workplace-viz', 'info')


def find_number_unique_nodes(edge_lst):
    node_subset = []
    for edge in edge_lst:
        node_subset += edge
    node_subset = list(set(node_subset))
    N = len(node_subset)
    #print(f'number of nodes are {N_subset}')
    return N, node_subset


def viz_workplace_2class(vem_result_path='amrl_output/K2/current_best/res_m788_new.npz', workplace_meta_path='../dataset/contisciani/workplace_meta.csv', isVerbose=False):
    '''
       Pie plot of hypergraph using result from VEM
    '''
    # convert hypergraph dataset from mmsbm format to HyMT format'''
    edgelst = np.load('../data/output/hye_subset.npz', allow_pickle=True)
    edgelst = edgelst['hye_subset']
    card_set = set([len(edge) for edge in edgelst])
    num_edges = len(edgelst)
    num_nodes, node_subset = find_number_unique_nodes(edgelst)
    count_dict = {'2': 0, '3': 0, '4': 0}
    for edge in edgelst:
        if len(edge) == 2:
            count_dict['2'] += 1
        elif len(edge) == 3:
            count_dict['3'] += 1
        elif len(edge) == 4:
            count_dict['4'] += 1
        else:
            pass
    freq_dict = count_dict
    for key in freq_dict:
        freq_dict[key] = count_dict[key] / num_edges
    if isVerbose:
        print('*** Summary on dataset ***')
        print(f'num. nodes is {num_nodes}; num. (hyper)edges are {len(edgelst)}')
        print(f'cardinaliest ranges is  {card_set};')
        print(f'summary on cardinaly counts is {freq_dict}')

    '''
        load estimated parameters from vem on dataset XXX
    '''
    res = np.load(vem_result_path, allow_pickle=True)
    phi = res['phi'][0]

    '''
        permutate phi for better viz
            according to a voting scheme by soft label see test_main.py
    '''
    #utils_mmsbm.swap(phi, 0, 1)   ### swap lablels
    u = {'HyMT': phi}
    A = np.ones(num_edges)
    B = np.zeros((num_nodes, num_edges))
    for e, subset in enumerate(edgelst):
        subset_relabeled = [0] * len(subset)
        for j in range(len(subset)):
            subset_relabeled[j] = node_subset.index(int(subset[j]))
        B[subset_relabeled, e] = 1
    '''
        (i) Load true cluster labels

        (ii) Turn a 5-class labels on departments to 2-class labels on floors
            DISQ, DCMT and SFLE -> ground floor   0,1,3 ->  0
            DSE, SRH  -> first floor  2,4 -> 1
    '''
    df_meta = pd.read_csv(workplace_meta_path)
    groups = df_meta['classID'].values
    floor_labels = np.copy(groups)
    for i, dpt in enumerate(groups):
        if dpt == 0 or dpt == 1 or dpt == 3:
            floor_labels[i] = 0
        else:
            floor_labels[i] = 1
    u['gt'] = np.zeros((num_nodes, 2))  # true department labels: 0(ground), 1(1st floor)
    for i in range(num_nodes):
        u['gt'][i, floor_labels[node_subset[i]]] = 1
    '''
        Visualization:  set up
    '''
    figsize = (9, 9)
    node_size = 0.005
    edge_linewt = 0.5
    edgecolor_edge = 'lightgrey'
    wedge_borderwt = 0.3
    wedgeprops = {
        'edgecolor': edgecolor_edge,
        'linewidth': wedge_borderwt,
        'alpha': 0.8,
    }
    '''
        Visualization: setup
    '''
    vcm = viz.vizCM(A, edgelst, {a: u[a] for a in u.keys()})
    vcm.set_node_attributes(vcm.G)
    G = nx.Graph(vcm.G)
    pos = nx.spring_layout(G)
    K = phi.shape[1]
    from_list = LinearSegmentedColormap.from_list
    cmax = K + 1
    # cm = from_list('tab20b', plt.cm.tab20b(range(0, cmax)), cmax)
    # cm = from_list('tab20', plt.cm.tab20(range(0, cmax)), cmax)
    cm = from_list('Set3', plt.cm.Set3(range(0, cmax)), cmax)
    if isVerbose:
        print(f'cm is {cm}')
        print(f'colors is {colors}')
    radius = node_size  # node size
    plt.figure(figsize=figsize)
    degree = dict(G.degree())
    '''
        Plot mixed membership graph on true labels
    '''
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    nx.draw_networkx_edges(
        G, pos, arrows=False, edge_color=edgecolor_edge, width=edge_linewt
    )
    for n, _ in G.nodes(data=True):
        wedge_sizes, wedge_colors = viz.extract_bridge_properties(
            vcm.nodeName2Id[n], cm, vcm.U['gt'], threshold=1e-10
        )
        if len(wedge_sizes) > 0:
            _, _ = plt.pie(
                wedge_sizes,
                center=pos[n],
                colors=wedge_colors,
                radius=(min(10, degree[n])) * radius,
                wedgeprops=wedgeprops,
                normalize=True,
            )
            ax.axis('equal')
    plt.tight_layout()
    '''
        Plot mixed membership graph on estimated phi
    '''
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    nx.draw_networkx_edges(
        G, pos, arrows=False, edge_color=edgecolor_edge, width=edge_linewt
    )
    for n, _ in G.nodes(data=True):
        wedge_sizes, wedge_colors = viz.extract_bridge_properties(
            vcm.nodeName2Id[n], cm, vcm.U['HyMT'], threshold=1e-10
        )
        if len(wedge_sizes) > 0:
            _, _ = plt.pie(
                wedge_sizes,
                center=pos[n],
                colors=wedge_colors,
                radius=(min(10, degree[n])) * radius,
                wedgeprops=wedgeprops,
                normalize=True,
            )
            ax.axis('equal')
    plt.tight_layout()
    plt.text(
        -3.2,
        1.1,
        'A: true cluster labels',
        weight='bold',
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=9
    )
    plt.text(
        -1,
        1.1,
        'B: estimated mixed cluster labels',
        weight='bold',
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=9
    )
    plt.show()
    #if True:
    #    plt.savefig('tempt_Final_Plot.png', dpi=300, transparent=False, bbox_inches='tight')




def swap(mat, i,j ):
    ''' swap in place the ith and jth column of mat'''
    mat[:, [i, j]] = mat[:, [j,i]]


def label_accuracy_2class(vem_result_path='amrl_output/K2/current_best/res_m788_new.npz', workplace_meta_path='../dataset/contisciani/workplace_meta.csv'):
    '''
        Prdiction accruracy using soft cluster assignment
    '''
    edgelst = np.load("../data/output/hye_subset.npz", allow_pickle=True)
    edgelst = edgelst["hye_subset"]
    node_deg = {}
    for edge in edgelst:
        for node in edge:
            if node in node_deg:
                node_deg[node] += 1
            else:
                node_deg[node] = 1
    popular_nodes = sorted(node_deg, key=node_deg.get, reverse=True)
    '''
        (i)  Load true cluster labels
        (ii) Turn a 5-class labels on departments to 2-class labels on floors
                DISQ, DCMT and SFLE -> ground floor:   0,1,3 ->  0
                DSE, SRH  -> first floor:  2,4 -> 1
    '''
    df_meta = pd.read_csv(workplace_meta_path)
    groups = df_meta['classID'].values
    true_cluster_assignment = np.copy(groups)
    for i, dpt in enumerate(groups):
        if dpt == 0 or dpt == 1 or dpt == 3:
            true_cluster_assignment[i] = 0
        else:
            true_cluster_assignment[i] = 1
    res = np.load(vem_result_path, allow_pickle=True)
    phi = res['phi'][0]
    nodal_clust_accuracy=0.0
    '''
        # swap phi
    print(phi)
    swap(phi,0,1)
    print(phi)
    '''
    nodal_clust_accuracy_corrected_degree=0.0
    for node in popular_nodes:
        #print(f"node {node} ; true cluster {true_cluster_assignment[node]}, soft cluster {phi[node,]}, soft label {np.argmax(phi[node])}, of deg {node_deg[node]}")
        true_label = true_cluster_assignment[node]
        soft_label = np.argmax(phi[node])
        if int(true_label) == int(soft_label):
            nodal_clust_accuracy += 1
            nodal_clust_accuracy_corrected_degree += node_deg[node]
    nodal_clust_accuracy = nodal_clust_accuracy/len(popular_nodes)
    nodal_clust_accuracy_corrected_degree = nodal_clust_accuracy_corrected_degree/len(edgelst)/2
    print(f'nodal cluster accuracy is {nodal_clust_accuracy}')
    print(f'degree corrected nodal cluster accuracy is {nodal_clust_accuracy_corrected_degree}')
    return nodal_clust_accuracy, nodal_clust_accuracy_corrected_degree


def label_accuracy_5class():
    '''
        Prdiction accruracy using soft cluster assignment
    '''
    edgelst = np.load("../data/output/hye_subset.npz", allow_pickle=True)
    edgelst = edgelst["hye_subset"]
    node_deg = {}
    for edge in edgelst:
        for node in edge:
            if node in node_deg:
                node_deg[node] += 1
            else:
                node_deg[node] = 1
    popular_nodes = sorted(node_deg, key=node_deg.get, reverse=True)
    '''
        (i)  Load true cluster labels
        (ii) Turn a 5-class labels on departments to 2-class labels on floors
                DISQ, DCMT and SFLE -> ground floor:   0,1,3 ->  0
                DSE, SRH  -> first floor:  2,4 -> 1
    '''
    df_meta = pd.read_csv(
        '/Users/yichen/Desktop/p2/my-mmsbm-hg-local/experiment3-realdata-workplace/dataset/contisciani/workplace_meta.csv'
    )
    groups = df_meta['classID'].values
    true_cluster_assignment = groups
    res = np.load('amrl_output/K5/res_m500.npz', allow_pickle=True)
    phi = res['phi'][0]
    '''
        swap phi according to func  utils_mmsbm.assign_new_cluster_labels(vote=vote_mat)
    '''
    print('swap 2 and 3')
    swap(phi,2,3)
    print('swap 0 and 4')
    swap(phi,0,4)
    nodal_clust_accuracy=0.0
    nodal_clust_accuracy_corrected_degree=0.0
    for node in popular_nodes:
        #print(f"node {node} ; true cluster {true_cluster_assignment[node]}, soft cluster {phi[node,]}, soft label {np.argmax(phi[node])}, of deg {node_deg[node]}")
        true_label = true_cluster_assignment[node]
        soft_label = np.argmax(phi[node])
        if int(true_label) == int(soft_label):
            nodal_clust_accuracy += 1
            nodal_clust_accuracy_corrected_degree += node_deg[node]
    nodal_clust_accuracy = nodal_clust_accuracy/len(popular_nodes)
    nodal_clust_accuracy_corrected_degree = nodal_clust_accuracy_corrected_degree/len(edgelst)/2
    print(f'nodal cluster accuracy is {nodal_clust_accuracy}')
    print(f'degree corrected nodal cluster accuracy is {nodal_clust_accuracy_corrected_degree}')
    return nodal_clust_accuracy, nodal_clust_accuracy_corrected_degree



def calc_vote_mat_2class(vem_result_path='amrl_output/K2/current_best/res_m788_new.npz', workplace_meta_path='../data/input/workplace_meta.csv'):
    '''
        Compute a voting matrix, each row represents corresponding current cluster's  weight to all clusters. The higher, more probably.
        Output 2-d nparray, shape K by K, K is the number of latent cluster
    '''
    edgelst = np.load("../data/output/hye_subset.npz", allow_pickle=True)
    edgelst = edgelst["hye_subset"]
    print(f"number of edges is {len(edgelst)}")
    node_deg = {}
    for edge in edgelst:
        for node in edge:
            if node in node_deg:
                node_deg[node] += 1
            else:
                node_deg[node] = 1
    print(f"number of nodes is {len(node_deg)}")
    popular_nodes = sorted(node_deg, key=node_deg.get, reverse=True)
    '''
        (i)  Load true cluster labels
        (ii) Turn a 5-class labels on departments to 2-class labels on floors
                DISQ, DCMT and SFLE -> ground floor:   0,1,3 ->  0
                DSE, SRH  -> first floor:  2,4 -> 1
    '''
    df_meta = pd.read_csv(
        workplace_meta_path
        #'../dataset/contisciani/workplace_meta.csv'
    )
    groups = df_meta['classID'].values
    true_cluster_assignment = np.copy(groups)
    for i, dpt in enumerate(groups):
        if dpt == 0 or dpt == 1 or dpt == 3:
            true_cluster_assignment[i] = 0
        else:
            true_cluster_assignment[i] = 1
    #print(true_cluster_assignment)
    res = np.load(vem_result_path, allow_pickle=True)
    phi = res['phi'][0]
    print(f'phi shape is {phi.shape}')
    K=phi.shape[1]
    vote_clust = np.zeros((K, K))
    nodal_clust_accuracy=0.0
    nodal_clust_accuracy_corrected_degree=0.0
    for node in popular_nodes:
        #print(f"node {node} ; true cluster {true_cluster_assignment[node]}, soft cluster {phi[node,]}, soft label {np.argmax(phi[node])}, of deg {node_deg[node]}")
        true_label = true_cluster_assignment[node]
        soft_label = np.argmax(phi[node])
        if int(true_label) == int(soft_label):
            nodal_clust_accuracy += 1
            nodal_clust_accuracy_corrected_degree += node_deg[node]
        vote = node_deg[node]
        vote_clust[true_label][soft_label] += vote
    nodal_clust_accuracy = nodal_clust_accuracy/len(popular_nodes)
    nodal_clust_accuracy_corrected_degree = nodal_clust_accuracy_corrected_degree/len(edgelst)/2
    vote_clust.astype(int)
    print("vote_clust is : ")
    print(vote_clust)
    print(f'nodal cluster accuracy is {nodal_clust_accuracy}')
    print(f'degree corrected nodal cluster accuracy is {nodal_clust_accuracy_corrected_degree}')
    return vote_clust



def calc_vote_mat_5class():
    '''
        Compute a voting matrix, each row represents corresponding current cluster's  weight to all clusters. The higher, more probably.
        Output 2-d nparray, shape K by K, K is the number of latent cluster
    '''
    edgelst = np.load("../data/output/hye_subset.npz", allow_pickle=True)
    edgelst = edgelst["hye_subset"]
    print(f"number of edges is {len(edgelst)}")
    node_deg = {}
    for edge in edgelst:
        for node in edge:
            if node in node_deg:
                node_deg[node] += 1
            else:
                node_deg[node] = 1
    print(f"number of nodes is {len(node_deg)}")
    popular_nodes = sorted(node_deg, key=node_deg.get, reverse=True)
    '''
        (i)  Load true cluster labels
        (ii) Turn a 5-class labels on departments to 2-class labels on floors
                DISQ, DCMT and SFLE -> ground floor:   0,1,3 ->  0
                DSE, SRH  -> first floor:  2,4 -> 1
    '''
    df_meta = pd.read_csv(
        '../dataset/contisciani/workplace_meta.csv'
    )
    groups = df_meta['classID'].values
    true_cluster_assignment = groups
    res = np.load('amrl_output/K5/res_m500.npz', allow_pickle=True)
    phi = res['phi'][0]
    K=phi.shape[1]
    vote_clust = np.zeros((K, K))
    nodal_clust_accuracy=0.0
    nodal_clust_accuracy_corrected_degree=0.0
    for node in popular_nodes:
        #print(f"node {node} ; true cluster {true_cluster_assignment[node]}, soft cluster {phi[node,]}, soft label {np.argmax(phi[node])}, of deg {node_deg[node]}")
        true_label = true_cluster_assignment[node]
        soft_label = np.argmax(phi[node])
        if int(true_label) == int(soft_label):
            nodal_clust_accuracy += 1
            nodal_clust_accuracy_corrected_degree += node_deg[node]
        vote = node_deg[node]
        vote_clust[true_label][soft_label] += vote
    nodal_clust_accuracy = nodal_clust_accuracy/len(popular_nodes)
    nodal_clust_accuracy_corrected_degree = nodal_clust_accuracy_corrected_degree/len(edgelst)/2
    vote_clust.astype(int)
    print("vote_clust is : ")
    print(vote_clust)
    print(f'nodal cluster accuracy is {nodal_clust_accuracy}')
    print(f'degree corrected nodal cluster accuracy is {nodal_clust_accuracy_corrected_degree}')
    return vote_clust



def assign_new_cluster_labels(vote):
    '''
        Given vote matrix from func calc_vote_mat_2class(), define new clusters labels for better vizsualization. Run the func before plotting result from Exp3.

        Arguments:
            2d nparray vote, of shape K by K

        Output:
            a dictionary of cluster assignments. For instance new_pos_dict[i] = j, implying that col i of phi goes to j
    '''
    #vote = np.array([[468., 315.],[314., 527.]])
    K=np.shape(vote)[0]
    rowsum=np.sum(vote, axis=1)
    rowsum.astype(int)
    print(rowsum)
    rowsum_sort=np.sort(rowsum)[::-1]
    print(rowsum_sort)
    ind=0
    new_pos_dict={}
    pos_taken=[]
    pos_flag=np.array([True]*K) # positions not taken yet
    print('Arrange phi columns as follows:')
    while ind<K:
        which_row=rowsum_sort[ind]
        pos=np.where(rowsum==which_row)[0][0]
        vote_row = vote[pos]
        temp_new_pos=np.argmax(vote_row)
        #print(temp_new_pos)
        #print(pos_taken)
        if temp_new_pos not in pos_taken:
            print(f"col {pos} goes to col {temp_new_pos}")
            new_pos_dict[pos] = temp_new_pos
            pos_taken = np.append(pos_taken, temp_new_pos)
            pos_flag[temp_new_pos]=False
        else: # if temp_new_pos is taken
            curr_vote_row =vote_row[pos_flag]
            temp_new_pos=np.argmax(curr_vote_row)
            curr_vote=curr_vote_row[temp_new_pos]
            temp_new_pos = np.where(vote_row == curr_vote)[0]
            if len(temp_new_pos) <2: # only one position is available
                temp_new_pos = temp_new_pos[0]
            else:
                pos_taken=np.where(pos_flag==False)
                temp_new_pos = np.setdiff1d(temp_new_pos, pos_taken)[0]
            #print(temp_new_pos)
            print(f"col {pos} goes to col {temp_new_pos}")
            new_pos_dict[pos] = temp_new_pos
            pos_taken = np.append(pos_taken, temp_new_pos)
            pos_flag[temp_new_pos]=False
        #print(f"flag is {pos_flag}")
        ind+= 1
    new_pos_dict=dict(sorted(new_pos_dict.items()))
    print(f' To sum, the new position dictionary is {new_pos_dict}')

if __name__ == '__main__':
    pass

