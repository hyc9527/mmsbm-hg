# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=false


import itertools
import numpy as np
import random
from formatted_logger import formatted_logger


log = formatted_logger('MMSB-general-hg-workplace', 'info')


def gen_singleton_B(b0, b1, h0, num_cluster):
    '''
    Generate a h0-dim singleton B such that
        B[k_1,k_2,...,k_h0] = b_0 if all equal to each other;
        B[k_1, ...,k_h0] = b_1 else;
    Arguments:
        float b0;
        flat b1;
        int h0 len of edge
        num_cluster K, number of cluster
    Output:
        h0-d array, a base B, of shape(K,K,....,K)
    '''
    B = np.ones(num_cluster**h0)
    shape = tuple(np.array([num_cluster] * h0))
    B = np.reshape(B, shape)
    K_ranges = [range(0, num_cluster)] * (h0)
    K_power = [
        x for x in itertools.product(*K_ranges)
    ]  # each row is (k_1, k_2, ..., k_{h0})
    for h0_seq in K_power:
        if all(x == h0_seq[0] for x in h0_seq):
            B[h0_seq] = b0
        else:
            B[h0_seq] = b1
    return B






def gen_diagonal_B(b0_array, b1, h0, num_cluster):
    '''
    Generate a h0-dim diagonal B such that
        B[k_1,k_2,...,k_h0] = b_{b0,k} if all equal to k;
        B[k_1, ...,k_h0] = b_1 else;
    Arguments:
        list b0; b0 = [b01,b02,...,b0K]
        flat b1;
        int h0 len of edge
        num_cluster K, number of cluster
    Output:
        h0-d array, a base B, of shape(K,K,....,K)
    '''
    K = len(b0_array)
    assert K == num_cluster
    B = np.ones(num_cluster**h0)
    shape = tuple(np.array([num_cluster] * h0))
    B = np.reshape(B, shape)
    K_ranges = [range(0, num_cluster)] * (h0)
    K_power = [
        x for x in itertools.product(*K_ranges)
    ]  # each row is (k_1, k_2, ..., k_{h0})
    for k in range(K):
        for h0_seq in K_power:
            if all(x == h0_seq[0] for x in h0_seq):
                B[h0_seq] = b0_array[k]
            else:
                B[h0_seq] = b1
    return B

def gen_B_lst(b0, b1, h_lst, num_cluster, IsDiag=False):
    '''
    Generate a list h-dim singleton B such that
        B[k_1,k_2,...,k_h0] = b_0 if all equal to each other;
        B[k_1, ...,k_h0] = b_1 else;
    Arguments:
        float b0;
        flat b1;
        int h_lst
        num_cluster , number of clusters
        IsDiag, a flag whether to generate a list of diagonal Bs. If not, singleton Bs are generated instead.
    Output:
       a list of  h-d array, a base B, of shape(K,K,....,K)
    '''
    B_lst = []
    for h in h_lst:
        if not IsDiag:
            B_lst.append(gen_singleton_B(b0,b1,h,num_cluster))
        else:
            B_lst.append(gen_diagonal_B(b0,b1,h,num_cluster))
    return B_lst



def find_number_unique_nodes(edge_lst):
    node_subset = []
    for edge in edge_lst:
        node_subset += edge
    node_subset = list(set(node_subset))
    N = len(node_subset)
    #print(f"number of nodes are {N_subset}")
    return N, node_subset


def convert_edgelst_to_Ylst(edge_lst):
    '''
        Convert a list of tuples (k-sets) to a list of adjacency tensors
        Argument:
            Edges_lst: a list of h-sets
        Output:
            Y:  nparray, shape [N,N,...,N]
    '''
    N, node_subset = find_number_unique_nodes(edge_lst)
    edge_lst_size = [len(edge) for edge in edge_lst]
    h_lst = list(set(edge_lst_size))
    Ylst = []
    for h in h_lst:
        Y = np.zeros([N] * h)
        for subset in edge_lst:
            if len(subset) == h:
                subset_relabeled = [0]*h
                for j in range(h):
                    subset_relabeled[j] = node_subset.index(int(subset[j]))
                Y[tuple(subset_relabeled)] = 1
        Ylst.append(Y)
        #print(f"{h}-dim Y's sum is {np.sum(Y)}")
    return Ylst



def preprocess_constisciani_data(data_path, num_hyperedges=20,  isVerbose=True, isSaveEdge=True):
    '''
        Import dataset 'workplace' in contisciani's paper
            - select a subset of edge list
            - turn edge list to tensor list

        Argument:
            str data_path, the path where the data file `workpalce.npz` sits, eg data_path='../workpalce.npz'
            int num_hyperedges, select a random subset of full dataset; 1 <= num_hyperedges <= 788
        Ouput:
            Y_lst: a list of adjacency tensors
            h_lst: a range of cardinalities, eg h_lst = [2,3]
    '''
    data = np.load(data_path, allow_pickle=True)
    B = data["B"]  # incidence matrix of dimension NxE
    hye = data["hyperedges"]  # array of length E, containing the sets of hyperedges (as tuples)
    E = B.shape[1]
    ''' select a ramdom subset of hyperedges '''
    E_subset = num_hyperedges
    index_edge = sorted(random.sample([x for x in range(E)], k=E_subset))
    hye_subset = hye[index_edge]
    N_subset,_ = find_number_unique_nodes(hye_subset)
    if isVerbose:
        print('*** Summary on the preprocessed dataset:')
        print(f"Num. of  nodes is {N_subset} ")
        print(f"Num. of (Hyper)edges is : {len(hye_subset)}")
    ''' turn edge list to a list of adjacency tensor '''
    Y_lst = convert_edgelst_to_Ylst(hye_subset)
    if isSaveEdge:
        np.savez_compressed('./output/hye_subset.npz', hye_subset = hye_subset)
        print("Hyperedge data is saved in ./output/hye_subset.npz. To load the saved file, run: \
                 \n\t edgelst = np.load('./output/hye_subset.npz', allow_pickle = True) \
                 \n\t edgelst = edgelst['hye_subset']")
    edge_lst_size = [len(edge) for edge in hye_subset]
    h_lst = list(set(edge_lst_size))
    return Y_lst, h_lst


if __name__ == "__main__":
    pass




