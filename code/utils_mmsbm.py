# pyright:  reportGeneralTypeIssues=false, reportUnboundedVariable=false, reportMissingImports=false

import itertools
import matplotlib.pyplot as plt
import numpy as np
import mmsbm
# import sklearn.model_selection
import os
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import numpy as np
from palettable.colorbrewer.qualitative import Set2_7, Set3_12
colors = Set2_7.mpl_colors

def gen_singleton_B(b0, b1, h0, num_cluster):
    """
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
    """
    b0 = float(b0)
    assert isinstance(b0, float) # check b0 is a number not an array
    B = np.zeros(num_cluster**h0)
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
    """
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
    """
    K = len(b0_array)
    assert K == num_cluster # check b0_array is an array
    B = np.zeros(num_cluster**h0)
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



def gen_mmsb_hg(num_node, h0, affinity_tensor, nodal_clust_dist):
    """
        Generate adjacency tensor  Y from model MMSB(N,h0,K,B,Pi)

        Arguments:
            int num_nodes, N ;
            int h0, cardinality of uniform set.
            int num_clust, K;
            float alpha_0;
            h0-dim affinity tensor, shape (K,K,...,K), affinity tensor B
            2-dim nodal_clust_dist, shape (N,K), mat pi, each row is a prob over [K]
        Output:
            h0-dim adjacency tensor, shape (N,...,N), adjacency tensor Y in airoldi(08)
    """
    #K_ranges = [range(0, num_clust)] * (h0)
    #K_ranges = [
    #    x for x in itertools.product(*K_ranges)
    #]  # each row is (k_1, k_2, ..., k_{h0})
    hg = np.zeros([num_node] * h0)
    nodeset = [x for x in range(num_node)]
    all_subsets = list(itertools.permutations(nodeset, h0))
    for subset in all_subsets:
        labels_subset = np.ones(len(subset))
        for j in range(len(labels_subset)):
            node = subset[j]
            z = np.random.multinomial(1, nodal_clust_dist[node, :])
            labels_subset[j] = np.where(z > 0)[0]
        labels_subset = tuple(labels_subset.astype(int))
        b = affinity_tensor[labels_subset]
        hg[subset] = np.random.binomial(1, b)
    return hg




def gen_singleton_hg(b0, b1, num_node, h0, num_cluster=2, alpha0=0.1):
    """
    Generate adjacency tensor  from a singleton hypergraph from mm-sbm
    """
    #b0 = 0.9
    #b1 = 0.1
    true_B = gen_singleton_B(b0, b1, h0, num_cluster)
    true_pi = np.random.dirichlet([alpha0] * num_cluster, size=num_node)
    Y = gen_mmsb_hg(num_node, h0, true_B, true_pi)
    print(f"Generate a singleton hg using affiliate params: main line b0 ={b0}, off main line b1 ={b1} ")
    return Y


def gen_diagonal_hg(b0_arr, b1, num_node, h0, num_cluster=2, alpha0=0.1):
    """
    Generate adjacency tensor  from a diagonal hypergraph from mm-sbm
    """
    #b0 = 0.1 * np.random.random(num_cluster) + 0.9
    #b1 = 0.1
    true_B = gen_diagonal_B(b0_arr, b1, h0, num_cluster)
    true_pi = np.random.dirichlet([alpha0] * num_cluster, size=num_node)
    Y = gen_mmsb_hg(num_node, h0, true_B, true_pi)
    print(f"Generate a diagonal hg of shape {Y.shape} using affinity params: Main line b0 ={b0_arr}, off main line b1={b1} ")
    return Y



def trim_list_of_tuple(lst_of_tuple):
    lens = [len(x) for x in lst_of_tuple]
    ncols = min(lens)
    nrows = len(lens)
    res = np.zeros((nrows, ncols))
    for j in range(ncols):
        for i in range(nrows):
            res[i,j] = lst_of_tuple[i][j]
    return res




def experiment_single_vs_diag(N, h0, true_K, num_chains, isTrueModelDiag= False, output_folder = '../data/output/'):
    elbo_sg = []
    elbo_dg = []
    ''' set up affinity parameters'''
    b0_arr = 0.1 * np.random.random(true_K) + 0.6
    b1 = 0.1
    if not isTrueModelDiag:
        Y = gen_singleton_hg(b0_arr[0], b1, N, h0, true_K) # true model: singleton model
    else:
        Y = gen_diagonal_hg(b0_arr, b1, N, h0, true_K) # true model: diagonal model
    #print(f"y sum is {np.sum(Y)}")
    for iter in range(num_chains):
        print(f'Chain no.{iter} is starting.')
        model_sg = mmsbm.MMSB_hg(Y, h0, true_K)
        res_sg = model_sg.run_vem(IsDiagonalOptB=False) # run singleton vem
        model_dg = mmsbm.MMSB_hg(Y, h0, true_K)
        res_dg = model_dg.run_vem(IsDiagonalOptB=True) # run diagonal vem
        if elbo_sg is None:
            elbo_sg = [tuple(res_sg["elbo_e_m"])]
            elbo_dg = [tuple(res_dg["elbo_e_m"])]
        else:
            elbo_sg += [tuple(res_sg["elbo_e_m"])]
            elbo_dg += [tuple(res_dg["elbo_e_m"])]
    elbo_sg_trim = trim_list_of_tuple(elbo_sg) # keep every tuple same length by throwing away tail(s).
    elbo_dg_trim = trim_list_of_tuple(elbo_dg)
    if isTrueModelDiag:
        np.savez(output_folder+'test_res_sg_dg_true_dg.npz', elbo_sg_trim=elbo_sg_trim,elbo_dg_trim=elbo_dg_trim)
        print("elbo from from diagonal model are saved. To load, run res = np.load('test_res_sg_dg_true_dg.npz')")
    else:
        np.savez(output_folder+'test_res_sg_dg_true_sg.npz', elbo_sg_trim=elbo_sg_trim,elbo_dg_trim=elbo_dg_trim)
        print("elbo from singleton model are saved. To load, run res = np.load('test_res_sg_dg_true_sg')")


def plot_exp_single_vs_diag(data, IsSavePlot=False):
    '''
        Quick check on experiment_single_vs_diag result
    '''
    res = np.load(data)
    elbo_sg = res['elbo_sg_trim']
    elbo_dg = res['elbo_dg_trim']
    iter_sg = [x for x in range(elbo_sg.shape[1])]
    ave_elbo_sg = np.mean(elbo_sg, axis=0)
    upd_ave_elbo_sg = np.quantile(elbo_sg, .85, axis=0)
    lpd_ave_elbo_sg = np.quantile(elbo_sg, .15, axis=0)
    plt.plot(iter_sg, ave_elbo_sg, label = 'singleton')
    plt.fill_between(iter_sg,upd_ave_elbo_sg, lpd_ave_elbo_sg, color = 'blue', alpha = 0.1)
    iter_dg = [x for x in range(elbo_dg.shape[1])]
    ave_elbo_dg = np.mean(elbo_dg, axis=0)
    upd_ave_elbo_dg = np.quantile(elbo_dg, .85, axis=0)
    lpd_ave_elbo_dg = np.quantile(elbo_dg, .15, axis=0)
    plt.plot(iter_dg, ave_elbo_dg, label = 'diagonal')
    plt.fill_between(iter_dg,upd_ave_elbo_dg, lpd_ave_elbo_dg, color = 'orange', alpha = 0.1)
    plt.legend()
    output_name= os.path.splitext(data)[0]
    if IsSavePlot:
        plt.savefig(output_name)
    plt.show()



def viz_experiment_single_vs_diag(isSavePlot=True):
    '''
        Formal visualization of result from experiment_single_vs_diag
    '''
    # Fonts
    params = {
        'axes.labelsize': 8,
        'ps.fonttype':42,
        'font.size': 8,
        'font.family': 'Arial',
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [10, 6]
       }
    mpl.rcParams.update(params)

    # Major plot
    fig = plt.figure()
    ## Subplot 1
    res = np.load('../data/output/res_sg_dg_true_dg.npz')
    ax = fig.add_subplot(121)

    ### curve 1
    elbo_sg = res['elbo_sg_trim']
    iter_sg = [x for x in range(elbo_sg.shape[1])]
    ave_elbo_sg = np.mean(elbo_sg, axis=0)
    upd_ave_elbo_sg = np.quantile(elbo_sg, .75, axis=0)
    lpd_ave_elbo_sg = np.quantile(elbo_sg, .15, axis=0)
    ax.grid(axis='y', color='0.9', linestyle='-.', linewidth= 0.5)
    ax.plot(iter_sg, ave_elbo_sg, '--o',label = 'singleton vem',linewidth = 1,color=colors[2])
    ax.fill_between(iter_sg,upd_ave_elbo_sg, lpd_ave_elbo_sg, color=colors[0], alpha = 0.1)

    ### curve 2
    elbo_dg = res['elbo_dg_trim']
    iter_dg = [x for x in range(elbo_dg.shape[1])]
    ave_elbo_dg = np.mean(elbo_dg, axis=0)
    upd_ave_elbo_dg = np.quantile(elbo_dg, .75, axis=0)
    lpd_ave_elbo_dg = np.quantile(elbo_dg, .15, axis=0)
    ax.plot(iter_dg, ave_elbo_dg, '--v',label = 'diagonal vem',linewidth = 1,color=colors[1])
    ax.fill_between(iter_dg,upd_ave_elbo_dg, lpd_ave_elbo_dg, color = colors[1], alpha = 0.1)

    ## legend
    legend = ax.legend(loc=4)
    ax.set_ylabel('log likelihood')
    ax.set_xlabel('VEM iteration')
    frame = legend.get_frame()
    frame.set_facecolor('1')
    frame.set_edgecolor('0.90')


    ## Trim up x-axis and y-axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)
    ax.set_axisbelow(True)

    # Subplot 2
    ax1 = fig.add_subplot(122)
    res = np.load('../data/output/res_sg_dg_true_sg.npz')

    ## line 1
    elbo_sg = res['elbo_sg_trim']
    iter_sg = [x for x in range(elbo_sg.shape[1])]
    ave_elbo_sg = np.mean(elbo_sg, axis=0)
    upd_ave_elbo_sg = np.quantile(elbo_sg, .75, axis=0)
    lpd_ave_elbo_sg = np.quantile(elbo_sg, .15, axis=0)
    ax1.grid(axis='y', color='0.9', linestyle='-.', linewidth= 0.5)
    ax1.plot(iter_sg, ave_elbo_sg, '--o',label = 'singleton vem', linewidth = 1,color=colors[2])
    ax1.fill_between(iter_sg,upd_ave_elbo_sg, lpd_ave_elbo_sg, color = colors[0], alpha = 0.1)

    ## line 2
    elbo_dg = res['elbo_dg_trim']
    iter_dg = [x for x in range(elbo_dg.shape[1])]
    ave_elbo_dg = np.mean(elbo_dg, axis=0)
    upd_ave_elbo_dg = np.quantile(elbo_dg, .75, axis=0)
    lpd_ave_elbo_dg = np.quantile(elbo_dg, .15, axis=0)
    ax1.plot(iter_dg, ave_elbo_dg, '--v',linewidth = 1,label = 'diagonal vem',color=colors[1])
    ax1.fill_between(iter_dg,upd_ave_elbo_dg, lpd_ave_elbo_dg, color = colors[1], alpha = 0.1)

    ## Legend
    legend = ax1.legend(loc=4)
    ax1.set_xlabel('VEM iteration')
    frame = legend.get_frame()
    frame.set_facecolor('1')
    frame.set_edgecolor('0.90')

    ## Trim up x-axis and y-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(axis='x', direction='out')
    ax1.tick_params(axis='y', length=0)
    ax1.set_axisbelow(True)
    fig.text(0.1, 0.9, "True model: diagonal affinity $B$", weight="bold", horizontalalignment='left', verticalalignment='center')
    fig.text(0.5, 0.9, "True model: singleton affinity $B$", weight="bold", horizontalalignment='left', verticalalignment='center')

    if isSavePlot:
        plt.savefig('Final_Plot.png', dpi=300, transparent=False, bbox_inches='tight')






























if __name__ == "__main__":
    pass



