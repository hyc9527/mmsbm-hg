# pyright:  reportGeneralTypeIssues=false, reportUnboundedVariable=false

import itertools
import matplotlib.pyplot as plt
import numpy as np
import mmsbm_online
import mmsbm_full_batch
import sklearn.model_selection #type:ignore
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl



def gen_singleton_B(b0, b1, h0, num_cluster):
    """
    Generate a h0-dim array B such that
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
    B = np.ones(num_cluster**h0)
    shape = tuple(np.array([num_cluster]*h0))
    B = np.reshape(B,shape)
    K_ranges = [range(0, num_cluster)]*(h0)
    K_power = [x for x in itertools.product(*K_ranges)] # each row is (k_1, k_2, ..., k_{h0})
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


def gen_mmsb_hg(
    num_node, h0 , num_clust,affinity_tensor, nodal_clust_dist):
    """
    generate an adjacency tensor  Y from MMSB(N,h0,K,B,Pi)
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
    # shape = tuple(np.array([num_clust]*h0))
    # B = np.reshape(B,shape)
    K_ranges = [range(0, num_clust)]*(h0)
    K_ranges = [x for x in itertools.product(*K_ranges)] # each row is (k_1, k_2, ..., k_{h0})
    hg = np.zeros([num_node] * h0)
    nodeset = [x for x in range(num_node)]
    all_subsets = list(itertools.permutations(nodeset, h0))
    for subset in all_subsets:
        labels_subset = np.ones(len(subset))
        for j in range(len(labels_subset)):
            node = subset[j]
            z = np.random.multinomial(1, nodal_clust_dist[node, :])
            #print(f'z = {z}')
            #print(np.where(z>0))
            labels_subset[j] = np.where(z>0)[0]
        labels_subset = tuple(labels_subset.astype(int))
        #print(f' m = {labels_subset}')
        b = affinity_tensor[labels_subset]
        #print(b)
        hg[subset] = np.random.binomial(1, b)
    return hg


def gen_singleton_hg(num_node, h0, num_cluster=2, alpha0=0.1):
    """
        Generate adjacency tensor  from a singleton hypergraph from mm-sbm
    """
    b0=0.9
    b1=0.1
    true_B = gen_singleton_B(b0, b1, h0, num_cluster)
    true_pi = np.random.dirichlet([alpha0] * num_cluster, size=num_node)
    Y = gen_mmsb_hg(
        num_node, h0, num_cluster, true_B, true_pi
    )
    return Y


def gen_diagonal_hg(num_node, h0, num_cluster=2, alpha0=0.1):
    """
    Generate adjacency tensor  from a diagonal hypergraph from mm-sbm
    """
    b0 = 0.1 * np.random.random(num_cluster) + 0.9
    b1 = 0.1
    true_B = gen_diagonal_B(b0, b1, h0, num_cluster)
    true_pi = np.random.dirichlet([alpha0] * num_cluster, size=num_node)
    Y = gen_mmsb_hg(num_node, h0, num_cluster, true_B, true_pi)
    print(f"A diagonal hg is generated: Main line b0 ={b0}, off line b1={b1} ")
    return Y


def split_hg_by_node(Y_to_split, test_ratio=0.3):
    """
        Old way to splite h0-dim adjacency matrix Y into two tensor of same size: Y_train and Y_test.
        Vertice splitting method slicing full tensor into trn set and test set. one problem is train set is too sparse.
        Arguments:
            Y_to_split, h0-dim array
        Output:
            Y_train, h0-dim array
            Y_test, h0-dim array
    """
    N = Y_to_split.shape[0]
    nodeset = [x for x in range(N)]
    nodeset_train, nodeset_test = sklearn.model_selection.train_test_split(
        nodeset, test_size=test_ratio
    )
    Y_train = Y_to_split.copy()
    Y_train = 0*Y_train
    Y_test = Y_to_split.copy()
    for edge, entry in np.ndenumerate(Y_to_split):
        IsEdgeInTrn=set(edge).issubset(nodeset_train)
        if IsEdgeInTrn:
            Y_train[edge] = 1
        else:
            Y_test[edge] = 0
        return Y_train, Y_test


def split_hg(Y_to_split, test_ratio=0.3):
    """
        Splite h0-dim adjacency matrix Y into two tensor of same size: Y_train and Y_test.
        Method: random split edge set into train set and test set.
        Arguments:
            Y_to_split, h0-dim array
        Output:
            Y_train, h0-dim array
            Y_test, h0-dim array
    """
    num_edge = int(np.sum(Y_to_split))
    edge_indices = [x for x in range(num_edge)]
    edge_trn, edge_tst = sklearn.model_selection.train_test_split(
        edge_indices, test_size=test_ratio
    )
    assert len(edge_trn) > 1
    assert len(edge_tst) > 1
    print(f'Splitting hg : trn size {len(edge_trn)}; tst size {len(edge_tst)};')
    Y_trn = Y_to_split.copy()
    Y_tst = Y_to_split.copy()
    counter = 0
    #print(Y_to_split)
    for edge, entry in np.ndenumerate(Y_to_split):
        #print(edge, entry)
        if int(entry) == 1:
            counter += 1
            if counter < len(edge_trn) + 1:
                Y_tst[edge] = 0
            else:
                Y_trn[edge] = 0
    return Y_trn, Y_tst





def trim_list_of_tuple(lst_of_tuple):
    lens = [len(x) for x in lst_of_tuple]
    ncols = min(lens)
    nrows = len(lens)
    res = np.zeros((nrows, ncols))
    for j in range(ncols):
        for i in range(nrows):
            res[i,j] = lst_of_tuple[i][j]
    return res



def experiment_online_vs_full(N, h_lst, true_K, num_chains, output_folder = './output/'):
    elbo_online = []
    elbo_full = []
    for h in h_lst:
        Y_sg_unif = gen_singleton_hg(N, h, true_K)
        for iter in range(num_chains):
            print(f'h0 = {h}, chain {iter}')
            model_online = mmsbm_online.MMSB_hg_online(Y_sg_unif, h,true_K)
            res_online = model_online.run_vem_online()
            model_full = mmsbm_full_batch.MMSB_hg(Y_sg_unif, h,true_K)
            res_full = model_full.run_vem(IsDiagonalOptB=False)
            if elbo_online is None:
                elbo_online = [tuple(res_online["elbo"][0:])]
                elbo_full = [tuple(res_full["elbo"][0:])]
            else:
                elbo_online += [tuple(res_online["elbo"][0:])]
                elbo_full += [tuple(res_full["elbo"][0:])]
        elbo_online_trim = trim_list_of_tuple(elbo_online) # keep every tuple same length by throwing away tail(s).
        elbo_full_trim = trim_list_of_tuple(elbo_full)
        np.savez(output_folder+f'test_res_online_vs_full_h{h}.npz', elbo_online_trim=elbo_online_trim,elbo_full_trim=elbo_full_trim)
        print("exp2-h{h} results are saved. To load, run np.load('test_res_online_vs_full_on_hXXX_uniform_hg.npz')")



def viz_experiment_online_vem(isSavePlot=True):
    #from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
    #from palettable.colorbrewer.qualitative import Set2_7, Set3_12
    #colors = Set2_7.mpl_colors
    """ Fonts """
    params = {
        "axes.labelsize": 8,
        "ps.fonttype": 42,
        "font.size": 8,
        "font.family": "Arial",
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": False,
        "figure.figsize": [10, 10]
    }
    mpl.rcParams.update(params)
    fig = plt.figure()
    """ Load data """
    # for subplot 1
    res1 = np.load("../data/output/res_online_vs_full_h2.npz")
    # for subplot 2
    res2 = np.load("../data/output/res_online_vs_full_h3.npz")
    # for subplot 3
    res3 = np.load("../data/output/res_online_vs_full_h4.npz")
    """ Subplot 1 """
    ax1 = fig.add_subplot(131)
    elbo_online = res1["elbo_online_trim"]
    elbo_full = res1["elbo_full_trim"]
    iter_online = [x for x in range(elbo_online.shape[1])]
    ave_elbo_online = np.mean(elbo_online, axis=0)
    upd_ave_elbo_online = np.quantile(elbo_online, 0.75, axis=0)
    lpd_ave_elbo_online = np.quantile(elbo_online, 0.15, axis=0)
    ax1.grid(axis="y", color="0.9", linestyle="-.", linewidth=0.5)
    ax1.plot(
        iter_online,
        ave_elbo_online,
        "--o",
        linewidth=0.5,
        label="online vem",
        color='green',
    )
    ax1.fill_between(
        iter_online, upd_ave_elbo_online, lpd_ave_elbo_online, color='green', alpha=0.1
    )
    iter_full = [x for x in range(elbo_full.shape[1])]
    ave_elbo_full = np.mean(elbo_full, axis=0)
    upd_ave_elbo_full = np.quantile(elbo_full, 0.75, axis=0)
    lpd_ave_elbo_full = np.quantile(elbo_full, 0.15, axis=0)
    ax1.plot(
        iter_full, ave_elbo_full, "--v", linewidth=0.5, label="full batch", color='red'
    )
    ax1.fill_between(
        iter_full, upd_ave_elbo_full, lpd_ave_elbo_full, color='red', alpha=0.1
    )
    ax1.set_ylabel("log likelihood")
    ax1.set_xlabel("VEM iteration")
    legend = ax1.legend(loc=4)
    frame = legend.get_frame()
    frame.set_facecolor("1")
    frame.set_edgecolor("0.90")
    """ Trim up x-axis and y-axis """
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="x", direction="out")
    ax1.tick_params(axis="y", length=0)
    ax1.set_axisbelow(True)
    """
        Subplot 2
    """
    ax2 = fig.add_subplot(132)
    elbo_online = res2["elbo_online_trim"]
    elbo_full = res2["elbo_full_trim"]
    iter_online = [x for x in range(elbo_online.shape[1])]
    ave_elbo_online = np.mean(elbo_online, axis=0)
    upd_ave_elbo_online = np.quantile(elbo_online, 0.95, axis=0)
    lpd_ave_elbo_online = np.quantile(elbo_online, 0.05, axis=0)
    ax2.grid(axis="y", color="0.9", linestyle="-.", linewidth=0.5)
    ax2.plot(
        iter_online,
        ave_elbo_online,
        "--o",
        linewidth=0.5,
        label="online vem",
        color='green',
    )
    ax2.fill_between(
        iter_online, upd_ave_elbo_online, lpd_ave_elbo_online, color='green', alpha=0.1
    )
    iter_full = [x for x in range(elbo_full.shape[1])]
    ave_elbo_full = np.mean(elbo_full, axis=0)
    upd_ave_elbo_full = np.quantile(elbo_full, 0.95, axis=0)
    lpd_ave_elbo_full = np.quantile(elbo_full, 0.05, axis=0)
    ax2.plot(
        iter_full, ave_elbo_full, "--v", linewidth=0.5, label="regular vem", color='red'
    )
    ax2.fill_between(
        iter_full, upd_ave_elbo_full, lpd_ave_elbo_full, color='red', alpha=0.1
    )
    ax2.set_xlabel("VEM iteration")
    ax2.legend(loc=4)
    """ Trim up x-axis and y-axis """
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="x", direction="out")
    ax2.tick_params(axis="y", length=0)
    ax2.set_axisbelow(True)

    """
        Subplot 3
    """
    ax3 = fig.add_subplot(133)
    elbo_online = res3["elbo_online_trim"]
    elbo_full = res3["elbo_full_trim"]
    iter_online = [x for x in range(elbo_online.shape[1])]
    ave_elbo_online = np.mean(elbo_online, axis=0)
    upd_ave_elbo_online = np.quantile(elbo_online, 0.95, axis=0)
    lpd_ave_elbo_online = np.quantile(elbo_online, 0.05, axis=0)
    ax3.grid(axis="y", color="0.9", linestyle="-.", linewidth=0.5)
    ax3.plot(
        iter_online,
        ave_elbo_online,
        "--o",
        linewidth=0.5,
        label="online vem",
        color='green',
    )
    ax3.fill_between(
        iter_online, upd_ave_elbo_online, lpd_ave_elbo_online, color='green', alpha=0.1
    )
    iter_full = [x for x in range(elbo_full.shape[1])]
    ave_elbo_full = np.mean(elbo_full, axis=0)
    upd_ave_elbo_full = np.quantile(elbo_full, 0.95, axis=0)
    lpd_ave_elbo_full = np.quantile(elbo_full, 0.05, axis=0)
    ax3.plot(
        iter_full, ave_elbo_full, "--v", linewidth=0.5, label="regular vem", color='red'
    )
    ax3.fill_between(
        iter_full, upd_ave_elbo_full, lpd_ave_elbo_full, color='red', alpha=0.1
    )
    ax3.set_xlabel("VEM iteration")
    ax3.legend(loc=4)
    """ Trim up x-axis and y-axis """
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.tick_params(axis="x", direction="out")
    ax3.tick_params(axis="y", length=0)
    ax3.set_axisbelow(True)
    fig.text(
            0.1, 0.9, "A: h = 2", weight="bold", horizontalalignment="left", verticalalignment="center"
    )
    fig.text(
            0.5, 0.9, "B: h = 3", weight="bold", horizontalalignment="left", verticalalignment="center"
    )
    fig.text(
            0.9, 0.9, "C: h = 4", weight="bold", horizontalalignment="left", verticalalignment="center"
    )
    plt.show()
    if isSavePlot:
        plt.savefig("./output/tempt_Final_Plot.png", dpi=300, transparent=False, bbox_inches="tight")




















def quick_3plots(isSavePlot=True):
    #from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
    #from palettable.colorbrewer.qualitative import Set2_7, Set3_12
    #colors = Set2_7.mpl_colors
    """ Fonts """
    params = {
        "axes.labelsize": 8,
        "ps.fonttype": 42,
        "font.size": 8,
        "font.family": "Arial",
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": False,
        "figure.figsize": [10, 6]
    }
    mpl.rcParams.update(params)
    fig = plt.figure()
    """ Load data """
    # for subplot 1
    res1 = np.load("./output/test_res_online_vs_full_h2.npz")
    # for subplot 2
    res2 = np.load("./output/test_res_online_vs_full_h3.npz")
    # for subplot 3
    res3 = np.load("./output/test_res_online_vs_full_h4.npz")
    """ Subplot 1 """
    ax1 = fig.add_subplot(131)
    elbo_online = res1["elbo_online_trim"]
    elbo_full = res1["elbo_full_trim"]
    iter_online = [x for x in range(elbo_online.shape[1])]
    ave_elbo_online = np.mean(elbo_online, axis=0)
    upd_ave_elbo_online = np.quantile(elbo_online, 0.75, axis=0)
    lpd_ave_elbo_online = np.quantile(elbo_online, 0.15, axis=0)
    ax1.grid(axis="y", color="0.9", linestyle="-.", linewidth=0.5)
    ax1.plot(
        iter_online,
        ave_elbo_online,
        "--o",
        linewidth=0.5,
        label="online vem",
        color='green',
    )
    ax1.fill_between(
        iter_online, upd_ave_elbo_online, lpd_ave_elbo_online, color='green', alpha=0.1
    )
    iter_full = [x for x in range(elbo_full.shape[1])]
    ave_elbo_full = np.mean(elbo_full, axis=0)
    upd_ave_elbo_full = np.quantile(elbo_full, 0.75, axis=0)
    lpd_ave_elbo_full = np.quantile(elbo_full, 0.15, axis=0)
    ax1.plot(
        iter_full, ave_elbo_full, "--v", linewidth=0.5, label="full batch", color='red'
    )
    ax1.fill_between(
        iter_full, upd_ave_elbo_full, lpd_ave_elbo_full, color='red', alpha=0.1
    )
    ax1.set_ylabel("log likelihood")
    ax1.set_xlabel("VEM iteration")
    legend = ax1.legend(loc=4)
    frame = legend.get_frame()
    frame.set_facecolor("1")
    frame.set_edgecolor("0.90")
    """ Trim up x-axis and y-axis """
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="x", direction="out")
    ax1.tick_params(axis="y", length=0)
    ax1.set_axisbelow(True)
    """
        Subplot 2
    """
    ax2 = fig.add_subplot(132)
    elbo_online = res2["elbo_online_trim"]
    elbo_full = res2["elbo_full_trim"]
    iter_online = [x for x in range(elbo_online.shape[1])]
    ave_elbo_online = np.mean(elbo_online, axis=0)
    upd_ave_elbo_online = np.quantile(elbo_online, 0.95, axis=0)
    lpd_ave_elbo_online = np.quantile(elbo_online, 0.05, axis=0)
    ax2.grid(axis="y", color="0.9", linestyle="-.", linewidth=0.5)
    ax2.plot(
        iter_online,
        ave_elbo_online,
        "--o",
        linewidth=0.5,
        label="online vem",
        color='green',
    )
    ax2.fill_between(
        iter_online, upd_ave_elbo_online, lpd_ave_elbo_online, color='green', alpha=0.1
    )
    iter_full = [x for x in range(elbo_full.shape[1])]
    ave_elbo_full = np.mean(elbo_full, axis=0)
    upd_ave_elbo_full = np.quantile(elbo_full, 0.95, axis=0)
    lpd_ave_elbo_full = np.quantile(elbo_full, 0.05, axis=0)
    ax2.plot(
        iter_full, ave_elbo_full, "--v", linewidth=0.5, label="regular vem", color='red'
    )
    ax2.fill_between(
        iter_full, upd_ave_elbo_full, lpd_ave_elbo_full, color='red', alpha=0.1
    )
    ax2.set_xlabel("VEM iteration")
    ax2.legend(loc=4)
    """ Trim up x-axis and y-axis """
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="x", direction="out")
    ax2.tick_params(axis="y", length=0)
    ax2.set_axisbelow(True)

    """
        Subplot 3
    """
    ax3 = fig.add_subplot(133)
    elbo_online = res3["elbo_online_trim"]
    elbo_full = res3["elbo_full_trim"]
    iter_online = [x for x in range(elbo_online.shape[1])]
    ave_elbo_online = np.mean(elbo_online, axis=0)
    upd_ave_elbo_online = np.quantile(elbo_online, 0.95, axis=0)
    lpd_ave_elbo_online = np.quantile(elbo_online, 0.05, axis=0)
    ax3.grid(axis="y", color="0.9", linestyle="-.", linewidth=0.5)
    ax3.plot(
        iter_online,
        ave_elbo_online,
        "--o",
        linewidth=0.5,
        label="online vem",
        color='green',
    )
    ax3.fill_between(
        iter_online, upd_ave_elbo_online, lpd_ave_elbo_online, color='green', alpha=0.1
    )
    iter_full = [x for x in range(elbo_full.shape[1])]
    ave_elbo_full = np.mean(elbo_full, axis=0)
    upd_ave_elbo_full = np.quantile(elbo_full, 0.95, axis=0)
    lpd_ave_elbo_full = np.quantile(elbo_full, 0.05, axis=0)
    ax3.plot(
        iter_full, ave_elbo_full, "--v", linewidth=0.5, label="regular vem", color='red'
    )
    ax3.fill_between(
        iter_full, upd_ave_elbo_full, lpd_ave_elbo_full, color='red', alpha=0.1
    )
    ax3.set_xlabel("VEM iteration")
    ax3.legend(loc=4)
    """ Trim up x-axis and y-axis """
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.tick_params(axis="x", direction="out")
    ax3.tick_params(axis="y", length=0)
    ax3.set_axisbelow(True)
    fig.text(
            0.1, 0.9, "A: h = 2", weight="bold", horizontalalignment="left", verticalalignment="center"
    )
    fig.text(
            0.5, 0.9, "B: h = 3", weight="bold", horizontalalignment="left", verticalalignment="center"
    )
    fig.text(
            0.9, 0.9, "C: h = 4", weight="bold", horizontalalignment="left", verticalalignment="center"
    )
    plt.show()
    if isSavePlot:
        plt.savefig("./output/tempt_Final_Plot.png", dpi=300, transparent=False, bbox_inches="tight")




def quick_2plots(isSavePlot=True):
    #from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
    #from palettable.colorbrewer.qualitative import Set2_7, Set3_12
    #colors = Set2_7.mpl_colors
    """ Fonts """
    params = {
        "axes.labelsize": 8,
        "ps.fonttype": 42,
        "font.size": 8,
        "font.family": "Arial",
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": False,
        "figure.figsize": [10, 6]
    }
    mpl.rcParams.update(params)
    fig = plt.figure()
    """ Load data """
    # for subplot 1
    res1 = np.load("./output/res_online_vs_full_h2.npz")
    # for subplot 2
    res2 = np.load("./output/res_online_vs_full_h3.npz")
    # for subplot 3
    # res3 = np.load("./output/res_online_vs_full_h4.npz")
    """ Subplot 1 """
    ax1 = fig.add_subplot(131)
    elbo_online = res1["elbo_online_trim"]
    elbo_full = res1["elbo_full_trim"]
    iter_online = [x for x in range(elbo_online.shape[1])]
    ave_elbo_online = np.mean(elbo_online, axis=0)
    upd_ave_elbo_online = np.quantile(elbo_online, 0.75, axis=0)
    lpd_ave_elbo_online = np.quantile(elbo_online, 0.15, axis=0)
    ax1.grid(axis="y", color="0.9", linestyle="-.", linewidth=0.5)
    ax1.plot(
        iter_online,
        ave_elbo_online,
        "--o",
        linewidth=0.5,
        label="online vem",
        color='green',
    )
    ax1.fill_between(
        iter_online, upd_ave_elbo_online, lpd_ave_elbo_online, color='green', alpha=0.1
    )
    iter_full = [x for x in range(elbo_full.shape[1])]
    ave_elbo_full = np.mean(elbo_full, axis=0)
    upd_ave_elbo_full = np.quantile(elbo_full, 0.75, axis=0)
    lpd_ave_elbo_full = np.quantile(elbo_full, 0.15, axis=0)
    ax1.plot(
        iter_full, ave_elbo_full, "--v", linewidth=0.5, label="full batch", color='red'
    )
    ax1.fill_between(
        iter_full, upd_ave_elbo_full, lpd_ave_elbo_full, color='red', alpha=0.1
    )
    ax1.set_ylabel("log likelihood")
    ax1.set_xlabel("VEM iteration")
    legend = ax1.legend(loc=4)
    frame = legend.get_frame()
    frame.set_facecolor("1")
    frame.set_edgecolor("0.90")
    """ Trim up x-axis and y-axis """
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="x", direction="out")
    ax1.tick_params(axis="y", length=0)
    ax1.set_axisbelow(True)
    """
        Subplot 2
    """
    ax2 = fig.add_subplot(132)
    elbo_online = res2["elbo_online_trim"]
    elbo_full = res2["elbo_full_trim"]
    iter_online = [x for x in range(elbo_online.shape[1])]
    ave_elbo_online = np.mean(elbo_online, axis=0)
    upd_ave_elbo_online = np.quantile(elbo_online, 0.95, axis=0)
    lpd_ave_elbo_online = np.quantile(elbo_online, 0.05, axis=0)
    ax2.grid(axis="y", color="0.9", linestyle="-.", linewidth=0.5)
    ax2.plot(
        iter_online,
        ave_elbo_online,
        "--o",
        linewidth=0.5,
        label="online vem",
        color='green',
    )
    ax2.fill_between(
        iter_online, upd_ave_elbo_online, lpd_ave_elbo_online, color='green', alpha=0.1
    )
    iter_full = [x for x in range(elbo_full.shape[1])]
    ave_elbo_full = np.mean(elbo_full, axis=0)
    upd_ave_elbo_full = np.quantile(elbo_full, 0.95, axis=0)
    lpd_ave_elbo_full = np.quantile(elbo_full, 0.05, axis=0)
    ax2.plot(
        iter_full, ave_elbo_full, "--v", linewidth=0.5, label="regular vem", color='red'
    )
    ax2.fill_between(
        iter_full, upd_ave_elbo_full, lpd_ave_elbo_full, color='red', alpha=0.1
    )
    ax2.set_xlabel("VEM iteration")
    ax2.legend(loc=4)
    """ Trim up x-axis and y-axis """
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="x", direction="out")
    ax2.tick_params(axis="y", length=0)
    ax2.set_axisbelow(True)

    fig.text(
            0.1, 0.9, "A: h = 2", weight="bold", horizontalalignment="left", verticalalignment="center"
    )
    fig.text(
            0.5, 0.9, "B: h = 3", weight="bold", horizontalalignment="left", verticalalignment="center"
    )
    #fig.text(
    #        0.9, 0.9, "C: h = 4", weight="bold", horizontalalignment="left", verticalalignment="center"
    #)
    plt.show()
    if isSavePlot:
        plt.savefig("./output/tempt_Final_Plot.png", dpi=300, transparent=False, bbox_inches="tight")

















if __name__ == "__main__":
    pass




