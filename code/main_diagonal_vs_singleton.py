#!/usr/bin/env python
# pyright:  reportGeneralTypeIssues=false, reportUnboundedVariable=false, reportMissingImports=false


import utils_mmsbm_uniform





def run_experiment_single_vs_diag_analysis():
    """
        Run multiple chains of two VEM (with diagonal update and singleton update) on (i) and (ii):
            (i) simulated hg from diagonal MMSB_hg
            (ii) simulated hg from singleton MMSB_hg

        Output:
            - save result in  `/data/output/test_res_sg_dg_true_sg.npz`
            - save result in  `/data/output/test_res_sg_dg_true_dg.npz`

    """

    """ Simulate $h0$-uniform enerate hypergraph """
    num_node =10    # num. of nodes
    h0=3            # cardinality of hyper-edge
    num_clust=2     # num. of latent clusters
    num_chains=3    # num of chains




    """ Run multiple chains of  vem on singleton hg """
    utils_mmsbm_uniform.experiment_single_vs_diag(num_node, h0,num_clust,num_chains,isTrueModelDiag= False)

    """ Run multiple chains of vem on diagonal hg """
    utils_mmsbm_uniform.experiment_single_vs_diag(num_node, h0,num_clust,num_chains, isTrueModelDiag= True)

    """ quick check on experiment result """
    #utils_mmsbm_uniform.plot_exp_single_vs_diag(data="../data/output/test_res_sg_dg_true_sg.npz", IsSavePlot=True)
    #utils_mmsbm_uniform.plot_exp_single_vs_diag(data="../data/output/test_res_sg_dg_true_dg.npz", IsSavePlot=True)


if __name__ =="__main__":
    run_experiment_single_vs_diag_analysis()



