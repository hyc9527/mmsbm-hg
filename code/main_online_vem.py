#!/usr/bin/env python
# pyright:  reportGeneralTypeIssues=false, reportUnboundedVariable=false, reportMissingImports=false


import utils_mmsbm_uniform

'''
    Experiment: run two VEMs on simulated hg
'''

def experiment_online_vs_diag():

    num_node=6
    h_lst=[2,3,4]
    num_clust=2
    num_chains=1

    # Run experiment and save result in ./output/
    utils_mmsbm_uniform.experiment_online_vs_full(num_node, h_lst, num_clust, num_chains)

    # Viz of result
    utils_mmsbm_uniform.quick_3plots() # three subplots for h = 2,3,4
    #utils_mmsbm_uniform.quick_2plots() # two subplots for h = 2,3


if __name__=='__main__':
    experiment_online_vs_diag()


