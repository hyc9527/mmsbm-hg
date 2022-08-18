#!/usr/bin/env python
# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=false

import utils_realdata_workplace_viz

def viz_workplace_K2 ():

    ''' input paths'''
    vem_result_path_demo='../data/output/res_workplace_m788.npz'     # estimated parameters
    workplace_meta_path_demo='../data/input/workplace_meta.csv'    # original workplace dateset


    ''' check whether there is a need to permute cluster labels to match true label'''
    vote_mat=utils_realdata_workplace_viz.calc_vote_mat_2class(vem_result_path=vem_result_path_demo, workplace_meta_path= workplace_meta_path_demo)

    ''' compute dict on how to permute phi's columns '''
    utils_realdata_workplace_viz.assign_new_cluster_labels(vote=vote_mat)


    ''' check cluster accuracy '''
    acc1,acc2 = utils_realdata_workplace_viz.label_accuracy_2class(vem_result_path=vem_result_path_demo, workplace_meta_path= workplace_meta_path_demo)
    print(acc1, acc2)

    ''' pie plot of general hypergraph'''
    #print('Start ploting workplace hypergraph')
    utils_realdata_workplace_viz.viz_workplace_2class( vem_result_path=vem_result_path_demo, workplace_meta_path= workplace_meta_path_demo)  # Binary classification



if __name__ == "__main__":
    viz_workplace_K2()



