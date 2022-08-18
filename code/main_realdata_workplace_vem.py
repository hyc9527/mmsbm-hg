#!/usr/bin/env python
# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=false
from mmsbm_general_hg import MMSB_hg_general
import utils_realdata_workplace


'''
    Experiment: run vem on the dataset workplace.npz (Contisciani etc, 2022)
        Input:
            full dataset is in ./dataset/contisciani/workplace.npz
                - num. of nodes = 92
                - num. of hyperedges = 788
                - cardinalities k = 2,3,4
        Output: save estimated parameters in ./output
'''

def run_vem_workplace():
    ''' Load subset of  dataset workplace '''
    data_path_demo = '../data/input/workplace.npz'

    ''' Covert npz dataset to tensor'''
    Y_lst, h_lst = utils_realdata_workplace.preprocess_constisciani_data(data_path=data_path_demo, num_hyperedges=20, isVerbose=True, isSaveEdge=True)

    ''' Run vem '''
    true_K = 2
    alpha0 = 0.1
    model = MMSB_hg_general(Y_lst, h_lst, true_K, alpha0)
    res = model.run_vem_general(verbose=True)
    #print(f"estimated alpha is {res['alpha']}")
    #print(f" estimated phi is {res['phi']}")
    print(f"ELBO is {res['elbo']}")
    model.save_output()


if __name__ == "__main__":
    run_vem_workplace()




