# Mixed membership stochastic blockmodel for general hypergraphs

It is a python implementation of variational EM algorithms in Chapter 5

### Contents
- `code`:
    - `mmsbm*.py`: various VEM algorithms
    - `utils*.py`: helpers to run experiment 1(diagonal vs singleton),2(online vs full-batch),3(realdata workplace)
    - `demo*.ipynb`: demo on analysing result from algorithms in experiments.
    - `vizCM.py` and `viz_contisciani_tool.py` are adapted from `Hypergraph-MT` Copyright (c) 2022 Martina Contisciani and Caterina De Bacco for visualization.
- `data`:
    - `data/input`: dataset Workplace, including  `workplace.npz` and `workplace_meta.csv`
    - `data/output`: experiments result returned by `code/mmsbm*.py`. Note that all parameters in the files `mmsbm*.py` are set up for simple tests only, not for the result to visualize. See more details about experiments in Chapter 5.


### Requirements
The code base is maintained in Python 3.8 with several modules in `requirements.txt`. To install these modules, run:
```shell
    conda install -n YOUR_ENV -r requirements.txt
```


### Usage

Experiments are documented in `main*.py` and `demo*.ipynb` files in folder `/code/`

- Experiment $1$: identifiability of mmsbm models.

We simulate two types of hypergraphs and run two VEMs on these simulated data respectively. ELBOs are reported. One may refer to the notebook `demo_diagonal_vs_singleton.ipynb`.


- Experiment $2$: online VEM vs full batch VEM.
Two VEMs, online and full batch are compared against each other in varisous set-ups. See the file `demo_onlin_vem.ipynb`.


- Experiment $3$: analytics on real data 'Workplace'
We run rugular VEM on the general hypergraph data 'Workplace'.Report posterior membership against true labels. Visualization modules are adapted from the paper Contisciani etc, 2022.








