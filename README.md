# Mixed membership stochastic blockmodel for general hypergraphs

It is a python implementation of variational EM algorithms in Chapter 5

### Contents
- `code`:
    - `mmsbm*.py`: various VEM algorithms
    - `utils*.py`: helpers to run experiment 1(diagonal vs singleton),2(online vs full-batch),3(realdata workplace)
    - `analytics*.ipynb`: demo on analysing result from algorithms in experiments.
    - `vizCM.py` and `viz_contisciani_tool.py` are adapted from `Hypergraph-MT` Copyright (c) 2022 Martina Contisciani and Caterina De Bacco for visualization.
- `data`:
    - `data/input`: dataset Workplace, including  `workplace.npz` and `workplace_meta.csv`
    - `data/output`: experiments result returned by `code/mmsbm*.py`. Note that all parameters in the files `mmsbm*.py` are set up for simple tests only, not for the result to visualize. See more details about experiments in Chapter 5.


### Requirements
The code base is in Python 3.8 with modules in `requirements.txt`. One may install them
```shell
    conda install -n YOUR_ENV -r requirements.txt
```


### Usage

To run various algorithms:
```shell
    cd code
    python main_*.py
```
And we also include jupyter notebooks to illustrate how experiments are executed.







