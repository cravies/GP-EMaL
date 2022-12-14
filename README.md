# GPMaLMO (+ new complexity measures)

# Goal
* Currenty GPMaLMO optimises embedding accuracy and minimises tree size to make readable trees
* However tree size is a poor proxy for tree interpretability
* This is because it doesn't take into account functional complexity (among other things)
* We want to try to improve the complexity measurment so we can better optimise for tree readability

Done:

In Alpha:
* Runtime 
* Tikhonov regularisation

Unimplemented
* Tree skewness
* Dictionary complexity

# How to:

Rough usage (from the src/ directory):   
`python3 -m gpmalmo.gpmal_nc --help`  
e.g. `python3 -m gpmalmo.gpmal_nc -d COIL20 --dir "datasets/"`

* Datasets used in the paper are in datasets/
* Add your own datasets in csv format, with a header line:  
Header: classPosition,#features,#classes,seperator. e.g.  
`classLast,1024,20,comma (from COIL20.data)`
* Most GP parameters are configured in gpmalmo/rundata.py
