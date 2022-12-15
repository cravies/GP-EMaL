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
* Dictionary complexity

Unimplemented
* Tree skewness

# How to:

Usage (from the src/ directory):   
To run on iris.data (in /src/datasets dir) for 10 generations, using size minimisation 
as our seconday objective to minimise (alongside neighbourhood structure embedding loss)
```bash
make run DATASET=iris GENS=10 OBJ=size
```
To clean up output files in the directory:
```bash
make clean
```
To run (non exhaustive) tests
```
make test
```

Note:
* Datasets used in the paper are in datasets/
* Add your own datasets in csv format, with a header line:  
Header: classPosition,#features,#classes,seperator. e.g.  
`classLast,1024,20,comma (from COIL20.data)`
* Most GP parameters are configured in gpmalmo/rundata.py
