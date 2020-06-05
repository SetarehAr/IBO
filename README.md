# IBO
Importance-based Bayesian Optimization


We have implemented our code on the top of the RoBO package found at https://github.com/automl/RoBO. For the importance sampling component, we have used the importance-sampling package found at https://github.com/idiap/importance-sampling. So these two packages and their corresponding dependencies need to be installed to run our code. 

In the "experiments_IBO" directory in "IBO_master_reduced", there is a subdirectory for each experiment including two subdirectories, "initial_data" and "methods".  For each experiment, the initial hyperparameter values which are drawn  from a Latin hypercube are saved in initial_data subdirectory. The implementation of IBO can be found in the methods subdirectory under the "exp_name_IBO.py".
