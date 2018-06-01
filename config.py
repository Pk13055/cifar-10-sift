'''
	this is the configuration file for the k-means script.
	contains variables that should be changed with caution

'''

# default number of clusters
default_K = 2

# tolerance to check closeness of centroids
cent_tolerance = 1e-7

# max and min counts
max_run_count = 1e5
min_run_count = 1e2

# max K random init
max_K = 5

# number of different inital values for centroids
# as long as we choose so long as K < max_K
total_initials = 5

# J convergence 
J_tolerance = 1e-5
