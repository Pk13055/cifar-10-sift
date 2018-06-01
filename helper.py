'''
	helper functions for the k-means algorithm which are not 
	directly related to it.

'''
import numpy as np
import config

def sanitize(obj, def_char = ''):
	return type(obj)(filter(lambda x: x != def_char, obj))


def is_close(new_centroid, old_centroid):
	new_centroid = np.matrix(new_centroid)
	old_centroid = np.matrix(old_centroid)
	return np.all(np.divide(abs(new_centroid - old_centroid),\
		 abs(new_centroid)) < config.cent_tolerance)

def is_close_cost(new_J, old_J):
	return abs(new_J - old_J) / new_J < config.J_tolerance
