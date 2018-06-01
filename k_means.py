#!/usr/bin/env python3

'''
	This is an implementation of K-means algorithm to identify clustering
	(Uses numpy and matplotlib)
	Numpy is required, matplotlib is optional, but recommended
	(for complete docs rtfm)

'''

# system imports
from sys import argv as rd
import numpy as np
import pickle
import config
import helper

try:
	import matplotlib.pyplot as plt
	is_graph = True
except:
	is_graph = False


# custom imports
import random
from statistics import mean
from math import inf, sqrt
from time import sleep


# function to plot graph visualizing data
def plotGraph(grouping, final_c):
	n = len(final_c[0])
	K = len(final_c)

	if n < 2:
		return False

	else:
		x_ind, y_ind = (0, 0)
		r = lambda: random.randint(0,255)
		colors = []

		while x_ind == y_ind:
			x_ind, y_ind = random.choice(range(n)), random.choice(range(n))

		for dataset in grouping:
			color = "#%02X%02X%02X" % (r(),r(),r())
			colors.append(color)
			xi_s, yi_s = ([_[x_ind] for _ in dataset], [_[y_ind] for _ in dataset])
			plt.plot(xi_s, yi_s, color = color, marker = '.', linestyle = 'None')

		xi_center_new, yi_center_new = ([_[x_ind] for _ in final_c], [_[y_ind] for _ in final_c])
		c_no = 1
		for i, j, k in zip(xi_center_new, yi_center_new, colors):
			# plot the centroid
			plt.plot([i], [j], markerfacecolor = k, marker = 'D', linestyle = 'None', \
				markersize = '10' , markeredgecolor = 'black')
			# annotate it
			plt.annotate(s = "Cluster %d" % (c_no), xy = (i, j), xytext = (-5, 10), \
				textcoords = 'offset points', ha = 'right', va = 'bottom',\
				bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
			c_no += 1

		plt.title("x%d vs. x%d (K = %d)" % (y_ind + 1, x_ind + 1, K))
		plt.xlabel("x%d ->" % (x_ind + 1))
		plt.ylabel("x%d ->" % (y_ind + 1))

		plt.show()
		return True

# function to generate random centroids
def init_centroids(dataset, K):
	ind_range = range(len(dataset))
	indices = 2 * [ind_range]
	if K < len(dataset) / 2:
		while len(indices) != len(set(indices)):
			indices = [random.choice(ind_range) for _ in range(K)]
	else:
		indices = [i for i in range(K)]

	# return the formulated centroids
	return [dataset[_] for _ in indices]


# the function processes the dataset and chooses the initial centoids
def process_data(filename):
	raw_data = pickle.load(open(filename, 'rb'))
	dataset = np.ndarray.tolist(raw_data)
	return dataset


# function to calculate cost
def J(closest, xi_s, centroids):
	cost = 0
	for close_ind, ex in zip(closest, xi_s):
		cost += sum([(xi - ui) ** 2 for xi, ui in zip(ex, centroids[close_ind])])
	cost *= (1 / len(xi_s))
	return cost

# this functions applies K-means to find the centoids
def find_centroids(dataset, K_list, is_graph = is_graph):

	m = len(dataset)
	n = len(dataset[0])
	across_K_centroid = []
	across_K_cost = []
	across_K_cluster = []

	for K in list(set(K_list)):
		print("\n\n\nRunning K-means with %d clusters" % K)
		sleep(2)

		master_counter = 1
		if K <= config.max_K:
			master_limit = config.total_initials
		else:
			master_limit = 1

		master_centroid_history = []
		master_cost_history = []
		master_cluster_history = []

		# master loop to loop over different inital centroid values
		while master_counter <= master_limit:
			centroid_history = [init_centroids(dataset, K)] # this will record centroid convergence
			print("\n\nInitialization #%d : " % (master_counter), centroid_history[-1])

			cost_history = [inf] # J values history

			# this returns the final grouping
			# DOES NOT STORE HISTORY
			cluster_grouping = []

			# variable to check if clean results obtained
			is_clean = False

			run_count = 1
			while True:
				# stores the index of the centroid to which each example is close to
				close_to = []
				for ex in dataset:
					# measures the closeness of the current training example to each centroid
					# ith value is closeness to ith centroid
					closeness = []
					current_centroids = centroid_history[-1]
					for cluster in current_centroids:
						cur_close = sum([(xi - ci) ** 2 for xi, ci in zip(ex, cluster)])
						closeness.append(cur_close)
					# append the index of the closest centroid to the closeness history
					close_to.append(closeness.index(min(closeness)))

				# stores the cost of the previous centroids chosen
				prev_cost = J(close_to, dataset, current_centroids)
				cost_history.append(prev_cost)

				print("#%d" % run_count, "J : " ,cost_history[-1])

				# stores the grouped data
				cluster_grouping = [ [] for _ in range(K) ]

				# iterating and aggregating examples according to closeness
				[ cluster_grouping[clus_ind].append(dataset[x_index]) \
				for x_index, clus_ind in enumerate(close_to) ]

				# stores the new centroid values
				new_centroids = []
				# iterating over clusters of data
				# this loops creates the new centoids
				for clus in cluster_grouping:
					# current clus-th centroid
					current_centroid = [ mean([_[i] for _ in clus]) for i in range(n) ]
					new_centroids.append(current_centroid)
				centroid_history.append(new_centroids)
				run_count += 1

				# conditions to check convergence/divergence

				if helper.is_close_cost(cost_history[-1], cost_history[-2]):
					is_clean = True
					print("J Convergence")
					break

				# break out if the centroids have converged
				if run_count > config.min_run_count and \
					helper.is_close(centroid_history[-1], centroid_history[-2]):
					print("Centroids have converged")
					is_clean = True
					break

				# run overflow
				if run_count > config.max_run_count:
					is_clean = False
					print("Run Overflow")
					break

			if is_clean:
				print("History updated!")
				master_centroid_history.append(centroid_history[-1])
				master_cost_history.append(cost_history[-1])
				master_cluster_history.append(cluster_grouping)

			print("Restarting with different initialization")
			master_counter += 1

		# return the final centroids
		min_ind = master_cost_history.index(min(master_cost_history))
		print("Final cost J (K = %d): " % (K) , master_cost_history[min_ind])

		if is_graph:
			plotGraph(master_cluster_history[min_ind], master_centroid_history[min_ind])
		else:
			print("Final Centroids : \n", master_centroid_history[min_ind])

		across_K_centroid.append(master_centroid_history[min_ind])
		across_K_cluster.append(master_cluster_history[min_ind])
		across_K_cost.append(master_cost_history[min_ind])

	min_ind = across_K_cost.index(min(across_K_cost))
	print("\n\n\nRUN(S) COMPLETE")
	print("K - means run with", K_list, "clusters")
	print("Final clusters, K :", K_list[min_ind], end = "\n\n\n")
	return across_K_centroid[min_ind] #, across_K_cluster[min_ind]


# find the closest cluster head to a given example
# returns the index
def closest_to(xi_s, centroids):
	closeness = [sqrt(sum([(xi - ci) ** 2 for xi, ci in \
		zip(xi_s, cent)])) for cent in centroids]
	return closeness.index(min(closeness))

# once calculated to find new closeness or cluster group
def query_y(centroids):


	# this is for taking input from the user
	print("Enter the xi(s)", end = " : ")
	try:
		xi_s = list(map(float, list(filter( lambda x: x != '', input().strip(' ').split(' ')))))
		if len(xi_s) != len(centroids[0]):
			raise Exception
		print("I/P (x) -> ", xi_s)
		cluster = closest_to(xi_s, centroids) + 1
		print("Belongs to cluster %d" % (cluster))
		return cluster

	except:
		valid_input = False
		print("Invalid Input")
		return -1


# tie everything together
def main():
	filename = rd[1]
	# number of clusters
	try:
		K = [int(x) for x in rd[2:]]
	except:
		K = [config.default_K]

	# convert raw file data into pythonic data
	dataset = process_data(filename)
	# find centroid by applying K - means
	final_centroids = find_centroids(dataset, K)

	cont = True
	while cont:
		query_y(final_centroids)
		print("Calculate another (Y/n) : ", end = "")
		ans, cont = str(input()), False
		if ans == "" or ans[0] == "" or ans[0] == "y":
			cont = True


# for script access
if __name__ == '__main__':
	print("K-means Implementation (v2)")
	main()
