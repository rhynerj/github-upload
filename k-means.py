# imports
import tarfile
import zipfile
import urllib
import urllib.request
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import random
import argparse
    
###################################################################################################################################
# DATA REVIEVAL AND CLEANING
###################################################################################################################################
# retrieve dataset as an array using indicated data retrieval function and preprocess it
# also takes in number of components for pca
# returns pre-processed data
def cleaned_data(dataset, pca_components) :
    # get data array
    if dataset == 'gene' :
        data_array = gene_dataset()
    elif dataset == 'dass' :
        data_array = dass_dataset()
    """ # scale data
    scaled_data = min_max_scale(data_array)
    # run pca to get reduced data
    reduced_data = pca(scaled_data, pca_components) """
    # preprocess: scale data and run pca
    preprocessed_data = preprocess(data_array, pca_components)
    return preprocessed_data

# Downlod from online repository

# downloads and unpacks a dataset
# takes in url, archive, folder type
def dataset_retrieve(url, archive, f_type) :

    # url for downloading, from given url and archive
    download_url = urllib.parse.urljoin(url, archive)

    # download the archive
    r = urllib.request.urlretrieve (download_url, archive)

    if f_type == "tar" :
        # unzip the archive
        with tarfile.open(archive, 'r:gz') as opened_archive :
             opened_archive.extractall()
    elif f_type == "zip" :
        # unzip the archive
        with zipfile.ZipFile(archive, 'r') as opened_archive :
             opened_archive.extractall()


###################################################################################################################################
# retrieve data from file as np array
# takes in data_file, delimiter, upper_range (of data to retrieve)retrieves
# returns data array
def read_into_array(data_file, delim, upper_range) :
    
    # retrieve data as np array
    data = np.genfromtxt(data_file, delimiter=delim, usecols=range(1, upper_range), skip_header=1, dtype=int)

    return data

###################################################################################################################################
#  Gene Expression Cancer RNA-Seq Dataset

# downloads and unpacks the gene expression dataset
# returns the data as an array
def gene_dataset() :
    # set variables
    gene_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
    gene_archive = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"
    gene_data_file = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
    gene_delim = ","
    gene_upper_range = 350 #20532 # from data description: 20531 attributes

    # download gene expression archive files and extract data
    dataset_retrieve(gene_url, gene_archive, "tar")
    gene_data_array = np.genfromtxt(gene_data_file, delimiter=gene_delim, usecols=range(1, gene_upper_range), skip_header=1)

    return gene_data_array

###################################################################################################################################
# Depression Anxiety Stress Scales Dataset

# downloads and unpacks the dass dataset
# returns the data as an array
# set variables
def dass_dataset() :
    dass_url = "https://openpsychometrics.org/_rawdata/"
    dass_archive = "DASS_data_21.02.19.zip"
    dass_data_file = "/home/jrhyner/Documents/CS2810/final_project/DASS_data_21.02.19/data.csv"
    dass_delim = "\t"
    dass_upper_range = 127 # 3 measurements for 42 questions -> 126

    # download DASS file and extract data
    dataset_retrieve(dass_url, dass_archive, "zip")
    dass_data_array = np.genfromtxt(dass_data_file, delimiter=dass_delim, usecols=range(1, dass_upper_range), skip_header=1, dtype=int)

    return dass_data_array

###################################################################################################################################
# PREPROCESSING

# calls preprocessing steps on given data array
# also takes in number of components, num_components, for PCA
# returns processed data
def preprocess(data, num_components) :
    # scale data
    scaled_data = min_max_scale(data)

    # use PCA to reduce dimensions
    reduced_data = pca(scaled_data, num_components)

    # return preprocessed data
    return reduced_data

###################################################################################################################################
# MINMAXSCALING

# normalizes given data using minmax scaling
# takes in np array
# modeled after MinMaxScaler() in sklearn
def min_max_scale(data) :
    # min and max values of data
    data_max = np.max(data)
    data_min = np.min(data)

    # use min and max to scale data
    scaled_data = np.array([(x - data_min) / (data_max - data_min) for x in data])

    return scaled_data

###################################################################################################################################

# PRINCIPLE COMPONENT ANALYSIS

# PCA: reduces the number of dimensions in the dataset
# isolates the most variable components
# takes in a dataset and a number of dimensions
def pca(X, num_components) :
    # subtract the mean of each variable to mean center data
    X_meaned = X - np.mean(X , axis = 0)

    # calculate covariance matrix (of meaned_data)
    cov_mat = np.cov(X_meaned , rowvar = False)

    # calculate eigenvalues and -vectors from covariance matrix
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

    # sort eigenvals and vects, accoring to descending eigenval
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    # get top n of eigenvects
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]

    # transform data into reduced form
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced

###################################################################################################################################
# K-MEANS
###################################################################################################################################

# k-means class
class k_means :
    # init -> default max_loops is 100
    def __init__(self, k, max_loop=100) :
        self.k = k
        self.max_loop = max_loop

    # runs k_means (based on sklearn fit())
    def fit(self, data) :
        # initialize centroids
        self.centroids = self.init_centroids(data)
        # loop until reached max or centroids stop changing
        looped = 0
        same = False # centroids same as last time?
        while looped < self.max_loop and not same :
            # centroids at start of loop (from init or past loop)
            start_centroids = self.centroids
            distances = self.data_centroid_dists(data, start_centroids)
            # sort into clusters
            self.clusters = self.closest_dists(distances)
            # update centroids
            self.centroids = self.update_centroids(data, self.clusters)
            # update conditions
            same = np.array_equal(start_centroids, self.centroids)
            looped += 1
        # compute cluster stats
        self.sse = self.inertia(data, self.clusters, self.centroids)
        self.csizes = self.cluster_sizes(data)
        

    # initialize centroids
    def init_centroids(self, data) :
        num_rows = data.shape[0]
        # randomly select k indices
        rand_indices = np.random.choice(num_rows, size=self.k, replace=False)
        # select k random rows from data 
        centroids_arr = data[rand_indices, :]
        # set centroids
        self.centroids = centroids_arr
        return self.centroids
    
    # update centroids to be mean of data points
    def update_centroids(self, data, clusters) :
        # centroids array
        centroids_arr = np.zeros((self.k, data.shape[1]))
        for i in range(self.k) :
            # set centroid k to mean
            centroids_arr[i, :] = np.mean(data[clusters == i, :], axis=0)
        return centroids_arr
    
    # takes in a distance array and returns minimums
    def closest_dists(self, distance_arr) :
        return np.argmin(distance_arr, axis=1)
    
    # calculates the distance from each data point to each centroid
    def data_centroid_dists(self, data, centroid_arr) :
        # array of distances
        distance_arr = np.zeros((data.shape[0], self.k))
        for i in range(self.k) :
            # calculate norm of data - centroids
            dp_norm = np.linalg.norm(data - centroid_arr[i, :], axis=1)
            # insert square dist
            distance_arr[:, i] = np.square(dp_norm)
        # return results array
        return distance_arr

    # Cluster Stats

    # finds sizes of each cluster
    def cluster_sizes(self, data) :
        # size array
        size_arr = np.zeros((self.k))
        # all clusters
        clusters = self.clusters
        for i in range(self.k) :
            cluster_points = data[clusters == i]
            cluster_size = cluster_points.shape[0]
            size_arr[i] = cluster_size
        return size_arr


    # SSE
    # find inertia (sse)
    def inertia(self, data, clusters, centroids_arr) :
        # distances array
        dists_arr = np.zeros(data.shape[0])
        for i in range(self.k) :
            # clusterwise dists from centroid
            dists_arr[clusters == i] = np.linalg.norm(data[clusters == i] - centroids_arr[i], axis=1)
        # compute sse
        sse = np.sum(np.square(dists_arr))
        return sse


###################################################################################################################################
# EVALUATION AND VISUALIZATION
###################################################################################################################################
# Plot Clusters
# plots all clusters and their centroids with sizes
# takes in data retrieval function, k, and max_loop and forces pca to 2 to allow plotting
# also takes in x and y labels and title
def plot_clusters(data_fn, k_val, max_lp, x_label, y_label, title) :
    # clean data
    data = cleaned_data(data_fn, 2)
    # run k means
    km = k_means(k_val, max_lp)
    km.fit(data)
    # get centroids
    centroids = km.centroids
    # get sse
    sse = km.sse
    # generate list of colors for k clusters and centroid
    colors = rand_colors(k_val + 1)
    # plot data and centroids
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(0, k_val) :
        color = colors[i]
        cluster_label = 'cluster ' + str(i + 1) + ': size = ' + str(km.csizes[i])
        plt.scatter(data[km.clusters == i, 0], data[km.clusters == i, 1], c=color, label=cluster_label)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c=colors[-1], label='centroid')
    plt.legend()
    plt.figtext(0.02, 0.02, 'sse = ' + str(sse), fontsize = 10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    ax.set_aspect('equal')
    plt.show()

# randomly generate an n-length list of unique colors as rgb values
def rand_colors(n) :
    colors = []
    while len(colors) < n :
        color= "#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
        if color not in colors :
            colors.append(color)
    return colors

###################################################################################################################################
# Elbow Curve
# finds sses for range of k up to max k
# takes in data, max_k, max_loop
def elbow_plot(data, max_k, max_loop) :
    # find k_range_inertias
        inertia_list = k_range_inertias(data, max_k, max_loop)
        # extract ks for x-axis
        k_vals = inertia_list[0]
        # extract sses for y-axis
        sses = inertia_list[1]
        # plot sse over k
        plt.figure(figsize=(6, 6))
        plt.plot(k_vals, sses, '-o')
        plt.xlabel(r"Number of cluster *k*")
        plt.ylabel("Inertia (SSE)")
        # display results
        plt.show()

# runs k-means with a given range of ks, calculates inertia (sse) for each
# takes in data, max_k, max_loop
# where max_k is the upperbound of ks to test
# returns sses as k:sse dict
def k_range_inertias(data, max_k, max_lp) :
    # ks
    k_list = list(range(1, max_k))
    # sses
    sse_list = []
    # for each k in range, find inertia (sse) and add to list
    for n in k_list :
        km = k_means(k=n, max_loop=max_lp)
        km.fit(data)
        sse_list.append(km.sse)
    return [k_list, sse_list]


###################################################################################################################################
# MAIN FUNCTION
###################################################################################################################################
# parse user arguments and call elbow_plot or plot_clusters with given input
# required args: "elbow" or "cluster", and the dataset to retrieve, k (either to k-means input value or k_max, depending on 
# which function is being called)
# optional args: number of principle components, max_loop, x_label, y_label, and title
def parse_args() :
    # start by parsing subcommand
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("call_function", nargs="?", choices=['elbow', 'cluster'])
    parser.add_argument("dataset")
    parser.add_argument("k", type=int)
    parser.add_argument("--num_pc", type=int, default=2)
    parser.add_argument("--max_loop",type=int,  default=100)
    parser.add_argument("--x_label", default='x')
    parser.add_argument("--y_label", default='y')
    parser.add_argument("--title", default='title')
    # get args
    args = parser.parse_args()
    # call function with args
    # default to elbow
    call_function = "elbow" if args.call_function is None else args.call_function

    if call_function == "elbow" :
        in_data = cleaned_data(args.dataset, args.num_pc)
        elbow_plot(in_data, args.k, args.max_loop)
    elif call_function == "cluster" :
        plot_clusters(args.dataset, args.k, args.max_loop, args.x_label, args.y_label, args.title)
    else :
        print("Error: that is not a supported function")

# main function
if __name__ == '__main__' :
    parse_args()