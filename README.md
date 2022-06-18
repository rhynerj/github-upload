# READ ME
## K-means Project

This repository stores files for the final project in a Mathematics of Data Models course.
Includes:
* code for extracting and cleaning a genetic and a numerical dataset
* implentation of principal component analysis to reduce data dimensionality
* implementation of k-means algorithms to conduct unsupervised machine learning on data
* code for visualizations of results
* final report interpreting results

## Imprementation Instructions

Once downloaded, the k-means.py file can be run in the command line. Using argparse, the main function takes in three required arguments, in order: function to run ("elbow" or "cluster"), the name of the dataset to retrieve, and k (k represents the maximum k to try if the elbow function is being called or the k value to use in k-means if the cluster function is being called). The main function also takes in five optional arguments: number of principle components (default: 2), maximum times to loop in k-means (default: 100), x-axis label (default: x), y-axis label (default: y), and title (default: title). The number of principle components is only adjustable with the elbow function, while the axis and title labels are only used with the cluster function.

If the elbow function is called, input data is set to the specified dataset and the clean function is called. The cleaned_data function takes in a dataset name, and uses an if statement to call the retrieval function associated with that dataset. The retrieval function for each dataset downloads that dataset from its online location, extracts the files within, and reads the data file into an array. This file downloading portion was hard-coded for testing purposes, and unfortunately, I ran out of time to fix this, so the command-line arguments only works for the two datasets I worked with. To run the code with another dataset, the cleaned_data function could be updated with another if statement to retrieve data using custom clean functions to parse that data into a numpy array. The min-max scaling, PCA, and k-means portions of the code all accept numpy arrays, so the code should work as long as the data retrieval function reads the data into a numpy array.

In the cleaned_data function, the data is scaled using min-max scaling, and the specific number of principle components is chosen. Under the elbow cluster condition, preprocessed data is passed into elbow_plot function, which runs k-means on the data using k-values ranging from zero to the given maximum k-value. The inertia (SSE) for each value of k is determined (by creating a k-means instance, running the fit function on it, and extracting the SSE value) and k-values and inertias are then plotted on a line graph, which can be used to identify the optimal value of k.

Using this value of k, k-means clustering can then be visualized and assessed using the plot_clusters function. The plot_clusters function is called when “cluster” is passed into the command line. The plot_clusters function calls cleaned_data internally. This was intended to force the number of principle components selected to two so that the data points could be plotted, and I ran out of time to find a better way to do this. The plot_clusters function creates a k-means instance with the specified k and uses that to plot all data points colored by cluster (using a randomly generated unique color) and the centroids. The plot also displays the number of data points included in each cluster in the legend and the SSE at the bottom of the plot.

Due to time constraints, this is the only functionality I was able to successfully implement in the main function. However, the “Preprocessing” and “K-means” sections are very modular and can easily be run on any input data that has been formatted into a numpy array. These sections are further explained below:

### Preprocessing
This section includes the min_max_scale and pca functions. The min_max_scale function normalizes the given data using min-max scaling. It is modeled after  MinMaxScaler() in sklearn. The pca function conducts PCA on the given data and returns the data reduced to the specific number of components.

### K-means
This section includes the k-means algorithm as a class. An instance of k-means is initialized with a specific k, and an optional maximum looping parameter. The default for max_loop is 100. Once a k-means instance has been created, the k-means algorithm is run be calling k-means.fit, which takes in data as a numpy array. It randomly selects k rows in the data array as initial centroids. It calculates the new centroids by calculating the distance from each data point to each centroid and storing that in an array, finding the minimums of that array, and the using those minimums to update each centroid to the mean of the data points assigned to its cluster. The cluster assignments and centroids are both stored as attributes of the k-means instance for later retrieval. Looping continues until the centroids at the start of the loop match the calculated mean centroids or the max number of loops is reached. Once the loop is finished, the inertia (SSE) is calculated and the cluster sizes are determined and both stored as k-means attributes.
