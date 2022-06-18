import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import zipfile
import tarfile
import urllib

iter = 0
bool = False
while iter < 10 and not bool :
    print(iter)
    bool = (iter == 5)
    iter += 1


""" #https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip

# importing dataset
# https://realpython.com/k-means-clustering-python/
op_psyc_url = "https://openpsychometrics.org/_rawdata/"
archive_name = "DASS_data_21.02.19.zip"

# Build the url
full_download_url = urllib.parse.urljoin(op_psyc_url, archive_name)
 
# Download the file
r = urllib.request.urlretrieve (full_download_url, archive_name)

# Extract the data from the archive
with zipfile.ZipFile(archive_name, 'r') as zip_ref:
    zip_ref.extractall()


# retrieve data as a numpy array
datafile = "/home/jrhyner/Documents/CS2810/final_project/DASS_data_21.02.19/data.csv"

# generate column list -> only want numerical data cols (first 40 questions)
cols = []
for q in range(1, 43) :
    a_col = "Q" + str(q) + "A"
    i_col = "Q" + str(q) + "I"
    e_col = "Q" + str(q) + "E"
    cols.extend([a_col, i_col, e_col])
print(cols)

print(len(cols))

data = pd.read_csv(datafile, sep='\t', usecols=cols)
print(data.head())

# 126 is all dass questions, positions, and lapses -> update usecols range? -> 42*3
# +1 bc how range works --> 127
data2 = np.genfromtxt(
    datafile,
    delimiter="\t",
    usecols=range(0, len(cols) + 1),
    skip_header=1,
    dtype=int
)

rounded_data = np.round(data2, 2)

print("no rounding")
print(data2[:5, :3])
print("rounding")
print(rounded_data[:5, :3]) """

####################################################################

""" 
# importing dataset
# https://realpython.com/k-means-clustering-python/
uci_tcga_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
archive_name = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"

# Build the url
full_download_url = urllib.parse.urljoin(uci_tcga_url, archive_name)
 
# Download the file
r = urllib.request.urlretrieve (full_download_url, archive_name)

# Extract the data from the archive
tar = tarfile.open(archive_name, "r:gz")
tar.extractall()
tar.close()

# retrieve data as a numpy array
datafile = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"

data = np.genfromtxt(
    datafile,
    delimiter=",",
    usecols=range(1, 20532),
    skip_header=1
)

true_label_names = np.genfromtxt(
    labels_file,
    delimiter=",",
    usecols=(1,),
    skip_header=1,
    dtype="str"
)

print(data[:5, :3]) """


##################################################
""" # MinMax scaling

#Get the IRIS dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
 
#prepare the data
X = data.iloc[:,0:4]

#prepare the target
y = data.iloc[:,4]

def min_max_scale(data) :
    # min and max values of data
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

    # use min and max to scale data
    scaled_data = (data - data_min) / (data_max - data_min)

    return scaled_data

X_scaled = min_max_scale(X)

print("unscaled:")
print(X.head())

print("scaled:")
print(X_scaled.head())


fig, axes = plt.subplots(1,2)

axes[0].scatter(X[:,0], X[:,1], c=y)
axes[0].set_title("Original data")

axes[1].scatter(X_scaled[:,0], X_scaled[:,1], c=y)
axes[1].set_title("MinMax scaled data")

plt.show() """


##################################################

""" # PCA

def PCA(X , num_components):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced 


mat_reduced = PCA(data2 , 2)

print(mat_reduced) """

"""
 
#Get the IRIS dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
 
#prepare the data
x = data.iloc[:,0:4]
 
#prepare the target
target = data.iloc[:,4]

A = np.array([[1, 2], [3, 4], [5, 6]])
 
#Applying it to PCA function
mat_reduced = PCA(A , 2)

print(mat_reduced)
 
#Creating a Pandas DataFrame of reduced Dataset
principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
 
#Concat it with target variable to create a complete Dataset
principal_df = pd.concat([principal_df , pd.DataFrame(target)] , axis = 1)

#print(principal_df.head())

# showing results
plt.figure(figsize = (6,6))
sb.scatterplot(data = principal_df , x = 'PC1',y = 'PC2' , hue = 'target' , s = 60 , palette= 'icefire')
plt.show()  """

################################################################

""" # PCA 2
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# define a matrix
A = np.array([[1, 2], [3, 4], [5, 6]])
print("A")
print(A)
# calculate the mean of each column
M = np.mean(A.transpose(), axis=1)
print("M")
print(M)
# center columns by subtracting column means
C = A - M
print("C")
print(C)
# calculate covariance matrix of centered matrix
V = np.cov(C.transpose())
print("V")
print(V)
# eigendecomposition of covariance matrix
values, vectors = np.linalg.eigh(V)
print("vects")
print(vectors)
print("vals")
print(values)
# project data
P = np.dot(vectors.transpose() , C.transpose()).transpose()
print("P")
print(P) """

##########################################
"""  # k-means

# random chooosing
# https://www.kite.com/python/answers/how-to-select-random-rows-from-a-numpy-array-in-python
test_arr = np.asarray([[1,2,3], [5,6,7], [8,9,0]])

number_of_rows = test_arr.shape[0]

#Select 2 random indices
random_indices = np.random.choice(number_of_rows, size=2, replace=False)

#Select 2 random rows
random_rows = test_arr[random_indices, :]

#print(random_rows)


# testing min
# https://stackoverflow.com/questions/3282823/get-the-key-corresponding-to-the-minimum-value-within-a-dictionary
d = {"one": 12, "two": 4, "three": 6}
smallest = min(d, key=d.get)
#print(smallest)

# testing key retrieval
# https://stackoverflow.com/questions/42438808/finding-all-the-keys-with-the-same-value-in-a-python-dictionary

# initializing dict
test_dict = {"Jim": "y", "Bob": "y", "Ravioli": "n"}
  
# printing original dict
#print("The original dict is : " + str(test_dict))
  
# Get keys of particular value in dictionary
res = [ k for k,v in test_dict.items() if v == 'y']
  
# printing result 
#print("The keys corresponding to value : " + str(res))


# array referencing
arr1 = np.asarray([1,2,3])
#print("arr1 before: " + str(arr1))
past_arr1 = arr1
#print("past_arr1 before: " + str(past_arr1))
new_arr1 = np.asarray([4,5,6])
arr1 = new_arr1
#print("arr1 after: " + str(arr1))
#print("past_arr1 after: " + str(past_arr1))
#print(past_arr1 == arr1) """