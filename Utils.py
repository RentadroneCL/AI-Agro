
# Python program to print prime factors

import math
from sklearn import cluster
from skimage import data
import numpy as np

def Factors(n):
# A function to print all prime factors of
# a given number n

    n2 = int(n/2)
    fn = []
    for x in range(2,n2+1):
        if n%x == 0:
            if not([int(n/x), int(x)] in fn):
                fn.append([int(x), int(n/x)]) #lower member of factor pair #upper member of factor pair
    return fn

def rgb2gray(rgb):
# RGB array  to gray
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def km_clust(array, n_clusters):
## KMeans Segmentation Image

    # Create a line array, the lazy way
    if len(array.shape) == 3:
        X = array.reshape((-1, array.shape[-1]))
    else :
        X = array.reshape((-1, 1))


    # Define the k-means clustering problem
    k_m = cluster.KMeans(n_clusters = n_clusters, n_init=4)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_

    return(values, labels, k_m)

def kmeans_image(Z, n_clusters, weight_positional = 0):
    #Kmeans in Image

    grid = np.indices((Z.shape[0], Z.shape[1]))
    grid = grid.astype(float)
    grid[0] = grid[0] / grid[0].max() * weight_positional
    grid[1] = grid[1] / grid[1].max() * weight_positional

    if len(Z.shape) == 3:

        img = np.zeros((Z.shape[0], Z.shape[1], Z.shape[2] + 2))
        img[:, :, :3] = Z#[:,:,1]
        img[:, :, 3] = grid[0]
        img[:, :, 4] = grid[1]

    else:
        
        img = np.zeros((Z.shape[0], Z.shape[1], 3))
        img[:, :, 0] = Z#[:,:,1]
        img[:, :, 1] = grid[0]
        img[:, :, 2] = grid[1]



    # Group similar grey levels using 8 clusters
    values, labels, k_m = km_clust(img, n_clusters = n_clusters)
    print (values.shape)
    print (labels.shape)
    # Create the segmented array from labels and values
    img_segm = np.array([values[label][:3] for label in labels]) #np.choose(labels, values)
    # Reshape the array as the original image
    img_segm.shape = img[:,:,:3].shape

    return img_segm

def _main_():

    print("Número a descomponer: ", end="")
    nombre = int(input())
    print(Factors(nombre))

if __name__ == '__main__':

    _main_()
