
# Python program to print prime factors

import math
from sklearn import cluster
from skimage import data
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal

def factors_number(n):
# A function to print all prime factors of
# a given number n

    n2 = int(n/2)
    fn = []
    for x in range(2,n2+1):
        if n%x == 0:
            if not([int(n/x), int(x)] in fn):
                fn.append([int(x), int(n/x)]) #lower member of factor pair #upper member of factor pair
    if fn == []: fn.append([1, n])
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
    #Kmeans in Image with dimensional position
    #n_cluster is the number of n_clusters
    #weight_positional is you can put weights to coordinates, i.e. Kmeans with two more dimensionas.

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


def order_points_rect(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right,
    # the third is the bottom-right, and
    #the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def perspectiveTransform(Points):
    #Transform cuadrilater image segmentation to rectangle image
    # Return Matrix Transform
    Points = np.array(Points)
    Points_order = order_points_rect(Points)
    #dst = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]], dtype = "float32")

    (tl, tr, br, bl) = Points_order
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth , 0],
        [maxWidth , maxHeight ],
        [0, maxHeight ]], dtype = "float32")

    M = cv2.getPerspectiveTransform(Points_order, dst)
    return M, maxWidth, maxHeight

def subdivision_rect(factors, maxWidth, maxHeight, merge_percentaje = 0):
    ## From a rect (top-left, top-right, bottom-right, bottom-left) subidive in rectangle

    #factors = factors_number(n_divide)[-1] # First factor is smaller

    #if maxWidth > maxHeight:
    #    split_Width = [maxWidth / factors[1] * i for i in range(factors[1] + 1)]
    #    split_Height = [maxHeight / factors[0] * i for i in range(factors[0] + 1)]
    #else:
    #    split_Width = [maxWidth / factors[0] * i for i in range(factors[0] + 1)]
    #    split_Height = [maxHeight / factors[1] * i for i in range(factors[1] + 1)]
    merge_Width = maxWidth * merge_percentaje
    merge_Height = maxHeight * merge_percentaje
    split_Width = [maxWidth / factors[0] * i for i in range(factors[0] + 1)]
    split_Height = [maxHeight / factors[1] * i for i in range(factors[1] + 1)]

    sub_division = []
    for j in range(len(split_Height) - 1):
        for i in range(len(split_Width) - 1):

            sub_division.append([(max(split_Width[i] - merge_Width, 0) , max(split_Height[j] - merge_Height , 0)),
                                 (min(split_Width[i+1] + merge_Width , maxWidth - 1), max(split_Height[j] - merge_Height , 0)),
                                 (min(split_Width[i+1] + merge_Width , maxWidth - 1), min(split_Height[j+1] + merge_Height, maxHeight - 1)),
                                 (max(split_Width[i] - merge_Width, 0),  min(split_Height[j+1] + merge_Height, maxHeight - 1))])

    return np.array(sub_division)


def skeleton(bin_image, n_important = 100):
    #From binary image (0,255) transform to skeleton edge

    kernel_size = 3
    edges = cv2.GaussianBlur((bin_image.copy()).astype(np.uint8),(kernel_size, kernel_size),0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))
    height,width = edges.shape
    skel = np.zeros([height,width],dtype=np.uint8)      #[height,width,3]
    temp_nonzero = np.count_nonzero(edges)

    while (np.count_nonzero(edges) != 0 ):
        eroded = cv2.erode(edges,kernel)
        #cv2.imshow("eroded",eroded)
        temp = cv2.dilate(eroded,kernel)
        #cv2.imshow("dilate",temp)
        temp = cv2.subtract(edges,temp)
        skel = cv2.bitwise_or(skel,temp)
        edges = eroded.copy()

    """This function returns the count of labels in a mask image."""
    label_im, nb_labels = ndimage.label(skel)#, structure= np.ones((2,2))) ## Label each connect region
    label_areas = np.bincount(label_im.ravel())[1:]
    keys_max_areas = np.array(sorted(range(len(label_areas)), key=lambda k: label_areas[k], reverse = True)) + 1
    keys_max_areas = keys_max_areas[:n_important]
    L = np.zeros(label_im.shape)
    for i in keys_max_areas:
        L[label_im == i] = i

    labels = np.unique(L)
    label_im = np.searchsorted(labels, L)

    return label_im>0

def angle_lines(skel_filter, n_important = 100, angle_resolution = 360, threshold = 100, min_line_length = 200, max_line_gap = 50, plot = False):
    #Measure the angle of lines in skel_filter. Obs the angle is positive in clockwise.

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / angle_resolution  # angular resolution in radians of the Hough grid
    #threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    #min_line_length = 200  # minimum number of pixels making up a line
    #max_line_gap = 50  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLines(np.uint8(skel_filter),rho, theta, threshold)
    lines_P = cv2.HoughLinesP(np.uint8(skel_filter),rho, theta, threshold, np.array([]) ,min_line_length, max_line_gap)

    theta_P = [np.pi/2 + np.arctan2(line[0][3] - line[0][1],line[0][2]-line[0][0])  for line in lines_P[:n_important]]

    theta = lines[0:n_important,0,1]

    h = np.histogram(np.array(theta), bins = angle_resolution, range=(-np.pi,np.pi))
    peaks = signal.find_peaks_cwt(h[0], widths= np.arange(2,4))
    h_P = np.histogram(np.array(theta_P), bins = angle_resolution, range=(-np.pi,np.pi))
    peaks_P = signal.find_peaks_cwt(h_P[0], widths= np.arange(2,4))

    #h= np.histogram(np.array(theta), bins = angle_resolution, range=(-np.pi,np.pi))
    #peaks = argrelextrema(h[0], np.greater)
    #h_P = np.histogram(np.array(theta_P), bins = angle_resolution, range=(-np.pi,np.pi))
    #peaks_P = argrelextrema(h_P[0], np.greater)

    mesh = np.array(np.meshgrid(h[1][peaks], h_P[1][peaks_P]))
    combinations = mesh.T.reshape(-1, 2)
    index_min = np.argmin([abs(a-b) for a,b in combinations])
    theta_prop = np.mean(combinations[index_min])

    if plot:
        print('Theta in HoughLines: ', h[1][peaks])
        print('Theta in HoughLinesP: ', h_P[1][peaks_P])
        print('combinations: ', combinations)
        print('Theta prop: ', theta_prop)


        Z1 = np.ones((skel_filter.shape))*255
        Z2 = np.ones((skel_filter.shape))*255

        for line in lines[0:n_important]:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            #print((x1,y1,x2,y2))
            cv2.line(Z1,(x1,y1),(x2,y2),(0,0,255),2)

        for line in lines_P[:n_important]:
            x1,y1,x2,y2 = line[0]
            cv2.line(Z2,(x1,y1),(x2,y2),(0,0,255),2)

        plt.figure(0)
        plt.figure(figsize=(16,8))

        plt.imshow(skel_filter)
        plt.title('Skel_filter')

        fig, axs = plt.subplots(1, 2, figsize=(16,8))
        axs[0].imshow(Z1)
        axs[0].title.set_text('Lines HoughLines')

        axs[1].imshow(Z2)
        axs[1].title.set_text('Lines HoughLinesP')

        fig, axs = plt.subplots(1, 2, figsize=(16,8))
        axs[0].hist(lines[0:n_important,0,1], bins = 45, range=[-np.pi,np.pi])
        axs[0].title.set_text('Lines  HoughLines theta Histogram')


        axs[1].hist(theta_P, bins = 45, range=[-np.pi,np.pi])
        axs[1].title.set_text('Lines HoughLinesP theta Histogram')
        #print(lines.shape)
        #print(lines_P.shape)


    return theta_prop

def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

def lines2circles(List_lines, r_circle = 10, n_important = -1):

    centers_filter = np.ones((0, 2))
    centers = np.ones((0, 2))

    for i,Poly in enumerate(List_lines[:n_important]):

        P1 = np.mean(Poly[0:2], axis=0)
        P2 = np.mean(Poly[2:4], axis=0)

        distance = np.linalg.norm(P1-P2)
        n_step = int(distance/(r_circle *2 + 1))

        centers = np.concatenate((centers, np.array([P1 * t/n_step + P2 * (n_step-t)/n_step for t in range(n_step + 1)]).astype(int)))

    centers = (np.unique(np.uint(centers), axis = 0)).astype(np.float)

        ### Filter circle so close
    centers_filter = np.concatenate((centers_filter, np.array([centers[0]])))

    for c in centers[1:]:
        if np.min(np.linalg.norm(c - centers_filter, axis=1)) > 2 * r_circle:

            centers_filter = np.concatenate((centers_filter,np.array([c])))


    return  np.uint(np.array(centers_filter))
