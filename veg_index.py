import numpy as np
#from skimage import io
import georasters as gr
from matplotlib import path
import cv2

epsilon = 0.00001

class Image_Multi():

    def __init__(self, path_red = None, path_green = None, path_blue = None, path_nir = None, path_rededge = None):
        self.path_red = path_red
        self.path_green = path_green
        self.path_blue = path_blue
        self.path_nir = path_nir
        self.path_rededge = path_rededge

        if not(None in [path_red, path_green, path_blue, path_nir, path_rededge]):
            self.read_images()


    def read_images(self):
        self.im_red = gr.from_file(self.path_red)
        self.im_green = gr.from_file(self.path_green)
        self.im_blue = gr.from_file(self.path_blue)
        self.im_nir = gr.from_file(self.path_nir)
        self.im_rededge = gr.from_file(self.path_rededge)
        self.load_List_P()


    def load_images(self, im_red, im_green, im_blue, im_nir, im_rededge ):
        self.im_red = im_red
        self.im_green = im_green
        self.im_blue = im_blue
        self.im_nir = im_nir
        self.im_rededge = im_rededge
        self.load_List_P()

    def load_List_P(self):

        # Search Points of polygon
        countours, hierarchy = cv2.findContours(np.uint8(np.isnan(self.im_red.raster)), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        epsilon = 0.01 * cv2.arcLength(countours[np.argmax([ctln.shape[0] for ctln in countours])], True)
        approx = cv2.approxPolyDP(countours[np.argmax([ctln.shape[0] for ctln in countours])], epsilon, True)
        List_P = [(app[0][0], app[0][1]) for app in approx]
        center = (np.mean([point[0] for point in List_P]), np.mean([point[1] for point in List_P]))
        List_P = sorted(List_P, key = lambda point: (-np.pi * 3/4 - np.arctan2((point[1] - center[1]), (point[0] - center[0]))) % 2*np.pi)
        self.list_P = List_P

    def list_images(self):

        return [self.im_red , self.im_green, self.im_blue ,self.im_nir, self.im_rededge]

    def NDVI(self):

        return np.divide(self.im_nir - self.im_red, self.im_nir + self.im_red + epsilon)

    def GNDVI(self):

        return np.divide(self.im_nir - self.im_green, self.im_nir + self.im_green + epsilon)

    def NDRE(self):

        return np.divide(self.im_nir - self.im_rededge, self.im_rededge + self.im_nir + epsilon)

    def LCI(self):

        return np.divide(self.im_nir - self.im_rededge, self.im_nir + self.im_red + epsilon)

    def OSAVI(self):

        return np.divide(self.im_nir - self.im_red, self.im_nir + self.im_red + 0.16 +epsilon)

    def OSAVI_16(self):

        return 1.6*np.divide(self.im_nir - self.im_red, self.im_nir + self.im_red + 0.16 +epsilon)

    def RGB(self, lim = 4000):
        #Function return RGB GeoRaster

        # Bounded values
        Z = np.zeros((self.im_red.raster.shape[0], self.im_red.raster.shape[1], 3))
        Z[:,:,0] = self.im_red.raster.copy()
        Z[:,:,1] = self.im_green.raster.copy()
        Z[:,:,2] = self.im_blue.raster.copy()

        Z[Z[:, :, 0] > lim] = lim
        Z[Z[:, :, 1] > lim] = lim
        Z[Z[:, :, 2] > lim] = lim
        Z[:, :, 0] = Z[:, :, 0] / (np.nanmax(Z[:, :, 0]) - np.nanmin(Z[:, :, 0]))
        Z[:, :, 1] = Z[:, :, 1] / (np.nanmax(Z[:, :, 1]) - np.nanmin(Z[:, :, 1]))
        Z[:, :, 2] = Z[:, :, 2] / (np.nanmax(Z[:, :, 2]) - np.nanmin(Z[:, :, 2]))
        Z[np.isnan(self.im_red.raster)] = np.nan

        (xmin, xsize, x, ymax, y, ysize) = self.im_red.geot

        return gr.GeoRaster(Z.copy(),(xmin, xsize, x, ymax, y, ysize),
                            nodata_value=self.im_red.nodata_value,
                            projection=self.im_red.projection,
                            datatype=self.im_red.datatype)

    def Segmentation(self, List_P):

        #List_P= [(P1_x, P1_y), (P2_x, P2_y), (P3_x, P3_y)] # Pixels not geocordinate
        # Sort Order of polygon points
        center = (np.mean([point[0] for point in List_P]), np.mean([point[1] for point in List_P]))
        List_P = sorted(List_P, key = lambda point: ((-np.pi * 3/4) - np.arctan2((point[1] - center[1]), (point[0] - center[0]))) % (2*np.pi))

        eps = 5

        while (min([f[0] for f in List_P]) - eps < 0) or (min([f[1] for f in List_P]) - eps < 0) or (max([f[1] for f in List_P]) + eps > self.im_red.raster.shape[0]) or (max([f[0] for f in List_P]) + eps > self.im_red.raster.shape[1]):
            eps -= 1
            #print("Activado epsilon")

        x_rect = np.uint(min([f[0] for f in List_P]) - eps)
        y_rect = np.uint(min([f[1] for f in List_P]) - eps)
        h_rect = np.uint(max([f[1] for f in List_P]) - min([f[1] for f in List_P]) + 2 * eps)
        w_rect = np.uint(max([f[0] for f in List_P]) - min([f[0] for f in List_P]) + 2 * eps)

        List_P = [(x - x_rect, y - y_rect) for x,y in List_P]
        poly = path.Path(List_P)

        xv,yv = np.meshgrid(range(w_rect), range(h_rect))
        flags = ~poly.contains_points(np.hstack((xv.flatten()[:,np.newaxis], yv.flatten()[:,np.newaxis])))

        list_rasters  = []

        for  Im in self.list_images():

            (xmin, xsize, x, ymax, y, ysize) = Im.geot
            I = Im.raster[y_rect: y_rect + h_rect, x_rect : x_rect + w_rect].copy()
            I[flags.reshape(I.shape)] = np.nan
            new_Im = gr.GeoRaster(cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value= np.nan),
                                 (xmin + xsize * (w_rect + 1), xsize, x, ymax + ysize * (h_rect + 1), y, ysize),
                                 nodata_value=Im.nodata_value,
                                 projection=Im.projection,
                                 datatype=Im.datatype)

            #new_Im.raster[flags.reshape(new_Im.raster.shape)] = np.nan

            list_rasters.append(new_Im.copy())




        im_seg = Image_Multi()
        im_seg.load_images(im_red = list_rasters[0], im_green = list_rasters[1], im_blue = list_rasters[2],
                           im_nir = list_rasters[3], im_rededge = list_rasters[4])

        return im_seg
