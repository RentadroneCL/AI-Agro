import numpy as np
#from skimage import io
import georasters as gr
from matplotlib import path

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

    def load_images(self, im_red, im_green, im_blue, im_nir, im_rededge ):
        self.im_red = im_red
        self.im_green = im_green
        self.im_blue = im_blue
        self.im_nir = im_nir
        self.im_rededge = im_rededge

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

    def Segmentation(self, List_P):
        #List_P= [(P1_x, P1_y), (P2_x, P2_y)]
        epsilon = 0


        x_rect = min([f[0] for f in List_P]) - epsilon
        y_rect = min([f[1] for f in List_P]) - epsilon
        h_rect = max([f[1] for f in List_P]) - min([f[1] for f in List_P]) + 2 * epsilon
        w_rect = max([f[0] for f in List_P]) - min([f[0] for f in List_P]) + 2 * epsilon

        List_P = [(x - x_rect, y - y_rect) for x,y in List_P]
        poly = path.Path(List_P)

        xv,yv = np.meshgrid(range(w_rect), range(h_rect))
        flags = ~poly.contains_points(np.hstack((xv.flatten()[:,np.newaxis], yv.flatten()[:,np.newaxis])))

        list_rasters  = []

        for  Im in self.list_images():

            (xmin, xsize, x, ymax, y, ysize) = Im.geot

            new_Im = gr.GeoRaster(Im.raster[y_rect: y_rect + h_rect, x_rect : x_rect + w_rect].copy(),
                                 (xmin + xsize * w_rect, xsize, x, ymax + ysize * h_rect, y, ysize),
                                 nodata_value=Im.nodata_value,
                                 projection=Im.projection,
                                 datatype=Im.datatype)

            new_Im.raster[flags.reshape(new_Im.raster.shape)] = np.nan

            list_rasters.append(new_Im.copy())




        im_seg = Image_Multi()
        im_seg.load_images(list_rasters[0], list_rasters[1], list_rasters[2], list_rasters[3], list_rasters[4])

        return im_seg
