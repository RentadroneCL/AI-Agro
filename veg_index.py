import numpy
#from skimage import io
import georasters as gr

epsilon = 0.00001

class Image_Multi():

    def __init__(self, path_red, path_green, path_blue, path_nir, path_rededge):
        self.path_red = path_red
        self.path_green = path_green
        self.path_blue = path_blue
        self.path_nir = path_nir
        self.path_rededge = path_rededge
        self.read_images()


    def read_images(self):
        self.im_red = gr.from_file(self.path_red)
        self.im_green = gr.from_file(self.path_green)
        self.im_blue = gr.from_file(self.path_blue)
        self.im_nir = gr.from_file(self.path_nir)
        self.im_rededge = gr.from_file(self.path_rededge)

    def NDVI(self):

        return numpy.divide(self.im_nir - self.im_red, self.im_nir + self.im_red + epsilon)

    def GNDVI(self):

        return numpy.divide(self.im_nir - self.im_green, self.im_nir + self.im_green + epsilon)

    def NDRE(self):

        return numpy.divide(self.im_nir - self.im_rededge, self.im_rededge + self.im_nir + epsilon)

    def LCI(self):

        return numpy.divide(self.im_nir - self.im_rededge, self.im_nir + self.im_red + epsilon)

    def OSAVI(self):

        return numpy.divide(self.im_nir - self.im_red, self.im_nir + self.im_red + 0.16 +epsilon)

    def OSAVI_16(self):

        return 1.6*numpy.divide(self.im_nir - self.im_red, self.im_nir + self.im_red + 0.16 +epsilon)
