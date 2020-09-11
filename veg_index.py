import numpy as np
#from skimage import io
import georasters as gr
from matplotlib import path
import cv2
import Utils
import scipy.ndimage as ndimage

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

        im_seg.list_P =  List_P

        return im_seg

    def subdivision_rect(self, split_Weight = 10, split_Height = 2, overlap = 0.01):
        ## subdivide image in rectangles, keep the perspective

        Points = np.array(self.list_P)
        Points_order = Utils.order_points_rect(Points)

        M, maxWidth, maxHeight = Utils.perspectiveTransform(Points)

        split_Weight, split_Height = 15, 3
        sub_division = Utils.subdivision_rect([split_Weight, split_Height], maxWidth, maxHeight, overlap)

        sub_division_origin = cv2.perspectiveTransform(np.array(sub_division), np.linalg.inv(M))

        return np.uint(sub_division_origin)

    def correction_subimage(self, List_P):
        ## Rotate subimage with the objective of lines farming be vetical in transform crop
        List_new_P = []
        for P in List_P:

            im = self.Segmentation(P)
            NDVI = im.NDVI().raster


            Points = np.array(im.list_P, np.float)

            M, maxWidth, maxHeight = Utils.perspectiveTransform(Utils.order_points_rect(Points))

            warped = cv2.warpPerspective(NDVI, M, (maxWidth, maxHeight))
            warped[np.isnan(warped)] = 0

            # Otsu
            blur = cv2.GaussianBlur(warped * 255,(5,5),0).astype('uint8')
            ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            n_important = 100
            skel_filter = Utils.skeleton(th3, n_important = 100)

            theta_prop = Utils.angle_lines(skel_filter,  n_important = 100, angle_resolution = 720,
                                           threshold = 100, min_line_length = 200,
                                           max_line_gap = 50, plot = False)

            center = (np.mean([point[0] for point in P]), np.mean([point[1] for point in P]))
            matrix = cv2.getRotationMatrix2D(center=center, angle= -theta_prop*180/np.pi, scale=1)

            new_P = cv2.transform(np.array([P]), matrix)[0]

            List_new_P.append(new_P.copy())

        return List_new_P

    def detector_lines(self,List_new_P,
                            th_NDVI = 0.6,
                            vertical_kernel_size_h = 10,
                            vertical_kernel_size_w = 5,
                            th_small_areas = 30,
                            lines_width = 1,
                            merge_bt_line = 10):
        ## Input List of points of subimages, output   List of lines crop . One line is (top_left, top_right, bottom_right, bottom_left)

        List_lines_origin_complete =  np.ones((0, 4, 2))

        for P in List_new_P:

            P = Utils.order_points_rect(P)
            im = self.Segmentation(P)

            Points = np.array(im.list_P)
            M_sub, maxWidth, maxHeight = Utils.perspectiveTransform(Utils.order_points_rect(Points))

            NDVI = cv2.warpPerspective(im.NDVI().raster, M_sub, (maxWidth, maxHeight))
            H2 = (NDVI > th_NDVI).astype('uint8')
            ### Create kernel rotate #########333

            kernel = np.ones((vertical_kernel_size_h, vertical_kernel_size_w) , np.uint8)  # note this is a vertical kernel
            #kernel = np.ones((5, 5) , np.uint8)

            erode = cv2.erode(H2,kernel)
            closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
            skel = Utils.skeleton(closing, n_important = -1)


            closing_skel = cv2.morphologyEx(skel.astype(float), cv2.MORPH_CLOSE, kernel)
            closing_skel = cv2.morphologyEx(closing_skel, cv2.MORPH_CLOSE, kernel)

            label_im, nb_labels = ndimage.label(closing_skel)#, structure= np.ones((2,2))) ## Label each connect region
            label_areas = np.bincount(label_im.ravel())[1:]


            L = np.zeros(label_im.shape)

            for i in range(nb_labels):
                if label_areas[i] > th_small_areas:
                    L[label_im == (i + 1) ] = 1

            L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, kernel)

            label_im, nb_labels = ndimage.label(L)#, structure= np.ones((2,2))) ## Label each connect region
            label_areas = np.bincount(label_im.ravel())[1:]

            List_Centroid_WH = []
            for i in range(nb_labels):

                I = np.zeros(label_im.shape)
                I[label_im == (i + 1)] = 1
                # calculate moments of binary image
                Moments = cv2.moments(I)
                # calculate x,y coordinate of center
                cX = int(Moments["m10"] / Moments["m00"])
                cY = int(Moments["m01"] / Moments["m00"])
                cnts, hierarchy = cv2.findContours(I.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                width = np.max([x[0][0] for x in cnts[0]]) - np.min([x[0][0] for x in cnts[0]])
                height = np.max([y[0][1] for y in cnts[0]]) - np.min([y[0][1] for y in cnts[0]])
                List_Centroid_WH.append((cX, cY, width, height))

            if List_Centroid_WH == []:
                continue

            x_min = np.min([x[0] - x[2]/2 for x in List_Centroid_WH])
            x_max = np.max([x[0] + x[2]/2 for x in List_Centroid_WH])
            y_min = np.min([y[1] - y[3]/2 for y in List_Centroid_WH])
            y_max = np.max([y[1] + y[3]/2 for y in List_Centroid_WH])

            L_xw = sorted([[x[0],x[2]] for x in List_Centroid_WH])

            ## Filter width separation
            th = np.mean(L_xw, axis=0)[1]
            L_filter = [L_xw[0][0]]
            L_width = [L_xw[0][1]]
            for i in range(len(L_xw)):
                if (abs(L_filter[-1] - L_xw[i][0]) > th):
                    L_filter.append(L_xw[i][0])
                    L_width.append(L_xw[i][1])

            #filter mean dif separation
            dif = [L_filter[i] - L_filter[i-1] for i in range(1, len(L_filter))]
            th = np.mean(dif).astype('int') + np.std(dif).astype('int') - merge_bt_line
            L_filter_2 = [L_filter[0]]
            L_width_2 = [L_width[0]]
            for i in range(len(L_filter)):
                if (abs(L_filter_2[-1] - L_filter[i]) > th):
                    L_filter_2.append(L_filter[i])
                    L_width_2.append(L_width[i])

            ####################### List of Polygons ################
            List_lines = [] #(top-left, top-right,bottom-right, bottom-left
            avg_width = lines_width #np.mean(L_width_2)/3#L_width[i]
            for i in range(len(L_filter_2)):


                top_left = (int(L_filter_2[i] - avg_width/2) , y_min)
                top_right = (int(L_filter_2[i] + avg_width/2) , y_min)
                bottom_right = (int(L_filter_2[i] + avg_width/2) , y_max)
                bottom_left = (int(L_filter_2[i] - avg_width/2) , y_max)
                if  int(L_filter_2[i] - avg_width/2) > L.shape[1]:
                    break

                List_lines.append((top_left, top_right, bottom_right, bottom_left))

            List_lines_origin = cv2.perspectiveTransform(np.array(List_lines), np.linalg.inv(M_sub)) # In subdivide image
            List_lines_origin_complete = np.concatenate((List_lines_origin_complete,
                                                        List_lines_origin - Utils.order_points_rect(Points)[0] + Utils.order_points_rect(P)[0]))# In image complet # Put line in big image

        return List_lines_origin_complete
