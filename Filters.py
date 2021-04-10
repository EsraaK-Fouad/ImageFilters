#-------------------------------------------------------------------------------------------------------
'''                                 Filter Class                                                        '''
#----------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import math
#--------------------------------------------------------------------------
class Filter:
    def __init__(self,image ):
        self.image = image


#--------------------------------------------------------------------------------
#                                  Helper Function                             
#----------------------------------------------------------------------------  


    def normal_dist(self,x, mu, sigma):
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp  (-np.power((x - mu) / sigma, 2)) 
#--------------------------------------------------------------------------------
    def gaussian_kernel(self,size, sigma=1):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = self.normal_dist(kernel_1D[i], 0, sigma)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
        kernel_2D *= 1.0 / kernel_2D.max()
        return kernel_2D
#--------------------------------------------------------------------------------
    def convolution(self, kernel, average=False):
        
        if len(self.image.shape) == 3:
            print("Found 3 Channels : {}".format(self.image.shape))
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            print("Converted to Gray Channel. Size : {}".format(self.image.shape))
        else:
            print("Image Shape : {}".format(self.image.shape))
 
        print("Kernel Shape : {}".format(kernel.shape))
 
        
        image_row, image_col = self.image.shape
        kernel_row, kernel_col = kernel.shape
    
        output = np.zeros(self.image.shape)
    
        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)
    
        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = self.image
    
        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                if average:
                    output[row, col] /= kernel.shape[0] * kernel.shape[1]
    
        print("Output Image size : {}".format(output.shape))
    
        return output

#--------------------------------------------------------------------------------
#                                   Filters Function                             
#----------------------------------------------------------------------------  

    def gaussian_filter(self, kernel_size):
        kernel = self.gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
        gauss=self.convolution( kernel, average=True)
        cv2.imwrite('gauss_filt.jpg',gauss)
        return gauss
#------------------------------------------------------------------------
    def median_filter(self, filter_size):
        temp = []
        indexer = filter_size // 2
        median_filt=[]
        median_filt = np.zeros((len(self.image),len(self.image[0])))
        for i in range(len(self.image)):

            for j in range(len(self.image[0])):

                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(self.image) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(self.image[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(self.image[i + z - indexer][j + k - indexer])
                temp.sort()
                median_filt[i][j] = temp[len(temp) // 2]
                temp = []
        cv2.imwrite('median_filt.jpg', median_filt)
        return median_filt
#--------------------------------------------------------------------------------------------------
    def Averaging(self,size):
        mask=np.ones((size,size), dtype=int)
        mask=mask/(size*size)
        average= self.convolution( mask, average=False)
        cv2.imwrite('average_filt.jpg', average)
        return average

        



