
# _*_ coding: utf-8 _*_
from design import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QImage,QPixmap
import cv2
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt
import numpy as np
import sys
from PyQt5.QtWidgets import*
import pyqtgraph as pg
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from NOISE import Noise
from Filters import Filter 
from design import Ui_MainWindow

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.Hth=0.1
        self.Lth=0.01
        self.alpha=0.3
        self.ui.pushButton_Noise.clicked.connect(self.Browse)
        self.ui.comboBox_Noise.currentIndexChanged.connect(self.Noise_type)
        self.ui.horizontalSlider_Noise.valueChanged.connect(self.Noise_type)
        self.ui.horizontalSlider_Kernel.valueChanged.connect(self.Noise_type)

        self.ui.horizontalSlider_2.setMinimum(0)
        self.ui.horizontalSlider_2.setMaximum(9)
        self.ui.horizontalSlider.setMinimum(0)
        self.ui.horizontalSlider.setMaximum(9)
        self.ui.pushButton.clicked.connect(lambda:self.LoadImage(self.ui.input))
        self.ui.pushButton_2.clicked.connect(lambda:self.LoadImage(self.ui.input_5))
        self.ui.pushButton_5.clicked.connect(lambda:self.Whichbtn( self.ui.pushButton_5))
        self.ui.pushButton_4.clicked.connect(lambda:self.Whichbtn( self.ui.pushButton_4))

        self.ui.pushButton_12.clicked.connect(lambda: self.draw_colored_image_histogram_or_distribution_curve(self.ui.pushButton_12))
        self.ui.pushButton_11.clicked.connect(lambda: self.draw_colored_image_histogram_or_distribution_curve(self.ui.pushButton_11))


        self.ui.pushButton_7.clicked.connect(self.drawHistogram)
        self.ui.pushButton_6.clicked.connect(self.Hybrid)
        self.ui.pushButton_3.clicked.connect(self.ApplyFilter)
        self.ui.horizontalSlider.valueChanged.connect(self.lowThreshold)
        self.ui.horizontalSlider.valueChanged.connect(self.ApplyFilter)
        self.ui.horizontalSlider_2.valueChanged.connect(self.HighThreshold)
        self.ui.horizontalSlider_2.valueChanged.connect(self.ApplyFilter)

        self.ui.horizontalSlider_3.setMinimum(10)
        self.ui.horizontalSlider_3.setMaximum(100)
        self.ui.horizontalSlider_3.setValue(10)
        self.ui.horizontalSlider_3.setSingleStep(10)
        self.ui.horizontalSlider_3.valueChanged.connect(self.slider_change)

        self.flage =0
        self.ui.pushButton_9.clicked.connect(lambda: self.LoadImage(self.ui.input_9))

        self.ui.comboBox_2.currentText()
        self.ui.comboBox_2.currentTextChanged.connect(self.LPFfunction)
        self.ui.comboBox_3.currentText()
        self.ui.comboBox_3.currentTextChanged.connect(self.HPFfunction)

#-------------------------------------------------------------------------------------------------------
#                                         Noise & Filters 
#-------------------------------------------------------------------------------------------------------




    def Browse(self):
        self.filepath = QtWidgets.QFileDialog.getOpenFileName()
        self.path = self.filepath[0]
        self.image_noise = cv2.imread(self.path,0)
        pixmap=self.display_image(self.image_noise )
        self.ui.view_image.setPixmap(pixmap)
        self.ui.view_image.setScaledContents(True)
    def display_image(self, arr):
        image_arr = np.array(arr).astype(np.int8)
        qimage = QtGui.QImage(image_arr,image_arr.shape[1], image_arr.shape[0],QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(qimage)
        return pixmap
    #*************************************************************************************
    def Noise_type(self):
        string=str(self.ui.comboBox_Noise.currentText())
        self.noise=Noise(self.image_noise )
        if string=="Uniform Noise":
            self.ui.view_Noise.clear()
            self.data=self.noise.uniform_noise(0,self.ui.horizontalSlider_Noise.value())
            pixmap=self.display_image(self.data)
            self.ui.view_Noise.setPixmap(pixmap)
            self.ui.view_Noise.setScaledContents(True)
          
        if string=="Gaussian Noise":
            self.ui.view_Noise.clear()
            self.data=self.noise.gaussian_noise(self.ui.horizontalSlider_Noise.value())
            pixmap=self.display_image(self.data)
            self.ui.view_Noise.setPixmap(pixmap)
            self.ui.view_Noise.setScaledContents(True)
        if string=="Salt & Pepper Noise" :
            self.ui.view_Noise.clear()
            self.data=self.noise.salt_and_pepper(self.ui.horizontalSlider_Noise.value()/1000)
            pixmap=self.display_image(self.data)
            self.ui.view_Noise.setPixmap(pixmap)
            self.ui.view_Noise.setScaledContents(True)
        if string=="Noise type":
            self.ui.view_Noise.clear()
            self.data=self.image_noise 
        
        if string=="original  image":
            self.ui.view_Noise.clear()
            self.data=self.image_noise 
            pixmap=self.display_image(self.image_noise )
            self.ui.view_Noise.setPixmap(pixmap)
            self.ui.view_Noise.setScaledContents(True)
        self.Filter_type(self.data) 
#*************************************************************************************
    def Filter_type(self,noisy_image):
        if (self.ui.horizontalSlider_Kernel.value()% 2 == 1):
            Value=self.ui.horizontalSlider_Kernel.value()
        else:
            Value=self.ui.horizontalSlider_Kernel.value()+1
    
        filter_type=Filter(noisy_image)
        guassian=filter_type.gaussian_filter( Value)
        pixmap=self.display_image(guassian)
        self.ui.view_guassian.setPixmap(pixmap)
        self.ui.view_guassian.setScaledContents(True)
        averaging=filter_type.Averaging( Value)
        pixmap1=self.display_image(averaging)
        self.ui.view_averaging.setPixmap(pixmap1)
        self.ui.view_averaging.setScaledContents(True)
        median=filter_type.median_filter( Value)
        pixmap2=self.display_image(median)
        self.ui.view_median.setPixmap(pixmap2)
        self.ui.view_median.setScaledContents(True)




#-------------------------------------------------------------------------------------------------------
#                                       
#-------------------------------------------------------------------------------------------------------


    def LoadImage(self,Label):
        self.flage = 1
        filename = QFileDialog.getOpenFileName()
        self.path = filename[0]
        self.image  =cv2.imread(self.path, 0)
        self.rgb_image = cv2.imread(self.path, 1)
        self.shape = self.image.shape

        R = self.rgb_image[:, :, 0]
        G = self.rgb_image[:, :, 1]
        B = self.rgb_image[:, :, 2]

        # By using LUMA-REC.709
        grayValue = 0.2125 * R + 0.7154 * G + 0.0721 * B
        gray_img = grayValue.astype(np.uint8)

        gray_img = QImage(gray_img, gray_img.shape[1], gray_img.shape[0], QImage.Format_Grayscale8)
        self.displayImage(gray_img,Label)

#*********************************** gray scale transformation from scratch ***************************************
    def displayImage(self,image,Label):
        pixmap = QPixmap(image)
        Label.setPixmap(pixmap)
        Label.setScaledContents(True)

    def Whichbtn(self,Button):
        if Button ==  self.ui.pushButton_5:
            self.LoadImage(self.ui.input_6)
            self.imageA=self.image
        elif Button ==  self.ui.pushButton_4:
            self.LoadImage(self.ui.input_8)
            self.imageB=self.image

#########################################Hybrid###################################################################################3
    def Hybrid(self):
        width = int(self.imageA.shape[1] )
        height =  int(self.imageA.shape[0] )
        dim = (width, height)
        self.imageRA = cv2.resize(self.imageA, dim, interpolation = cv2.INTER_AREA)
        self.imageRB = cv2.resize(self.imageB, dim, interpolation = cv2.INTER_AREA)
        self.gaussian_kernel_size=3
        self.gaussian_filter = self.gaussian_kernel(self.gaussian_kernel_size)
        self.imageLA,self.gradient_direction=self.filter_Image(self.imageRA,self.gaussian_filter,self.gaussian_kernel_size)
        laplacian_filter =np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.imageHB,gradient_direction=self.filter_Image(self.imageRB,laplacian_filter,3)

        self.hybreid_Image=self.imageLA + self.imageHB
        self.hybreid_Image = np.array(self.hybreid_Image).astype(np.int8)
        self.hybreid_Image = QImage(self.hybreid_Image.data, self.hybreid_Image.shape[1], self.hybreid_Image.shape[0],QImage.Format_Grayscale8)
        self.displayImage(self.hybreid_Image,self.ui.input_7)    
        print('Hybrid Image')
##########################################Draw Histograms and cumulative curves ########################################
    def drawHistogram(self):
        histogram, bin_edges = np.histogram(self.image, bins=255, range=(0, 255.0))
        self.ui.MplWidget.canvas.axes.clear()
        self.ui.MplWidget.canvas.axes.plot(bin_edges[0:-1], histogram)
        self.ui.MplWidget.canvas.axes.set_title('Histogram')
        self.ui.MplWidget.canvas.draw()

    def unique(self,input_list):
        unique_list = []
        for x in input_list:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list

    def get_values_frequencies(self, channel):
        Hist = channel.flatten().tolist()
        uniqe_values = self.unique(Hist)
        uniqe_values = sorted(uniqe_values)
        frequencies = [Hist.count(x) for x in uniqe_values]
        return frequencies

    def calculate_cumulative_distribution(self,channel):
        frequencies = self.get_values_frequencies(channel)
        accumulator = []
        accumulator.append(float(frequencies[0]))
        for index in range(1, len(frequencies)):
            accumulator.append(accumulator[index - 1] + float(frequencies[index]))
        return accumulator

    def draw_channel_histogram_or_distribution_curve(self,channel, color,Button ):
        frequencies = self.get_values_frequencies(channel)
        cumulative_distribution = self.calculate_cumulative_distribution(channel)
        if Button ==  self.ui.pushButton_11:
            self.ui.MplWidget_3.canvas.axes.plot([i for i in range(len(frequencies))],frequencies, color=color)
            self.ui.MplWidget_3.canvas.axes.set_title('RGBHistograms')

        elif Button ==  self.ui.pushButton_12:
            self.ui.MplWidget_2.canvas.axes.plot([i for i in range(len(frequencies))],cumulative_distribution,color=color)
            self.ui.MplWidget_2.canvas.axes.set_title('cumulative_curves')


    def draw_colored_image_histogram_or_distribution_curve(self,Button):
        self.ui.MplWidget.canvas.axes.clear()
        for i, col in enumerate(['r', 'g', 'b']):
            self.draw_channel_histogram_or_distribution_curve(self.rgb_image[:, :, i], col,Button )
        if Button ==  self.ui.pushButton_11:
            self.ui.MplWidget_3.canvas.draw()
        elif Button == self.ui.pushButton_12:
            self.ui.MplWidget_2.canvas.draw()
        else:
            pass

    #############################################Filter##############################################################################3
    def filter_Image(self,image,filter,Kernal_size):    
            [rows, columns] = np.shape(image)  # we need to know the shape of the input grayscale image
            output_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)
            gradient_direction = np.zeros(shape=(rows, columns))
            for i in range(rows - (Kernal_size-1)):
                for j in range(columns - (Kernal_size-1)):
                    gx = np.sum(np.multiply(filter, image[i:i + Kernal_size, j:j + Kernal_size]))  # x direction
                    gy = np.sum(np.multiply(filter.T, image[i:i + Kernal_size, j:j + Kernal_size]))  # y direction
                    gradient_direction[i+1,j+1] = np.arctan2(gy,gx)
                    gradient_direction[i+1,j+1] = np.rad2deg(gradient_direction[i+1,j+1])
                    gradient_direction[i+1,j+1] += 180
                    output_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

            return output_filtered_image ,gradient_direction   

    def dnorm(self,x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
    
    def gaussian_kernel(self,size, sigma=1):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = self.dnorm(kernel_1D[i], 0, sigma)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
        kernel_2D *= 1.0 / kernel_2D.max()
        return kernel_2D

    def non_max_suppression(self,gradient_magnitude, gradient_direction):
        image_row, image_col = gradient_magnitude.shape
        output = np.zeros(gradient_magnitude.shape)
        PI = 180
        for row in range(1, image_row - 1):
            for col in range(1, image_col - 1):
                direction = gradient_direction[row, col]
    
                if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                    before_pixel = gradient_magnitude[row, col - 1]
                    after_pixel = gradient_magnitude[row, col + 1]
    
                elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                    before_pixel = gradient_magnitude[row + 1, col - 1]
                    after_pixel = gradient_magnitude[row - 1, col + 1]
    
                elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                    before_pixel = gradient_magnitude[row - 1, col]
                    after_pixel = gradient_magnitude[row + 1, col]
    
                else:
                    before_pixel = gradient_magnitude[row - 1, col - 1]
                    after_pixel = gradient_magnitude[row + 1, col + 1]
    
                if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                    output[row, col] = gradient_magnitude[row, col]
        return output 


    def threshold(self,image, low, high):
            strong = 1
            strong_row, strong_col = np.where(image > high)
            low_row, low_col = np.where(image < low)
        
            image[strong_row, strong_col] = strong
            image[low_row, low_col] = 0
            return image
 
    def hysteresis(self,image):
        image_row, image_col = image.shape

        top_to_bottom = image.copy()

        for row in range(1, image_row):
            for col in range(1, image_col):
                if top_to_bottom[row, col] > 0 and top_to_bottom[row, col] <1 :
                    if top_to_bottom[row, col + 1] == 1 or top_to_bottom[row, col - 1] == 1 or top_to_bottom[row - 1, col] == 1 or top_to_bottom[
                        row + 1, col] == 1 or top_to_bottom[
                        row - 1, col - 1] == 1 or top_to_bottom[row + 1, col - 1] == 1 or top_to_bottom[row - 1, col + 1] == 1 or top_to_bottom[
                        row + 1, col + 1] == 1:
                        top_to_bottom[row, col] = 1
                    else:
                        top_to_bottom[row, col] = 0

        bottom_to_top = image.copy()

        for row in range(image_row - 1, 0, -1):
            for col in range(image_col - 1, 0, -1):
                if bottom_to_top[row, col] > 0 and bottom_to_top[row, col]< 1:
                    if bottom_to_top[row, col + 1] == 1 or bottom_to_top[row, col - 1] == 1 or bottom_to_top[row - 1, col] == 1 or bottom_to_top[
                        row + 1, col] == 1 or bottom_to_top[
                        row - 1, col - 1] == 1 or bottom_to_top[row + 1, col - 1] == 1 or bottom_to_top[row - 1, col + 1] == 1 or bottom_to_top[
                        row + 1, col + 1] == 1:
                        bottom_to_top[row, col] = 1
                    else:
                        bottom_to_top[row, col] = 0

        right_to_left = image.copy()

        for row in range(1, image_row):
            for col in range(image_col - 1, 0, -1):
                if right_to_left[row, col] > 0 and right_to_left[row, col]< 1:
                    if right_to_left[row, col + 1] == 1 or right_to_left[row, col - 1] == 1 or right_to_left[row - 1, col] == 1 or right_to_left[
                        row + 1, col] == 1 or right_to_left[
                        row - 1, col - 1] == 1 or right_to_left[row + 1, col - 1] == 1 or right_to_left[row - 1, col + 1] == 1 or right_to_left[
                        row + 1, col + 1] == 1:
                        right_to_left[row, col] = 1
                    else:
                        right_to_left[row, col] = 0

        left_to_right = image.copy()

        for row in range(image_row - 1, 0, -1):
            for col in range(1, image_col):
                if left_to_right[row, col] > 0 and left_to_right[row, col]< 1:
                    if left_to_right[row, col + 1] == 1 or left_to_right[row, col - 1] == 1 or left_to_right[row - 1, col] == 1 or left_to_right[
                        row + 1, col] == 1 or left_to_right[
                        row - 1, col - 1] == 1 or left_to_right[row + 1, col - 1] == 1 or left_to_right[row - 1, col + 1] == 1 or left_to_right[
                        row + 1, col + 1] == 1:
                        left_to_right[row, col] = 1
                    else:
                        left_to_right[row, col] = 0

        final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

        final_image[final_image > 1] = 1

        return final_image
    def Filter(self,image,filter,size):
            self.Output_filtered_image,self.gradient_direction=self.filter_Image(image,filter,size)
            self.Output_filtered_image = np.array(self.Output_filtered_image).astype(np.int8)
            self.Output_filtered_image = QImage(self.Output_filtered_image.data, self.Output_filtered_image.shape[1], self.Output_filtered_image.shape[0],QImage.Format_Grayscale8)
            self.displayImage(self.Output_filtered_image,self.ui.output)    
    
    
    def ApplyFilter(self):
        combo=self.ui.comboBox.currentText()
        if combo=="Sobel Filter":
            Sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            self.Filter(self.image,Sobel_filter,3)
        elif combo=="Roberts Filter":
            roberts_filter = np.array([[ 1, 0], [ 0, -1]])
            self.Filter(self.image,roberts_filter,2)


        elif combo=="Prewitt Filter":
            prewitt_filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            self.Filter(self.image,prewitt_filter,3)


        elif combo=="Canny Filter":              
            self.gaussian_kernel_size=3
            self.gaussian_filter = self.gaussian_kernel(self.gaussian_kernel_size)
            self.Output_filtered_image,self.gradient_direction=self.filter_Image(self.image,self.gaussian_filter,self.gaussian_kernel_size)
            prewitt_filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            self.Output_filtered_image,self.gradient_direction=self.filter_Image(self.Output_filtered_image,prewitt_filter,3)

            self.Output_filtered_image=self.non_max_suppression(self.Output_filtered_image,self.gradient_direction)
            self.Output_filtered_image=self.Output_filtered_image/self.Output_filtered_image.max()

            self.Output_filtered_image=self.threshold(self.Output_filtered_image,self.Lth,self.Hth)
            self.Output_filtered_image=self.hysteresis( self.Output_filtered_image)*255
            self.Output_filtered_image = np.array(self.Output_filtered_image).astype(np.int8)
            self.Output_filtered_image = QImage(self.Output_filtered_image.data, self.Output_filtered_image.shape[1], self.Output_filtered_image.shape[0],QImage.Format_Grayscale8)
            self.displayImage(self.Output_filtered_image,self.ui.output)
    
     ######################### the End Filter #######################################################################
    def lowThreshold(self,value):
        self.Lth=value/100
        print(self.Lth)

    def HighThreshold(self,value):
        self.Hth=value/10
        print(self.Hth)
    def Alpha(self,value):
        self.alpha=value/10
        print(self.alpha)
#8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
    def draw_cicle(self, diameter):
        assert len(self.shape) == 2
        arr = np.zeros(self.shape, dtype=np.bool)
        center = np.array(arr.shape) / 2.0
        for iy in range(self.shape[0]):
            for ix in range(self.shape[1]):
                arr[iy, ix] = (iy - center[0]) ** 2 + (ix - center[1]) ** 2 < diameter ** 2
        return (arr)

    def image_model(self):
        fft = np.fft.fft2(self.image)
        self.fshift = np.fft.fftshift(fft)
        self.abs_fft_img = 20 * np.log(np.abs(self.fshift))

        self.circleIN = self.draw_cicle(self.ui.horizontalSlider_3.value()).astype(int)  # low pass
        self.circleOUT = ~self.circleIN + 2
#***********************************************************************************************************************
    def HPFfunction(self):
        if self.flage == 1:
            self.image_model()
            HPF_img = self.abs_fft_img * self.circleOUT

            HPFishift = np.fft.ifftshift(self.fshift * self.circleOUT)
            HPFimback = np.uint8(np.real(np.fft.ifft2(HPFishift)))

            HPF_selection = self.ui.comboBox_3.currentText()
            if HPF_selection == 'HPF_img':
                self.pw5 = pg.ImageView(self.ui.input_10)
                self.pw5.show()
                self.pw5.setImage(HPFimback.T)
               
            elif HPF_selection == 'fft_img':
                self.pw6 = pg.ImageView(self.ui.input_10)
                self.pw6.show()
                self.pw6.setImage(self.abs_fft_img.T)
            elif HPF_selection == 'HPF_fft_img':
                self.pw7 = pg.ImageView(self.ui.input_10)
                self.pw7.show()
                self.pw7.setImage(HPF_img.T)
            else:
                pass
        else:
            pass

    # ***********************************************************************************************************************
    def LPFfunction(self):
        if self.flage == 1:
            self.image_model()
            LPF_img = self.abs_fft_img * self.circleIN

            LPFishift = np.fft.ifftshift(self.fshift * self.circleIN)
            LPFimback = np.uint8(np.real(np.fft.ifft2(LPFishift)))

            LPF_selection = self.ui.comboBox_2.currentText()
            if LPF_selection == 'LPF_img':
                self.pw1 = pg.ImageView(self.ui.input_11)
                self.pw1.show()
                self.pw1.setImage(LPFimback.T)
            elif LPF_selection == 'fft_img':
                self.pw2 = pg.ImageView(self.ui.input_11)
                self.pw2.show()
                self.pw2.setImage(self.abs_fft_img.T)

            elif LPF_selection == 'LPF_fft_img':
                self.pw3 = pg.ImageView(self.ui.input_11)
                self.pw3.show()
                self.pw3.setImage(LPF_img.T)
            else:
                pass
        else:
            pass

    def slider_change(self):
        self.LPFfunction()
        self.HPFfunction()

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()

