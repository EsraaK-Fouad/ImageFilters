#-------------------------------------------------------------------------------------------------------
'''                                 Noise Class                                                        '''
#----------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
#--------------------------------------------------------------------------
class Noise:
    def __init__(self,path ):
        self.image = path
    def salt_and_pepper(self,noise_level):
        noisy_image = np.zeros(self.image.shape,np.uint8)
        self.thres = 1 - noise_level 
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                rdn = random.random()
                if rdn < noise_level:
                    noisy_image[i][j] = 0
                elif rdn > self.thres:
                    noisy_image[i][j] = 255
                else:
                    noisy_image[i][j] = self.image[i][j]
        cv2.imwrite('sp_noise.jpg', noisy_image)
        return noisy_image
     
    def uniform_noise(self,a,b,plot=False):
        x=self.pseudo_uniform(a,b,self.image.shape[0]*self.image.shape[1],1)
        x=x.reshape((self.image.shape[0],self.image.shape[1]))
        uniform=0.2*x+self.image
        cv2.imwrite('uniform.jpg', uniform)
        return uniform
    def pseudo_uniform_good(self,mult=16807,mod=(2**31)-1,seed=123456789,size=1):
        U=np.zeros(size)
        X=(seed*mult+1)%mod
        U[0]=X/mod
        for i in range(1,size):
            X=(X*mult+1)%mod
            U[i]=X/mod
        return U
    def pseudo_uniform(self,low=0,high=1,size=1000,seed=123456789):
        return low+(high-low)*self.pseudo_uniform_good(seed=seed,size=size)  
    def pseudo_norm(self,mu=0,sigma=1,size=1):
        t=time.perf_counter()
        seed1=int(10**9*float(str(t-int(t))[0:]))
        U1=self.pseudo_uniform(size=size,seed=seed1)
        t=time.perf_counter()
        seed2=int(10**9*float(str(t-int(t))[0:]))
        U2=self.pseudo_uniform(size=size,seed=seed2)
        z0=np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)
        z0=z0*sigma+mu
        return z0
    def gaussian_noise(self,sigma,plot=False):
        x=self.pseudo_norm(0,sigma,self.image.shape[0]*self.image.shape[1])
        if plot==True:
            sns.distplot(x)
            plt.show()
        x=x.reshape(self.image.shape[0],self.image.shape[1])
        guassian=x+self.image
        cv2.imwrite('guassian_noise.jpg', guassian)
        return guassian


