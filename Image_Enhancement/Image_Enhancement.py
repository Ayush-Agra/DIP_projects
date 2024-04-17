
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio

def func(img,kernel):
    row,col=img.shape
    nw=np.zeros((row,col))
    for i in range(1,row-1):
        for j in range(1,col-1):
            arr = img[i-1:i+2,j-1:j+2]
            nw[i][j] = int(np.sum(np.multiply(arr,kernel)))
    return nw
def gaussian(img,sigma):
    kernel=np.zeros((3,3))
    variance = pow(sigma,2)
    sz=3//2;
    for i in range(-sz,sz+1):
        for j in range(-sz,sz+1):
            kernel[sz+i][sz+j]=np.exp(-(i**2 + j**2) / (2 * variance**2))
    kernel=kernel/(np.sum(kernel))
    return func(img,kernel)
def laplace(img):
    laplace = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    laplac_img = func(img2, laplace)
    laplac_img = np.multiply(k,laplac_img)
    return laplac_img

#input
path="F:\skeleton.tif"
img=iio.imread(path)
sigma=1;
k=1;

#processing the image
img2=gaussian(img,1)
laplac_img=laplace(img2)
sharp_img = img - laplac_img
sharp_img += np.abs(sharp_img.min())
sharp_img *= (255/sharp_img.max())
sharp_img = sharp_img.astype(np.uint8)

c = 256
gamma = 0.5
power_transformed_img = c * np.power((sharp_img/255),gamma)
power_transformed_img = power_transformed_img.astype(np.uint8)

#output
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.gray()
plt.imshow(img)
plt.axis('off')
plt.title("Input")

fig.add_subplot(1, 2, 2)
plt.gray()
plt.imshow(power_transformed_img)
plt.axis('off')
plt.title("Output")
plt.show()

print("Order Of Gaussian filter = 0")
print("Sigma = ",sigma)
print("k = ",k)
print("Gamma = ",gamma)