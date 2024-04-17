
import numpy as np
import imageio.v2 as im
from matplotlib import pyplot as plt

Ps = 0.1
Pp = 0.1
mean = 0 
variance = 900
size = int(input("Enter the kernel Size. 3,5 or 7 - "))
d = 6

image_path = "ckt-board-orig.tif"
image = im.imread(image_path)

def uniform_limits(mean, variance):
    std_dev = (variance * 12) ** 0.5

    lower_limit = mean - std_dev / 2
    upper_limit = mean + std_dev / 2

    return lower_limit, upper_limit

def uniform(image,mean,variance):
    a,b = uniform_limits(mean, variance)
    noise = np.random.uniform(a,b, image.shape)
    image_a = np.clip(image + noise, 0, 255).astype(np.uint8)
    return image_a

image_a = uniform(image,mean,variance)



def add_salt_and_pepper_noise(image, Ps, Pp):
    
    noisy_image = np.copy(image)

    salt_mask = np.random.rand(*image.shape) < Ps
    pepper_mask = np.random.rand(*image.shape) < Pp

    # Add salt noise (white pixels)
    noisy_image[salt_mask] = 255

    # Add pepper noise (black pixels)
    noisy_image[pepper_mask] = 0

    return noisy_image

image_b = add_salt_and_pepper_noise(image_a,Ps, Pp)


def mean_filter(image, size):
    height,width = image.shape

    filtered = np.zeros_like(image)
    pad = size // 2

    for i in range(pad,height - pad):
        for j in range(pad, width - pad):
            neighour = image[i-pad:i+pad+1,j-pad:j+pad+1]
            mean = np.mean(neighour)
            filtered[i,j] = mean

    return filtered

image_c = mean_filter(image_b,size)

def alpha_trimmed_filter(image, size, d):
    height,width = image.shape

    filtered = np.zeros_like(image)
    pad = size // 2

    for i in range(pad,height - pad):
        for j in range(pad, width - pad):
            neighour = image[i-pad:i+pad+1,j-pad:j+pad+1]
            t1,t2 = neighour.shape
            neighour = neighour.reshape((t1*t2))
            neighour = np.sort(neighour)
            neighour = neighour[d//2:-d//2]
            mean = np.mean(neighour)
            filtered[i,j] = mean

    return filtered

image_d = alpha_trimmed_filter(image_b,size, d)

plt.subplot(2,2,1)
plt.imshow(image_a)
plt.subplot(2,2,2)
plt.imshow(image_b)
plt.subplot(2,2,3)
plt.imshow(image_c)
plt.subplot(2,2,4)
plt.imshow(image_d)
plt.show()

print(d)

print("Best Kernel Size is 3*3 by visualization.")