import imageio.v2 as im
import numpy as np
import matplotlib.pyplot as plt

img = im.imread('blurry_moon.tif')
M,N = img.shape
# print(M,N)
F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)

fig = plt.figure(figsize=(20,20))

fig.add_subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Original Image")

def butterworth():                            
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 50
    k = 0.4
    n = 1
    for u in range(M):
        for v in range(N):
            Duv = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = 1 / (1 + (D0/Duv)**(2*n))

    G = Fshift * H
    g = np.abs(np.fft.ifft2(G))

    Out = img + k*g
    
    fig.add_subplot(2, 2, 3)
    plt.imshow(Out, cmap='gray')
    plt.axis('off')
    plt.title("Butterworth High Pass")
    
    print("\nFor Butterworth High Pass Filtering: ")
    print("n = ",n)
    print("k = ",k)
    print("Do: ",D0)
    print("")

butterworth()

def gaussian():                               
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 25
    k = 0.4
    for u in range(M):
        for v in range(N):
            Duv = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = 1 - np.exp(-(Duv**2)/(2*(D0**2)))

    G = Fshift * H
    g = np.abs(np.fft.ifft2(G))

    Out = img + k*g
    
    fig.add_subplot(2, 2, 4)
    plt.imshow(Out, cmap='gray')
    plt.axis('off')
    plt.title("Gaussian High Pass")
    
    plt.savefig('Ayush_ce2.tif')
    
    print("\nFor gaussian High Pass Filtering: ")
    print("k = ",k)
    print("Do: ",D0)
    print("")


gaussian()

    
