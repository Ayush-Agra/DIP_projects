import imageio
import numpy as np
import matplotlib.pyplot as plt

def power_law_transform(image, gamma):
    return (np.power(image / 255.0, gamma) * 255).astype(np.uint8)

def rgb_to_hsi(rgb):
    rgb = rgb / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    numi = ((r - g) + (r - b)) / 2
    denom = np.sqrt((r - g)**2 + (r - b) * (g - b))
    H = np.arccos(np.clip(numi / (denom + 1e-6), -1, 1)) * 180 / np.pi
    H[b > g] = 360 - H[b > g]
    H /= 360.0
    S = 1 - (3 * np.minimum(r, np.minimum(g, b)) / (r + g + b + 1e-6))
    I = (r + g + b) / 3
    return np.stack((H, S, I), axis=-1)


def hsi_to_rgb(hsi):
    h, s, i = hsi[..., 0] * 360, hsi[..., 1], hsi[..., 2]
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)

    # Sector 1: 0 <= H < 120
    sector1 = np.logical_and(0 <= h, h < 120)
    b[sector1] = i[sector1] * (1 - s[sector1])
    r[sector1] = i[sector1] * (1 + s[sector1] * np.cos(h[sector1] * np.pi / 180) / np.cos((60 - h[sector1]) * np.pi / 180))
    g[sector1] = 3 * i[sector1] - (r[sector1] + b[sector1])

    # Sector 2: 120 <= H < 240
    sector2 = np.logical_and(120 <= h, h < 240)
    h_shifted = h[sector2] - 120
    r[sector2] = i[sector2] * (1 - s[sector2])
    g[sector2] = i[sector2] * (1 + s[sector2] * np.cos(h_shifted * np.pi / 180) / np.cos((60 - h_shifted) * np.pi / 180))
    b[sector2] = 3 * i[sector2] - (r[sector2] + g[sector2])

    # Sector 3: 240 <= H < 360
    sector3 = np.logical_and(240 <= h, h < 360)
    h_shifted = h[sector3] - 240
    g[sector3] = i[sector3] * (1 - s[sector3])
    b[sector3] = i[sector3] * (1 + s[sector3] * np.cos(h_shifted * np.pi / 180) / np.cos((60 - h_shifted) * np.pi / 180))
    r[sector3] = 3 * i[sector3] - (g[sector3] + b[sector3])

    return np.stack((r, g, b), axis=-1)

# Read the image
f = imageio.imread('colors.tif')
rgb_image = np.copy(f)

# Set the gamma value
gamma = 1.6

# Apply power-law transformation to the RGB image
new_image = power_law_transform(rgb_image, gamma)

# Convert RGB image to HSI
hsi_image = rgb_to_hsi(rgb_image)

# Apply power-law transformation to the intensity component of the HSI image
I_transform = power_law_transform(hsi_image[..., 2]*255, gamma)
hsi_image[..., 2] = I_transform / 255

# Convert the transformed HSI image back to RGB
img = hsi_to_rgb(hsi_image)

# Print the value of gamma
print("Value of gamma is:", gamma)

# Plot the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(rgb_image)
plt.title("Original RGB Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(new_image)
plt.title("Power Law Transform (RGB)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img)
plt.title("Power Law Transform (HSI to RGB)")
plt.axis("off")

plt.tight_layout()
plt.show()