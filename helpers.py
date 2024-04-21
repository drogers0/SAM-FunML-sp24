import numpy as np
import matplotlib.pyplot as plt
from config import *

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    ax.scatter(pos_points[:, 0], pos_points[:, 1], color=GREEN_COLOR, marker='*', s=marker_size, edgecolor='white',
            linewidth=LINEWIDTH)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color=RED_COLOR, marker='*', s=marker_size, edgecolor='white',
            linewidth=LINEWIDTH)

def closetn(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def downsample(image):
    width, height = image.shape[-2:] #width, height of the original image
    downsample_image = cv2.pyrDown(image) # cv2 downsample function
    d_width, d_height = downsample_image.shape[-2:] # width, height of the downsampled image

    # Zero-pad downsampled image
    leftright = width - d_width
    updown = height - d_height
    padded_image = torchvision.transforms.Pad((leftright/2, updown/2, leftright/2, updown/2)) # left, top, right, bottom
    result = padded_image(downsample_image)
    result = result.resize((width, height)) #resize to the original image
    return result