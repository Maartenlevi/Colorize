import numpy as np
import cv2
import matplotlib.pyplot as plt

def lab_to_rgb(L_tensor, ab_tensor):
    L = L_tensor.squeeze().cpu().numpy() * 100
    ab = ab_tensor.squeeze().cpu().numpy() * 128
    lab = np.zeros((256, 256, 3), dtype=np.float32)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab.transpose((1, 2, 0))
    rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return rgb

def show_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
