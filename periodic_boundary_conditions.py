import numpy as np

def get_image_particles(positions):
    n_images   = 2
    natm, ndim = positions.shape
    images     = np.zeros(n_images, natm, ndim)
    images[0]  = positions.copy()
    images[1]  = np.zeros(natm,ndim)
    images[1]  = positions[:,0] - 1
    return images
    