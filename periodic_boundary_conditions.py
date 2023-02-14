import numpy as np

def get_image_particles(positions):
    n_images   = 2
    natm, ndim = positions.shape
    images     = np.zeros(shape=(n_images, natm, ndim))
    images[:,:,:] = positions
    images[1,:,0] -= 1
    return images


print()