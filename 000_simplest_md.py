import numpy as np
from itertools import combinations
print()

def gradient_return(r_t, epsilon, sigma_t):
    z = sigma_t/r_t
    u = z*z*z
    return 24 * epsilon * u * (1 - 2 * u) / r_t

def potential_return(r_t,epsilon,sigma_t):
    z = sigma_t/r_t
    u = z*z*z
    return -4*epsilon*u*(1-u)

def force(positions):
    '''
    Given the xyz coordinates of n atoms,
    Calculate the distances between every pair of atoms
    For each atom, calculate the forces that other atoms exert on it on the XYZ directions
    '''
    #https://stackoverflow.com/questions/25965329/difference-between-every-pair-of-columns-of-two-numpy-arrays-how-to-do-it-more
    n_atoms, n_dimensions            = positions.shape
    each_atom_coordinates            = positions.reshape(n_atoms, 1, n_dimensions)#from stack overflow
    interatom_distances_per_dimension_per_atom = each_atom_coordinates - positions

    squared_cartesian_distances  = np.sum(interatom_distances_per_dimension_per_atom**2,axis=2)
    gradients_between_atoms      = gradient_return(squared_cartesian_distances, 1, 1)
    interatom_gradients_per_atom = gradients_between_atoms.reshape(n_atoms, n_atoms, 1)

    force_per_dimension_per_atom = interatom_gradients_per_atom * interatom_distances_per_dimension_per_atom
    force_per_dimension_per_atom[np.isnan(force_per_dimension_per_atom)] = 0
    new_forces = np.sum(force_per_dimension_per_atom,axis=0)
    return new_forces

ndim        = 3
natm        = 3
nsteps      = 10000
currenttime = 0
timestep    = 0.001

masses = np.ones(natm)
positions = np.zeros(shape=(natm,ndim))
                 #x coordinate,y coordinate,z coordinate
positions[0,:] = [0.5391356726,0.1106588251,-0.4635601962] #atom1
positions[1,:] = [-0.5185079933,0.4850176090,0.0537084789] #atom2
positions[2,:] = [0.0793723207,-0.4956764341,0.5098517173] #atom3

forces     =  force(positions)
velocities = np.zeros(shape=(natm,ndim))
potentials = np.zeros(shape=(natm,natm))


for step in range(nsteps):
    velocities += forces * timestep / 2 / masses #take a halfstep forward and calculate new positions
    positions  += velocities * timestep
    forces     =  force(positions)               
    velocities += forces * timestep / 2 / masses #take halfstep forward using new force information
    print()
print()
'''
positions[0,:] - positions
positions.reshape(3,1,3) the subtraction will always be between the innermost array

for each column
'''

