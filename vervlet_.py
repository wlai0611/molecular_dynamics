import numpy as np
import matplotlib.pyplot as plt
import time
import functions
import periodic_boundary_conditions

def interatomic_per_dimension_distances(coordinates):
    rows,columns = coordinates.shape
    stacked_rows = coordinates.reshape(rows, 1, columns)
    return stacked_rows - coordinates

def gradient_return(r_t, epsilon, sigma_t):
    z = sigma_t/r_t
    u = z*z*z
    return 24 * epsilon * u * (1 - 2 * u) / r_t

def potential_return(r_t,epsilon,sigma_t):
        z = sigma_t/r_t
        u = z*z*z
        return -4*epsilon*u*(1-u)

def distances_to_forces(gradients_between_atoms, interatom_distances_per_dimension_per_atom):
    rows, rows      = gradients_between_atoms.shape
    stacked_columns = gradients_between_atoms.reshape(rows, rows, 1)
    force_per_dimension_per_atom = stacked_columns * interatom_distances_per_dimension_per_atom
    
    return np.sum(force_per_dimension_per_atom,axis=0)

def get_interatomic_forces(coordinates):
    per_dimension_distances = interatomic_per_dimension_distances(coordinates = coordinates)
    euclidean_distances     = np.sum(per_dimension_distances**2, axis=2)
    euclidean_distances[euclidean_distances==0] = np.nan
    gradients_between_atoms = gradient_return(euclidean_distances, 1, 1)
    gradients_between_atoms[np.isnan(gradients_between_atoms)] = 0
    forces = distances_to_forces(gradients_between_atoms, per_dimension_distances)
    return euclidean_distances, forces

def compute_potential_energy(euclidean_distances):
        potentials_between_atoms     = potential_return(euclidean_distances, 1, 1)
        return np.sum(np.triu(potentials_between_atoms,1))

def compute_kinetic_energy(m, v):
        return 0.5*np.sum(np.dot(m, v**2))

def simulate(images, m = 1, nsteps = 10000, dt = 0.001,):
    '''
    User Inputs Atom Coordinates -> 
    Calculate Interatomic Forces -> Update Atom Velocities -> Update Atom Coordinates -> 
    Update Interatomic Forces    -> Update Atom Velocities -> Update Atom Coordinates ->
    Update Interatomic Forces    -> Update Atom Velocities -> Update Atom Coordinates -> ...
    Function Outputs Log At Each Timestep of 1) System-Level Energies and 2) Atom-Level Coordinates
    '''
    nimages,natoms,ndimensions = images.shape
    v  = np.zeros(shape=(natoms,ndimensions))
    m  = np.ones(natoms) * m
    trajectory = np.zeros(shape=(nsteps,nimages*natoms,ndimensions))
    potential_energies = np.zeros(shape=nsteps)
    kinetic_energies   = np.zeros(shape=nsteps)

    distances, F  = get_interatomic_forces(coordinates = images[0])
    for step in range(nsteps):
        a  =  F/m
        dv =  a * dt/2
        v  += dv

        dx =  v*dt
        images += dx

        distances, F  =  get_interatomic_forces(coordinates = images[0])
        a  =  F/m
        dv =  a * dt/2
        v  += dv

        trajectory[step] = images.reshape(nimages*natoms, ndimensions)
        potential_energies[step] = compute_potential_energy(distances)
        kinetic_energies[step]   = compute_kinetic_energy(m, v)
    results = {'trajectory':trajectory, 'potential_energies':potential_energies, 'kinetic_energies':kinetic_energies}
    return results

def initialize_positions():
    positions = np.array([
        [0.5391356726,0.1106588251,-0.4635601962], #atom1
        [-0.5185079933,0.4850176090,0.0537084789], #atom2
        [0.0793723207,-0.4956764341,0.5098517173], #atom3
    ])
    return positions

def run():
    positions = initialize_positions()
    start  = time.time()
    images = periodic_boundary_conditions.get_image_particles(positions)
    results = simulate(images, nsteps=10000)
    print(f"Duration: {time.time()-start}")

    fig, ax = plt.subplots()
    ax.plot(range(10000), results['kinetic_energies'], label='kinetic')
    ax.plot(range(10000), results['potential_energies'], label='potential')
    ax.plot(range(10000), results['kinetic_energies'] + results['potential_energies'], label='total')
    ax.set(ylabel='Energy', xlabel='Timesteps')
    ax.legend()
    plt.savefig('energy.jpg')
    functions.write_trajectory(results['trajectory'])
run()