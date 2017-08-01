# Monte_Carlo simulation w/ Lennard Jones Potential

import numpy as np

# Parameters
reduced_density = 0.9
reduced_temperature = 0.9
num_particles = 216

beta = 1/reduced_temperature
box_length = np.cbrt(num_particles / reduced_density)
cutoff = 3.0
cutoff2 = np.power(cutoff,2)
max_displacement = 0.1

spacing = int(np.cbrt(num_particles) + 1)
x_vector = np.linspace(0.0, box_length, spacing)
y_vector = np.linspace(0.0, box_length, spacing)
z_vector = np.linspace(0.0, box_length, spacing)
grid  = np.meshgrid(x_vector, y_vector, z_vector)
stack = np.vstack(grid)
coordinate = stack.reshape(3, -1).T
excess = len(coordinate) - num_particles
coordinate = coordinate[:-excess]
coordinate *= 0.95

#################### Energy #############
def minimum_squared_distance(r_i,r_j,box_length):
    rij = r_i - r_j
    rij = rij - box_length*np.round(rij/box_length)
#periodical boundary
    rij2 = np.dot(rij,rij)
    return rij2

def lennard_jones_potential(rij2):
    sig_by_r6 = np.power(1/rij2,3)
    sig_by_r12 = np.power(sig_by_r6,2)
    return 4.0*(sig_by_r12-sig_by_r6)


def total_potential_energy(coordinate, box_length):
    e_total = 0.0
    for i_particle in range(0, num_particles):
        for j_particle in range(0, i_particle):
            r_i = coordinate[i_particle]
            r_j = coordinate[j_particle]
            rij2 = minimum_squared_distance(r_i,r_j,box_length)
            if rij2 < cutoff2:
                e_pair =  lennard_jones_potential(rij2)
                e_total += e_pair
    return e_total


def get_molecule_energy(coordinate, i_particle):
    i_position = coordinate[i_particle]
    e_i = 0.0
    for j_particle in range(0, num_particles):
        if i_particle != j_particle:
            j_position = coordinate[j_particle]
            rij2 = minimum_squared_distance(i_position, j_position, box_length)
            
            if rij2 < cutoff2:
                e_pair =  lennard_jones_potential(rij2)
                e_i += e_pair
                
    return e_i

def tail_correction(box_length):
    volume = np.power(box_length,3)
    sig_by_cutoff3 = np.power(1.0/cutoff,3)
    sig_by_cutoff9 = np.power(sig_by_cutoff3,3)
    e_correction = sig_by_cutoff9 - 3.0*sig_by_cutoff3
    e_correction *= 8.0/9.0 * np.pi *num_particles/volume*num_particles
    return e_correction



