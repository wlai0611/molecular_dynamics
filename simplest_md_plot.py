import matplotlib.pyplot as plt
import numpy as np
import re

file = open('gibbs_properties.dat',mode='r')
contents = file.read()
lines = contents.split('\n')
kinetic_energies  = lines[0:-1:3]
potential_energies=lines[1::3]
total_energies    = lines[2::3]

energy_arr = np.zeros(shape=(len(kinetic_energies),3),dtype=float)

for row_num, energies in enumerate(zip(kinetic_energies,potential_energies,total_energies)):
    for column_num, energy in enumerate(energies):
        energy_value = float(re.split('\s+',energy)[-1])
        energy_arr[row_num, column_num] = energy_value

fig, ax = plt.subplots()
ax.plot(range(len(kinetic_energies)), energy_arr[:,0], label='kinetic')
ax.plot(range(len(kinetic_energies)), energy_arr[:,1], label='potential')
ax.plot(range(len(kinetic_energies)), energy_arr[:,2], label='total')
ax.set(ylabel='Energy', xlabel='Timesteps')
ax.legend()
plt.show()
print()