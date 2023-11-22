import matplotlib.pyplot as plt
import numpy as np

# Constants
Al_yield = 276e6 # Pa
Al_ultimate = 310e6 # Pa
Al_density = 2700 # kg/m^3
Al_E = 70e9 # Pa

# Inputs
perimeter = 0.1*4

thickness= np.arange(0.0005, 0.01, 0.0001)

# Calculations
area = perimeter*thickness
stress = Al_yield
F = stress*area / 4

# Buckling load for the thickness 
K = 1 # End conditions
L = 0.1*3 # Length of the CubeSat
E = Al_E

# Calculate the moment of inertia with the thickness for a square cross section with sides of 0.1 m
inner = 0.1 - 2*thickness
I = 0.1**4/12 - inner**4/12

# Calculate the critical buckling load
P_crit = 0.25* (np.pi**2 * E * I)/(K*L)**2


# Plotting the results side by side
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(thickness, F)
plt.xlabel('Thickness (m)')
plt.ylabel('Max Force (N)')
plt.yscale('log')
plt.title('Max Force vs Thickness')
plt.grid()

plt.subplot(1,2,2)
plt.plot(thickness, P_crit)
plt.xlabel('Thickness (m)')
plt.ylabel('Critical Buckling Load (N)')
plt.title('Critical Buckling Load vs Thickness')
plt.yscale('log')
plt.grid()
plt.show()


