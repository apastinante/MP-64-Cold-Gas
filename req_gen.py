import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from scipy.interpolate import interp1d
from scipy.optimize import brentq, fsolve



class req_generator():
    def __init__(self, c_mass, c_power, Cd, orbit_alt, lifetime, rad_orbit_acc, v_orbit_acc, lifetime_acc, c_surface_area, dens):
        self.mass = c_mass
        self.power = c_power
        self.propulsion_per = 0.7 # fraction of total mass
        self.Cd = Cd
        self.orbit_alt = orbit_alt *1000 # convert to meters
        self.orbit_rad = (orbit_alt + earth_radius)*1000 # convert to meters
        self.lifetime = lifetime*30*24*60*60 # convert to seconds
        self.v_orbit_acc = v_orbit_acc *1000
        self.rad_orbit_acc = rad_orbit_acc*1000
        self.lifetime_acc = lifetime_acc * 30*24*60*60 # convert to seconds
        self.c_surface_area = c_surface_area # m^2
        self.dens = dens
        self.vel = np.sqrt(grav_param/self.orbit_rad) 
                

        # Check the limitting condition (in track or radial orbit accuracy)
        self.orbit_period = 2*np.pi*np.sqrt((self.orbit_rad)**3/grav_param)

        self.n = 2*np.pi/self.orbit_period
        
        self.dv_err = -self.vel + np.sqrt(grav_param*(2/self.orbit_rad - 1/(self.orbit_rad+1)))

        dr_rev = -2*np.pi*(self.Cd * self.c_surface_area * self.dens)/(self.mass)*self.orbit_rad**2

        dn_rev = (3*self.n*dr_rev)/(2*self.orbit_rad)
    
        dr = -4*np.sqrt(2*self.v_orbit_acc*-dr_rev/self.orbit_period/(3*self.n))

        self.flag = 1

        # if dr>self.rad_orbit_acc:
        #     print("The radial orbit accuracy is the limiting factor")
        #     self.max_it_dev = (dr/4)**2 * (3*(np.pi*2/self.orbit_period))/(dr_rev/self.orbit_period)
        #     print("The maximum in-track orbit deviation is: ", self.max_it_dev)
        #     self.flag = 1
        # else:
        #     print("The in-track orbit accuracy is the limiting factor")
        #     print("The maximum radial orbit deviation is: ", np.abs(dr))
        #     self.max_rad_dev = np.abs(dr)
        #     self.flag = 2
        

    def compute_dv(self):
        # Update the reduction in orbital altirude and the delta-v
        # required to perform the maneuver, and if the condition is met
        # then perform the maneuver
        self.dr_rev = -2*np.pi*(self.Cd * self.c_surface_area * self.dens)/(self.mass)*self.orbit_rad**2

        if self.flag == 1: 
            dv_rev = np.pi*(self.Cd * self.c_surface_area * self.dens)/(self.mass)*self.orbit_rad*self.vel
            dv_man = self.rad_orbit_acc/self.dr_rev*dv_rev * -1
            self.dv_man= dv_man
            dt_man = self.rad_orbit_acc/self.dr_rev * self.orbit_period * -1
            self.dt = dt_man
            n_man = self.lifetime/self.orbit_period
            self.dv = n_man * dv_rev * 1.05 # add 5% margin according to R-DV-1
        
        else:
            self.dr_dt = self.dr_rev / self.orbit_period
            self.dt = 4*np.sqrt(2*self.v_orbit_acc/(3*self.n*-1*self.dr_dt))
            self.dv_man = 0.5*self.n*self.dt*-1*self.dr_dt
            n_man = self.lifetime/self.dt
            self.dt = self.dt
            self.dv_man = self.dv_man
            self.dv = n_man * self.dv_man *1.05 # add 5% margin according to R-DV-1
            
    def compute_thrust_req(self):
        # Calculate the thrust required to perform the maneuver
        # asuming no gravity losses and 3 degree loss due to the
        # thrust vectoring losses
        self.thrust_req = self.mass*self.dv_man/self.dt / np.cos(3*np.pi/180)
        # self.thrust_req = 0.5 *self.Cd * self.c_surface_area * self.dens * self.vel**2 / np.cos(3*np.pi/180)
    
    def compute_isp_req(self, Isp):
        # Calculate the dry mass at each Isp 
        dry_mass = self.mass*(np.exp(- self.dv/(9.80665*Isp)) - (1 - self.propulsion_per))
        return dry_mass
    
    def compute_power_req(self, thrust): # only 40% of the power is used for the thruster
        if self.flag ==1:
            if self.rad_orbit_acc > self.dr_rev:
                self.power_req = self.power * self.orbit_period *thrust / (self.mass*self.dv_man)
            else: 
                self.power_req = 0.4* self.power * self.orbit_period *thrust / (self.mass*self.dv_man) * self.rad_orbit_acc/self.dr_rev *-1
        else:
            if self.v_orbit_acc > self.dr_dt*.75*self.n*self.dt_man**2:
                self.power_req = self.power * self.orbit_period *thrust / (self.mass*self.dv_man)
            else: 
                self.power_req = 0.4* self.power * self.orbit_period *thrust / (self.mass*self.dv_man) * self.v_orbit_acc/(self.dr_dt*.75*self.n*self.dt_man**2)
        return self.power_req

def calc_cd(gamma, Re_t_mod):
        # Calculate the discharge factor using the equation 
        # from Tang and Fenn
        discharge_factor = 1 - ((gamma+1)/2)**0.75 * (3.266 - 2.128/(gamma+1))*Re_t_mod**-0.5 + 0.9428*(gamma-1)*(gamma+2)/(gamma+1)**0.5 * Re_t_mod**-1
        return discharge_factor

def calc_mass(R_t,rho,sigma,epsilon,alpha,pc,t_tanks, t_tankc, R_tank, L_tank,rho_tank):   
        # Calculate the mass of all the components

        # Calculate the mass of the tank
        m_tank = 4*np.pi*R_tank**2 *rho_tank *t_tanks + 2*np.pi*R_tank *rho_tank *t_tankc *L_tank
        m_nozzle = 2*np.pi*rho/sigma * 4 * (R_t**2*np.pi*(epsilon-1)/np.sin(alpha) * 0.5* pc*5e-3)
        m_chamber = 2* m_nozzle 
        m_feed = 2* 0.249 *2*m_chamber
        m_support =  2* 0.1 * (m_chamber +m_nozzle + m_feed +m_tank)
    #     # Calculate the thrust coefficient using the equation
        return m_tank + m_nozzle + m_chamber + m_feed + m_support ,  m_tank



def cf_pratio(p_ratio, gamma,vander_cons):
    return vander_cons*np.sqrt(2*gamma/(gamma-1)*(1-(p_ratio)**((gamma-1)/gamma))) + (p_ratio)*vander_cons/np.sqrt(2*gamma/(gamma-1) *(1-(p_ratio)**((gamma-1)/gamma))*(p_ratio)**(2/gamma)) 

def find_supersonic_solution(gamma, vander_cons, Cf):
    # Initial guess for supersonic solution
    initial_guess_supersonic = 0  # You may need to adjust this based on your problem

    # Solve for supersonic solution
    p_ratio_supersonic = brentq(lambda x: cf_pratio(x, gamma, vander_cons) - Cf, 0.00001, 0.52828, maxiter=1000)

    Cf_calc = cf_pratio(p_ratio_supersonic, gamma, vander_cons)
    # Solve for subsonic solution using the found supersonic solution as an initial guess
    p_ratio_subsonic = brentq(lambda x: cf_pratio(x, gamma, vander_cons) - Cf, 0.52828, 1, maxiter=1000)

    return p_ratio_subsonic, p_ratio_supersonic


class piping():
    def __init__(self, v,  L, D , rho, mu):
        self.L= L
        self.D = D
        self.rho = rho
        self.v = v
        self.mu = mu
        self.zeta = 0.7 # local head loss at valve
        self.Re = self.rho*self.v*self.D/self.mu

    def calc_deltap(self):
        # Calculate the pressure drop in the piping and valve
        self.deltap1 = self.zeta*self.rho*self.v**2
        if self.Re<2320:
            self.fric_factor = 64/self.Re
        elif 2320<self.Re<1e5:
            self.fric_factor = 0.3164/(self.Re**0.25)
        else:
            self.fric_factor = 0.0032 + 0.211*(1/self.Re)**0.237
        
        self.deltap2 = self.fric_factor*self.L*self.rho*self.v**2/(2*self.D)    
        self.deltap = self.deltap1 + self.deltap2
    

    

# In this project, you will propose the preliminary design of a cold gas 
# micro-propulsion system intended for compensating the atmospheric drag 
# force generated by a 3U CubeSat orbiting at 300 km altitude in Low Earth 
# Orbit, with intended lifetime of 6 months.

# Define orbit constants
grav_param = 3.986e5 * 1e9 # m^3/s^2
earth_radius = 6378.137 # km

dens_150 = 1.81e-9 # kg/m^3
dens_300 = 1.95e-11 # kg/m^3
dens_500 = 7.3e-13 # kg/m^3
dens_300_max = 1.71e-10 # kg/m^3



# Define cubesat parameters (3U)
c_drag_coeff = 2.2
c_surface_area = 0.03 # m^2
c_av_power = 12 # W
c_mass = [3.6,6]# kg

# Define thruster parameters
power_use = 0.4 # fraction of total power

# Calculate the requirements for the system
# Compute the requirements
#def __init__(self, mass, c_power, Cd, orbit_alt, lifetime, rad_orbit_acc, v_orbit_acc, lifetime_acc):

reqs = req_generator(c_mass[1], c_av_power, c_drag_coeff, 300, 6, 3, 20, 1, c_surface_area, dens_300)
reqs.compute_dv()
reqs.compute_thrust_req()
thrusts = np.linspace(1, 100, 1000)/1000 # convert to mN
power = reqs.compute_power_req(thrusts)
Isp = np.linspace(1, 200, 300)
dry_mass = reqs.compute_isp_req(Isp) * 1000 # convert to grams
prop_mass = c_mass[0]*(np.exp(reqs.dv/(9.80665*Isp)) - 1)/np.exp(reqs.dv/(9.80665*Isp)) *1000

response_time = reqs.dv_err*reqs.mass/(0.5*1e-3)
response_time = 5e-3 #s

maximum_thrust = reqs.dv_man*reqs.mass/(response_time)

print('The maximum thrust is: ', np.round(maximum_thrust*1000,5), ' mN')

print('The required thrust is: ', np.round(reqs.thrust_req*1000,5), ' mN')
print('The delta-v error is: ', reqs.dv_err, ' m/s')
print('The required response time is: ', np.round(response_time,2), ' s')

#Find where the dry mass is 0 
index = np.where(dry_mass<1000)
ind = index[0][-1]

min_isp = Isp[index[0][-1]]

print('The total required delta-v is: ', np.round(reqs.dv,2), ' m/s')
print('The minimum required thrust is: ', np.round(reqs.thrust_req*1000,5), ' mN')
# print('The time between maneuvers is: ', np.round(reqs.power_req,2), ' W')

# # plot the dry mass vs the Isp with a logarithmic scale
plt.plot(Isp, dry_mass, label='Dry mass')
plt.axvline(x=Isp[index[0][-1]], color='r', linestyle='--', label='Isp = '+str(np.round(Isp[index[0][-1]],2))+' s')
plt.fill_between(Isp, dry_mass, where=(Isp>=Isp[index[0][-1]]), color='green', alpha=0.5)

plt.xlabel(r'$I_{sp}$ [s]')
plt.ylabel('Dry Mass [g]')
plt.legend()
plt.title('Dry Mass vs Isp')
plt.ylim(bottom=0)
plt.grid()

plt.show()

# plt.plot(Isp, prop_mass, label='Propellant mass')
# plt.axvline(x=Isp[index[0][-1]], color='r', linestyle='--', label='Isp = '+str(np.round(Isp[index[0][-1]],2))+' s')
# plt.fill_between(Isp, prop_mass, where=(Isp<=Isp[index[0][-1]]), color='red', alpha=0.5)
# plt.fill_between(Isp, prop_mass, where=(Isp>=Isp[index[0][-1]]), color='green', alpha=0.5)

# plt.xlabel(r'$I_{sp}$ [s]')
# plt.ylabel('Propellant Mass [g]')
# plt.legend()
# plt.title('Propellant Mass vs Isp')
# plt.grid()
# plt.show()

# plt.plot(thrusts*1000, power, label='Power')
# plt.fill_between(thrusts*1000, power, where=(thrusts*1000>=reqs.thrust_req), color='green', alpha=0.5)
# plt.xlabel('Thrust [mN]')
# plt.ylabel('Power [W]')
# plt.legend()
# plt.title('Power vs Thrust')
# plt.grid()
# plt.show()



# Now we can load all the data from the excel file
# [Company/Institution,Name,Propellant,Thrust (mN),Power (W),T/P (mN/W),Dry mass (g),Isp (s)]
# But skip the first row 
thrusters = pd.read_csv('colg_gas_info.csv', delimiter=';')

# Find all the propellants that are unique in the list
propellants = np.unique(thrusters['Propellant'].values)
props = {
    'Argon': [39.9, 1.67], 
    'Nitrogen': [28 , 1.401], 
    'SF6': [146.1, 1.29], 
    'SO2': [64.1, 1.29], 
    'Xenon': [131.3, 1.66]
} 

#Calculate the limit velocity for each propellant in props which has molar mass and specific heat ratio
for prop in props:
    print('The maximum specific impulse for ', prop, ' is: ', np.round(np.sqrt(2*(props[prop][1])/(props[prop][1]-1) *8314.4621/props[prop][0]*300)/9.80665,2), ' s')
    # Append the specific impolses to the props dictionary
    props[prop].append(np.sqrt(2*(props[prop][1])/(props[prop][1]-1) *8314.4621/props[prop][0]*300)/9.80665)


pressures = np.linspace(1, 10, 10) * 1e5 # Pa

# Create a class for the thruster to set the relevant 
# parameters for the chamber, nozzle and inlet and 
# evaluate the performance




# Load the properties of Nitrogen in the pressure range chosen (1-3) bar with increments of 0.05 bar
properties = pd.read_csv('Nitrogen_Properties.csv', delimiter=',')

# Interpolate the properties to the pressure range chosen at 300 K
# Temperature (K);Pressure (bar);Density (kg/m3);Cv (J/mol*K);Cp (J/mol*K);Sound Spd. (m/s);Viscosity (uPa*s);Specific heat
temp = 300 # K
pressures = properties['Pressure (bar)'].values * 1e5 # Pa
densities = interp1d(pressures, properties['Density (kg/m3)'].values)
viscosities = interp1d(pressures, properties['Viscosity (uPa*s)'].values * 1e-6) # Pa*s
specific_heats_ratio = interp1d(pressures, properties['Specific heat'].values) 

# # Plot the interpolated properties
# plt.plot(pressures, densities(pressures), label='Density')

# plt.xlabel('Pressure [Pa]')
# plt.ylabel('Density [kg/m^3]')
# plt.legend()
# plt.grid()
# plt.show()

# plt.plot(pressures, viscosities(pressures), label='Viscosity')
# plt.xlabel('Pressure [Pa]')
# plt.ylabel('Viscosity [Pa*s]')
# plt.legend()
# plt.grid()
# plt.show()



# plt.plot(pressures, specific_heats_ratio(pressures), label='Specific heat ratio')
# plt.xlabel('Pressure [Pa]')
# plt.ylabel('Specific heat ratio')
# plt.legend()
# plt.grid()
# plt.show()



mmass = 28.0134 # kg/kmol
R_nitrogen = 8314.4621/28.0134 # J/kg*K
R_helium = 8314.4621/4.0026 # J/kg*K
# Define material properties: 316 Stainless Steel
density = 8000 # kg/m^3
yield_strength = 275e6 # Pa
ultimate_strength = 505e6 # Pa
modulus_elasticity = 200e9 # Pa
poisson_ratio = 0.3

# Define material properties for Ti-6Al-4V
density_t = 4430 # kg/m^3
yield_strength_t = 880e6 # Pa
ultimate_strength_t = 950e6 # Pa
modulus_elasticity_t = 113e9 # Pa
poisson_ratio_t = 0.34



div_angles = np.array([15,20])* np.pi/180 # rad
div_loss = 1- (1-np.cos(div_angles))/2 # divergence losses

# Define constants and parameters
u_param = 2 # U's of volume
V_tank = u_param*0.1*0.1*0.1 # m^3 (1.3U)

R_tank = 10/200 # m

# Assume cylindrical tank with hemispherical ends
# Length
L_tank = (V_tank-4/3 *np.pi *R_tank**3)/(np.pi*R_tank**2) # m

r_l = 12.5e-6 # m 


corr_vals_15 = []
corr_vals_20 = []

incorr_vals_15 = []
incorr_vals_20 = []

incorr_vals_15_complete = []
incorr_vals_20_complete = []

incorr_vals_15_complete_tank = []
incorr_vals_20_complete_tank = []

Isps = np.linspace(min_isp, 0.8*props['Nitrogen'][2], 100)
pressures = np.linspace(0.1, 20, 500) * 1e5 # Pa
for pc in pressures:
    for isp in Isps:
        for i,div_angle in enumerate(div_angles):
            v_list15 = []
            v_list20 = []

            # V_tank = u_param*0.1*0.1*0.1 # m^3 
            # R_tank = 9/200 
            # L_tank = (V_tank-4/3 *np.pi *R_tank**3)/(np.pi*R_tank**2) # m

            # Calculate the required mass of propellant
            prop_mass_req = c_mass[1]*(np.exp(reqs.dv/(9.80665*isp)) - 1)/np.exp(reqs.dv/(9.80665*isp))


            # Calculate flow properties at the feed pressure
            gamma = 1.4 #specific_heats_ratio(pc)
            rho = pc/(R_nitrogen*temp) #densities(pc)
            mu = 17.89e-6 #viscosities(pc)

            # Calculate the mass flow rate
            m_flow_req = 0.5*1e-3 / (isp*9.80665) / div_loss[i] # kg/s
            
            # Calculate throat radius
            vander_cons = np.sqrt(gamma)*(2/(gamma+1))**((gamma+1)/(2*(gamma-1)))

            A_t = m_flow_req/(pc *vander_cons) *  np.sqrt(R_nitrogen*temp)

            R_t = np.sqrt(A_t/np.pi)


            Cf = (0.5e-3)/(A_t*pc) 

            # p_rat_var = sp.symbols('p_rat_var')
            func_cf = lambda p_ratio : cf_pratio(p_ratio, gamma, vander_cons) - Cf
            try:    
                p_ratio_sub , p_ratio = find_supersonic_solution(gamma, vander_cons, Cf)
                
                # Check that the pressure ratio is a zero of the function
                # if not then the design is infeasible
                func_val = func_cf(p_ratio)
        
                if np.isnan(func_val) or np.abs(func_val)>1e-3:
                    flag = 0
                    incorr_vals_15.append([pc, isp, flag, func_val])
                    incorr_vals_20.append([pc, isp, flag, func_val])
                    continue

            except:
                flag = 0
                # print('The pressure ratio is not real or nan at pressure: ', pc, ' and Isp: ', isp)
                # print(gamma,rho)
                incorr_vals_15.append([pc, isp, flag])
                incorr_vals_20.append([pc, isp, flag])
                continue


            # If the pressure ratio is nan or not real 
            # then the design is infeasible
            # and we move on to the next iteration
            

            # Calculate the expansion ratio
            expansion_ratio = vander_cons * 1/np.sqrt((2*gamma/(gamma-1)) * p_ratio**(2/gamma) * (1-p_ratio**((gamma-1)/gamma)))

            # Calculate flow velocity 
            v = m_flow_req/(rho * np.pi * (0.8e-3)**2) # m/s

            v_req = 175/rho**0.43 # m/s

            if v>v_req:
                flag = 1
                incorr_vals_15.append([pc, isp, flag])
                incorr_vals_20.append([pc, isp, flag])
                continue
            
            # Calculate the nozzle exit radius
            R_e = R_t * np.sqrt(expansion_ratio)

            # If the throat radius is smaller than 10 um then the design is infeasible
            # or if it is larger than 0.53 mm
            # and we move on to the next iteration
            if R_t < 10e-6 or R_t > 0.53e-3:
                flag = 2
                incorr_vals_15.append([pc, isp, flag])
                incorr_vals_20.append([pc, isp, flag])
                continue

            # Calculate the discharge coefficient 
            T_crit_ratio = (2/(gamma+1))
            rho_crit_ratio = T_crit_ratio**(1/(gamma-1))
            T_t = temp*T_crit_ratio
            v_t = np.sqrt(gamma*R_nitrogen*T_t)
            Re_t = rho_crit_ratio*rho*v_t*R_t/mu
            Re_t_mod = Re_t * np.sqrt(R_t/r_l)
            Cd = calc_cd(gamma, Re_t_mod)

            
            # Calculate the change in R_t
            R_t_mod = R_t/(Cd)**0.5

            R_t = R_t_mod

            R_e = R_t * np.sqrt(expansion_ratio)
            

            pipes = piping(v, 26.7e-3, 1.6e-3, rho, mu)
            pipes.calc_deltap()

            # Calculate the pressure drop in the piping and valve
            deltap = pipes.deltap
            
            # for F in np.linspace(0.7, 0.9, 30): # Evaluate the tank fill fraction 
            # Calculate the pressure at the tank
            p_final = pc + deltap
            
            # Loop over volumes to find the correct tank size
            for u_param in np.linspace(1, 2, 10):
                # Calculate the tank volume
                V_tank = u_param*0.1*0.1*0.1 # m^3 (1.3U)
                R_tank = 8.5/200 # m

                # Assume cylindrical tank with hemispherical ends
                # Length
                L_tank = (V_tank-4/3 *np.pi *R_tank**3)/(np.pi*R_tank**2) # m

                #Solve for the fill fraction: 
                F = (prop_mass_req*R_nitrogen*temp/V_tank)/(p_final + prop_mass_req*R_nitrogen*temp/V_tank)

                p_init = prop_mass_req/(F*V_tank) *(R_nitrogen*temp)
            

                m_gas = p_final*V_tank/(R_helium*temp)
                t_tank_sph = p_init*R_tank/(2*yield_strength_t/3)
                t_tank_cyl = p_init*R_tank/(yield_strength_t/3)

                # Make sure the tank thickness is achievable, otherwise increase it
                if t_tank_cyl<0.5/1000:
                    t_tank_cyl = 0.5/1000
                if t_tank_sph < 0.5/1000:
                    t_tank_sph = 0.5/1000

                
                # Calculate the dry mass of the system 

                d_mass , m_tank = calc_mass(R_t,density,yield_strength,expansion_ratio,div_angle,pc,t_tank_sph, t_tank_cyl, R_tank, L_tank, density_t) # convert to grams

                d_mass = d_mass*1000 # convert to grams
                m_tank = m_tank*1000 # convert to grams

                d_mass += 6*4.7 # add valve 

                total_mass = d_mass + prop_mass_req*1000  + m_gas*1000
                if i == 0:
                    v_list15.append([pc, isp, total_mass, d_mass ,m_flow_req, v, t_tank_cyl*1000, t_tank_sph*1000 ,R_t*1000, R_e*1000, p_ratio, expansion_ratio, Cd, deltap, m_gas*1000, F, div_angle, prop_mass_req,p_init, L_tank,V_tank])
                else:
                    v_list20.append([pc, isp, total_mass, d_mass ,m_flow_req, v, t_tank_cyl*1000, t_tank_sph*1000 ,R_t*1000, R_e*1000, p_ratio, expansion_ratio, Cd, deltap, m_gas*1000, F, div_angle, prop_mass_req,p_init, L_tank,V_tank])

            # Find the indice in the list which satisfy the size requirements
            if i == 0:
                ind_real_15 = np.where(np.array(R_tank + np.array(v_list15)[:,6]/1000 < 10/200))

                # If the list is empty then then move to next iteration
                if len(ind_real_15[0]) == 0:
                    flag = 3
                    incorr_vals_15.append([pc, isp, flag])
                    continue
                
                # If the list is not empty then find the minimum total mass
                ind_min_dummy = np.where(np.array(v_list15)[ind_real_15[0],2]==np.min(np.array(v_list15)[ind_real_15[0],2]))
                ind_min_15 = int(ind_real_15[0][0])
                total_mass = np.min(np.array(v_list15)[ind_real_15[0],2])
                
                
            else:
                ind_real_20 = np.where(np.array(R_tank + np.array(v_list20)[:,6]/1000 < 10/200))

                # If the list is empty then then move to next iteration
                if len(ind_real_20[0]) == 0:
                    flag = 3
                    incorr_vals_20.append([pc, isp, flag])
                    continue

                # If the list is not empty then find the minimum total mass
                ind_min_dummy = np.where(np.array(v_list20)[ind_real_20[0],2]==np.min(np.array(v_list20)[ind_real_20[0],2]))
                ind_min_20 = int(ind_real_20[0][0])
                total_mass = v_list20[ind_min_20][2]


            if total_mass > 0.65*c_mass[1]*1000:
                if i == 0: 
                    incorr_vals_15_complete.append(v_list15[ind_min_15])
                else:
                    incorr_vals_20_complete.append(v_list20[ind_min_20])

                continue
            
            elif i == 0:
                corr_vals_15.append(v_list15[ind_min_15])
            else:
                corr_vals_20.append(v_list20[ind_min_20])




############################





############################
# Turn the lists into numpy arrays where every nested list is a row

corr_vals_15_arr = np.zeros((len(corr_vals_15), len(corr_vals_15[0])))
for i in range(len(corr_vals_15)):
    corr_vals_15_arr[i,:] = corr_vals_15[i]

corr_vals_20_arr = np.zeros((len(corr_vals_20), len(corr_vals_20[0])))
for i in range(len(corr_vals_20)):
    corr_vals_20_arr[i,:] = corr_vals_20[i]



incorr_vals_15_arr = np.zeros((len(incorr_vals_15), len(incorr_vals_15[0])))
for i in range(len(incorr_vals_15)):
    incorr_vals_15_arr[i,:] = incorr_vals_15[i]

incorr_vals_20_arr = np.zeros((len(incorr_vals_20), len(incorr_vals_20[0])))
for i in range(len(incorr_vals_20)):
    incorr_vals_20_arr[i,:] = incorr_vals_20[i]

try:
    incorr_vals_15_complete_arr = np.zeros((len(incorr_vals_15_complete), len(incorr_vals_15_complete[0])))
    for i in range(len(incorr_vals_15_complete)):
        incorr_vals_15_complete_arr[i,:] = incorr_vals_15_complete[i]
except:
    pass

try:
    incorr_vals_20_complete_arr = np.zeros((len(incorr_vals_20_complete), len(incorr_vals_20_complete[0])))
    for i in range(len(incorr_vals_20_complete)):
        incorr_vals_20_complete_arr[i,:] = incorr_vals_20_complete[i]

except:
    pass
corr_vals_15 = corr_vals_15_arr
corr_vals_20 = corr_vals_20_arr
incorr_vals_15 = incorr_vals_15_arr
incorr_vals_20 = incorr_vals_20_arr
incorr_vals_15_complete_tank = np.array(incorr_vals_15_complete_tank)
incorr_vals_20_complete_tank = np.array(incorr_vals_20_complete_tank)







# Plot the results for the incorrect (unfeasible) designs
#  (1 figure for each divergent angle)
# (flag=1 is red, flag=2 is orange and flag=3 is black)
# Then also plot the correct values with green 

ind_flag_0 = np.where(incorr_vals_15[:,2]==0)
ind_flag_1 = np.where(incorr_vals_15[:,2]==1)
ind_flag_2 = np.where(incorr_vals_15[:,2]==2)
ind_flag_3 = np.where(incorr_vals_15[:,2]==3)


# Find the minimum total mass for each divergence angle for the incorrect designs
# only the one with flag = 3
# ind_min_15 = np.where(incorr_vals_15_complete_arr[:,4]==np.min(incorr_vals_15_complete_arr[:,4]))
# ind_min_20 = np.where(incorr_vals_20_complete_arr[:,4]==np.min(incorr_vals_20_complete_arr[:,4]))

# # Print the results dry mass, propellant mass, gas mass, total mass, isp, pressure, fill fraction for each divergence angle
# print('Divergence angle of 15 degrees')
# print('The minimum dry mass is: ', np.round(incorr_vals_15_complete_arr[ind_min_15[0][0],5],2), ' g')
# print('The minimum propellant mass is: ', np.round(incorr_vals_15_complete_arr[ind_min_15[0][0],6],2), ' g')
# print('The minimum gas mass is: ', np.round(incorr_vals_15_complete_arr[ind_min_15[0][0],7],2), ' g')
# print('The minimum total mass is: ', np.round(incorr_vals_15_complete_arr[ind_min_15[0][0],4],2), ' g')
# print('The minimum isp is: ', np.round(incorr_vals_15_complete_arr[ind_min_15[0][0],1],2), ' s')
# print('The minimum pressure is: ', np.round(incorr_vals_15_complete_arr[ind_min_15[0][0],0],2), ' Pa')
# print('The minimum fill fraction is: ', np.round(incorr_vals_15_complete_arr[ind_min_15[0][0],3],2), ' %')

# print('Divergence angle of 20 degrees')
# print('The minimum dry mass is: ', np.round(incorr_vals_20_complete_arr[ind_min_20[0][0],5],2), ' g')
# print('The minimum propellant mass is: ', np.round(incorr_vals_20_complete_arr[ind_min_20[0][0],6],2), ' g')
# print('The minimum gas mass is: ', np.round(incorr_vals_20_complete_arr[ind_min_20[0][0],7],2), ' g')
# print('The minimum total mass is: ', np.round(incorr_vals_20_complete_arr[ind_min_20[0][0],4],2), ' g')
# print('The minimum isp is: ', np.round(incorr_vals_20_complete_arr[ind_min_20[0][0],1],2), ' s')
# print('The minimum pressure is: ', np.round(incorr_vals_20_complete_arr[ind_min_20[0][0],0],2), ' Pa')
# print('The minimum fill fraction is: ', np.round(incorr_vals_20_complete_arr[ind_min_20[0][0],3],2), ' %')


plt.scatter(incorr_vals_15[ind_flag_0,0], incorr_vals_15[ind_flag_0,1], color='blue', label='Unachievable exit velocty')
# plt.scatter(incorr_vals_15[ind_flag_1,0], incorr_vals_15[ind_flag_1,1], color='purple', label='Velocity too high')
# plt.scatter(incorr_vals_15[ind_flag_2,0], incorr_vals_15[ind_flag_2,1], color='red', label='Throat radius too small')
# plt.scatter(incorr_vals_15[ind_flag_3,0], incorr_vals_15[ind_flag_3,1], color='red', label='Tank too large')
plt.scatter(incorr_vals_15_complete_arr[:,0], incorr_vals_15_complete_arr[:,1], color='orange', label='Dry mass too high')
plt.scatter(corr_vals_15_arr[:,0], corr_vals_15_arr[:,1], color='green', label='Feasible design')
plt.xlabel('Pressure [Pa]')
plt.ylabel('Isp [s]')
plt.title('Incorrect designs for divergence angle of 15 degrees')
plt.legend(loc='lower left')
plt.grid()
plt.show()

ind_flag_0 = np.where(incorr_vals_20[:,2]==0)
ind_flag_1 = np.where(incorr_vals_20[:,2]==1)
ind_flag_2 = np.where(incorr_vals_20[:,2]==2)
ind_flag_3 = np.where(incorr_vals_20[:,2]==3)

plt.scatter(incorr_vals_20[ind_flag_0,0], incorr_vals_20[ind_flag_0,1], color='blue', label='Unachievable exit velocty')
# plt.scatter(incorr_vals_20[ind_flag_1,0], incorr_vals_20[ind_flag_1,1], color='red', label='Velocity too high')
# plt.scatter(incorr_vals_20[ind_flag_2,0], incorr_vals_20[ind_flag_2,1], color='orange', label='Throat radius too small')
# plt.scatter(incorr_vals_20[ind_flag_3,0], incorr_vals_20[ind_flag_3,1], color='red', label='Tank too large')
plt.scatter(incorr_vals_20_complete_arr[:,0], incorr_vals_20_complete_arr[:,1], color='orange', label='Dry mass too high')
plt.scatter(corr_vals_20_arr[:,0], corr_vals_20_arr[:,1], color='green', label='Feasible design')
plt.xlabel('Pressure [Pa]')
plt.ylabel('Isp [s]')
plt.title('Incorrect designs for divergence angle of 20 degrees')
plt.legend(loc='lower left')
plt.grid()
plt.show()


# isp_bounds = [ np.min(corr_vals_15_arr[:,0]) , np.max(corr_vals_15_arr[:,0])]
# pressure_bounds = [ np.min(corr_vals_15_arr[:,1]) , np.max(corr_vals_15_arr[:,1])]

# # Find all the unique values of the pressure and isp in the correct designs
# pressures_plot = np.unique(corr_vals_15_arr[:,0])
# isps_plot = np.unique(corr_vals_15_arr[:,1])

# pressures_plot2 = np.unique(corr_vals_20_arr[:,0])
# isps_plot2 = np.unique(corr_vals_20_arr[:,1])

# # Loop through the pressures_plot and isps_plot and find the minimum total mass at each pressure and isp
# # for the divergence angle of 15 degrees
# corr_val_min_15 = []
# for i, pressure in enumerate(pressures_plot):
#     for j, isp in enumerate(isps_plot):
#         try:
#             ind = np.where((corr_vals_15_arr[:,0]==pressure) & (corr_vals_15_arr[:,1]==isp))

#             corr_val_min_15.append(corr_vals_15_arr[ind[0][0],:])
#             colors_15[i,j] = corr_vals_15_arr[ind[0][0],2]
#         except:
#             continue

# corr_val_min_15 = np.array(corr_val_min_15)
# pressure_plot_c1 = np.unique(corr_val_min_15[:,0])
# pressure
# colors_15 = np.zeros((len(pressures_plot_c1), len(isps_plot_c1)))
# for i, pressure in enumerate(pressures_plot):
#     for j, isp in enumerate(isps_plot):
#         try:
#             ind = np.where((corr_vals_15_arr[:,0]==pressure) & (corr_vals_15_arr[:,1]==isp))
#             corr_val_min_15.append(corr_vals_15_arr[ind[0][0],:])
#             colors_15[i,j] = corr_vals_15_arr[ind[0][0],2]
        


# # Loop through the pressures_plot and isps_plot and find the minimum total mass at each pressure and isp
# # for the divergence angle of 20 degrees
# corr_val_min_20 = []
# colors_20 = np.zeros((len(pressures_plot2), len(isps_plot2)))

# for i, pressure in enumerate(pressures_plot2):
#     for j, isp in enumerate(isps_plot2):
#         ind = np.where((corr_vals_20_arr[:,0]==pressure) & (corr_vals_20_arr[:,1]==isp))
#         corr_val_min_20.append(corr_vals_20_arr[ind[0][0],:])
#         colors_20[i,j] = corr_vals_20_arr[ind[0][0],2]

# corr_val_min_20 = np.array(corr_val_min_20)

# # Plot the 3d scatterplot results for the minimum total mass at each 
# # pressure and isp for the divergence angle of 15 degrees where the 
# # color is representative of the total mass
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(corr_val_min_15[:,0], corr_val_min_15[:,1], corr_val_min_15[:,2], c=colors_15.flatten(), cmap='viridis', label='Feasible design 15 degrees')
# ax.set_xlabel('Pressure [Pa]')
# ax.set_ylabel('Isp [s]')
# ax.set_zlabel('Total mass [g]')
# plt.title('Feasible designs for divergence angle of 15 degrees')
# plt.grid()
# plt.show()

# # Plot the 3d scatterplot results for the minimum total mass at each
# # pressure and isp for the divergence angle of 20 degrees where the
# # color is representative of the total mass
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(corr_val_min_20[:,0], corr_val_min_20[:,1], corr_val_min_20[:,2], c=colors_20.flatten(), cmap='viridis', label='Feasible design 20 degrees')
# ax.set_xlabel('Pressure [Pa]')
# ax.set_ylabel('Isp [s]')
# ax.set_zlabel('Total mass [g]')
# plt.title('Feasible designs for divergence angle of 20 degrees')
# plt.grid()
# plt.show()




# # Plot a 3d scatter plot with the mass as the z-axis and the pressure and Isp as the x and y axis
# # (1 figure for each divergent angle) only for the correct designs
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(corr_val_min_15[:,0], corr_val_min_15[:,1], corr_val_min_15[:,2], label='Feasible design 15 degrees')
# ax.set_xlabel('Pressure [Pa]')
# ax.set_ylabel('Isp [s]')
# ax.set_zlabel('Total mass [g]')
# plt.title('Feasible designs for divergence angle of 15 degrees')
# plt.grid()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(corr_val_min_20[:,0], corr_val_min_20[:,1], corr_val_min_20[:,2], label='Feasible design 20 degrees')
# ax.set_xlabel('Pressure [Pa]')
# ax.set_ylabel('Isp [s]')
# ax.set_zlabel('Total mass [g]')
# plt.title('Feasible designs for divergence angle of 20 degrees')
# plt.grid()






# Find out of the correct designs which one is the lightest for each divergence angle

ind_15 = np.where(corr_vals_15[:,2]==np.min(corr_vals_15[:,2]))
ind_20 = np.where(corr_vals_20[:,2]==np.min(corr_vals_20[:,2]))

print('The lightest design for a divergence angle of 15 degrees is: ', corr_vals_15[ind_15[0][0],0], ' Pa and ', corr_vals_15[ind_15[0][0],1], ' s')
print('The lightest design for a divergence angle of 20 degrees is: ', corr_vals_20[ind_20[0][0],0], ' Pa and ', corr_vals_20[ind_20[0][0],1], ' s')


# Print the mass flow rate, flow velocity, throat radius, exit radius, 
# order: pc, isp, total_mass, d_mass ,m_flow_req, v, t_tank*1000, R_t*1000, R_e*1000, p_ratio, expansion_ratio, Cd, deltap, m_gas*1000, F# for the lightest design for each divergence angle
print('Divergence angle: 15 degrees')
print('Plenum Pressure: ', corr_vals_15[ind_15[0][0],0], ' Pa')
print('Specific impulse: ', corr_vals_15[ind_15[0][0],1], ' s')
print('Total mass: ', corr_vals_15[ind_15[0][0],2], ' g')
print('Dry mass: ', corr_vals_15[ind_15[0][0],3], ' g')
print('Mass flow rate: ', corr_vals_15[ind_15[0][0],4]*1000, ' g/s')
print('Flow velocity: ', corr_vals_15[ind_15[0][0],5], ' m/s')
print('Tank thickness cylinder: ', corr_vals_15[ind_15[0][0],6], ' mm')
print('Tank thickness sphere: ', corr_vals_15[ind_15[0][0],7], ' mm')
print('Throat radius: ', corr_vals_15[ind_15[0][0],8], ' mm')
print('Exit radius: ', corr_vals_15[ind_15[0][0],9], ' mm')
print('Pressure ratio: ', corr_vals_15[ind_15[0][0],10])
print('Expansion ratio: ', corr_vals_15[ind_15[0][0],11])
print('Discharge coefficient: ', corr_vals_15[ind_15[0][0],12])
print('Pressure drop: ', corr_vals_15[ind_15[0][0],13], ' Pa')
print('Gas mass: ', corr_vals_15[ind_15[0][0],14], ' g')
print('Fill fraction: ', corr_vals_15[ind_15[0][0],15])
print("Length of nozzle:", (corr_vals_15[ind_15[0][0],9] - corr_vals_15[ind_15[0][0],8])/np.tan(15*np.pi/180), 'mm')
print('Initial Pressure in tank:', corr_vals_15[ind_15[0][0],-3] , 'Pa')
print("Tank length:", corr_vals_15[ind_15[0][0],-2]*1000 , 'mm') # L_tank
print("Tank volume:", corr_vals_15[ind_15[0][0],-1]/0.1**3 , 'U') # V_tank

      



print('')
print('Plenum Pressure: ', corr_vals_20[ind_20[0][0],0], ' Pa')
print('Specific impulse: ', corr_vals_20[ind_20[0][0],1], ' s')
print('Total mass: ', corr_vals_20[ind_20[0][0],2], ' g')
print('Dry mass: ', corr_vals_20[ind_20[0][0],3], ' g')
print('Mass flow rate: ', corr_vals_20[ind_20[0][0],4]*1000, ' g/s')
print('Flow velocity: ', corr_vals_20[ind_20[0][0],5], ' m/s')
print('Tank thickness cylinder: ', corr_vals_20[ind_20[0][0],6], ' mm')
print('Tank thickness sphere: ', corr_vals_20[ind_20[0][0],7], ' mm')
print('Throat radius: ', corr_vals_20[ind_20[0][0],8], ' mm')
print('Exit radius: ', corr_vals_20[ind_20[0][0],9], ' mm')
print('Pressure ratio: ', corr_vals_20[ind_20[0][0],10])
print('Expansion ratio: ', corr_vals_20[ind_20[0][0],11])
print('Discharge coefficient: ', corr_vals_20[ind_20[0][0],12])
print('Pressure drop: ', corr_vals_20[ind_20[0][0],13], ' Pa')
print('Gas mass: ', corr_vals_20[ind_20[0][0],14], ' g')
print('Fill fraction: ', corr_vals_20[ind_20[0][0],15])
print("Length of nozzle:", (corr_vals_20[ind_20[0][0],9] - corr_vals_20[ind_20[0][0],8])/np.tan(20*np.pi/180), 'mm')
print('Initial Pressure in tank:', corr_vals_20[ind_20[0][0],-3] , 'Pa')
print("Tank length:", corr_vals_20[ind_20[0][0],-2] *100 , 'mm') # R_tank
print("Tank volume:", corr_vals_20[ind_20[0][0],-1]/0.1**3 , 'U') # V_tank












 