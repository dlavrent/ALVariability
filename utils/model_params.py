'''
Default parameter values 
From AL literature
'''
import numpy as np

params = {}

###############################################
############## PARAMETER SETTING ##############
###############################################

        
# resting potential in V 
params['V0_ORN'] = -70e-3               # Dubin and Harris, 1997
params['V0_LN'] = -50e-3                # Seki et al, 2010
params['V0_PN'] = -55e-3                # Jeanne and Wilson, 2015

# firing threshold in V
params['Vthr_ORN'] = -50e-3             # Dubin and Harris, 1997
params['Vthr_LN'] = -40e-3              # Seki et al, 2010
params['Vthr_PN'] = -40e-3              # Jeanne and Wilson, 2015

# peak action potential voltage in V
params['Vmax_ORN'] = 0e-3               # Dubin, 1997
params['Vmax_LN'] = 0e-3                # Amplitude about 60 mV (Seki et al, 2010)
params['Vmax_PN'] = -30e-3              # Amplitude about 12 mV (Wilson and Laurent, 2005)

# spike undershoot potential in V
params['Vmin_ORN'] = -70e-3             # Estimated from Fig 1C in Cao et al, 2016, and assumed = V0
params['Vmin_LN'] = -60e-3              # Assumed lower than PNs
params['Vmin_PN'] = -55e-3              # Jeanne and Wilson, 2015

# membrane capacitance in F
params['Cmem_ORN'] = 73e-12  			# Set = PNs, but there's also 1.44 pF from moth (Levakova et al, 2019)
params['Cmem_LN'] = 64e-12    			# Using skeleton length estimate in Huang et al, 2018
params['Cmem_PN'] = 73e-12    			# Using skeleton length estimate in Huang et al, 2018

# membrane resistance in Ohm
params['Rmem_ORN'] = 1.8e9    			# Dubin and Harris, 1997
params['Rmem_LN'] = 1e9      			# Seki et al, 2010
params['Rmem_PN'] = 0.3e9      			# Jeanne and Wilson, 2015

# action potential duration in s
params['APdur_ORN'] = 2e-3				# assumed from Jeanne and Wilson, 2005
params['APdur_LN'] = 4e-3               # Seki et al, 2010
params['APdur_PN'] = 2e-3				# Jeanne and Wilson, 2005

# PSC amplitude in amps
params['PSCmag_ORN'] = 1.8e-12			# assumed from Kazama and Wilson, 2008
params['PSCmag_LN'] = 1.8e-12			# assumed from Kazama and Wilson, 2008
params['PSCmag_PN'] = 1.8e-12			# Kazama and Wilson, 2008 ## or 10.3 Luo et al, 2019

# PSC rise time in s
params['PSCrisedur_ORN'] = 1e-3			# assumed from Kazama and Wilson, 2008
params['PSCrisedur_LN'] = 1e-3			# assumed from Kazama and Wilson, 2008
params['PSCrisedur_PN'] = 1e-3			# Kazama and Wilson, 2008 ## or 0.38 Luo et al, 2019

# PSC decay half-life in s 
params['PSCfalldur_ORN'] = 2.2e-3		# assumed from Jeanne and Wilson, 2015
params['PSCfalldur_LN'] = 2.2e-3		# assumed from Jeanne and Wilson, 2015
params['PSCfalldur_PN'] = 2.2e-3		# EPSG timescale in Jeanne and Wilson, 2015

# Multiplier on PSCmag_ORN to represent the input PSC
# (Poisson-generated) that drives an ORN (scalar)
params['PSCweight_ORN'] = 1e6			# (swept)
params['inputPSCfalldur'] = 0.1e-3      # make it decay quickly

# maximal ORN rate evoked from odor
params['odor_rate_max']= 300			# Empirical max firing rate (cite)

# spontaneous ORN firing rate 
params['spon_fr_ORN'] = 10				# (cite)