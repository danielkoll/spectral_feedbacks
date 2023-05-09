import numpy as np
from . import phys, Planck
from .Thermodynamics import get_satvps,convert_molar_to_mass_ratio

import scipy.optimize

'''
THEORY SCALINGS FOR CLEARSKY LW FEEDBACKS.

** Format of scaling functions
INPUTS:
      Ts: surface temperature [K]
      Tstrat: tropopause/stratosphere temperature [K]
      RH: between 0-1
      params: Dummy object that holds thermodynamic values
              see 'default_params' and get_default_params()
      ppv_co2: CO2 volume mixing ratio. different from mass mixing ratio q_co2. e.g., 400e-6

OPTIONAL INPUTS/PARAMETERS:
      T0: temperature around which to approximate CC as a power law [K] {scalar}
      gammaLR: bulk lapse rate parameter [-] {same shape as Ts}
      dgammaLR/dTs: change of lapse rate with warming = [1/K] {same shape as Ts}
                    note, not used in get_lambda_surf()
      scaling const: optional tuning constant [-] {scalar}
                     for tuned values see e.g. 'scaling_consts_bulkLR'

OUTPUTS:
      lambda_i: different spectroscopic feedbacks  [W/m2/K]
      T_i: radiating temp(s) used in the scaling -> varies between individual feedbacks
'''


### -----------------------------------
### THEORY FUNCTIONS

#
class Dummy():
    pass

# ---
# SPECTROSCOPIC PARAMETERS
# CO2: taken from literature
co2_spec = Dummy()
co2_spec.n0 = 667.5      # [cm-1]
co2_spec.p0 = 1e5        # [Pa]
co2_spec.kappa0 = 500.   # [kg/m2]
co2_spec.ln0 = 10.2      # [cm-1]
# --
# H2O: fix band centers n0, fit (kappa,ln); fit lines first (smoothed via median), then cntm
#      Notation:
#      h2o_spec.kappa0 = kappa_{rot}, h2o_spec.ln0 = l_{rot}, 
#      h2o_spec.kappa1 = kappa_{v-r}, h2o_spec.ln1 = l_{v-r}
h2o_spec = Dummy()
h2o_spec.p0 = 1e5       # [Pa]
h2o_spec.kappa0 = 165.  # [kg/m2]
h2o_spec.n0 = 150.      # [cm-1]
h2o_spec.ln0 = 55.      # [cm-1]
h2o_spec.kappa1 = 15.   # [kg/m2]
h2o_spec.n1 = 1500.     # [cm-1]
h2o_spec.ln1 = 38.      # [cm-1]
# --
cntm_spec = Dummy()
cntm_spec.T0 = 300.     # [K] reference temp for cntm. can be different from T0 used in CC approx
cntm_spec.kappa0 = 3e-2 # [kg/m2]
cntm_spec.a = 7.        # power-law exponent;
##---


# ---
# THERMODYNAMIC PARAMETERS
#   -> set modern Earth as default. Modify this if interested in different atm. compositions, paleo, etc.
default_params = Dummy()
default_params.Rv = phys.H2O.R # moist component
default_params.cpv = phys.H2O.cp
default_params.Lvap = phys.H2O.L_vaporization_TriplePoint
default_params.satvap_T0 = phys.H2O.TriplePointT
default_params.satvap_e0 = phys.H2O.TriplePointP
default_params.esat = lambda T: get_satvps(T,default_params.satvap_T0,default_params.satvap_e0,default_params.Rv,default_params.Lvap) # use same fn as in LBL calcs
default_params.R = phys.air.R         # dry component
default_params.cp = phys.air.cp
default_params.ps_dry = 1e5           # surface pressure of dry component
default_params.g = 9.8                # surface gravity
default_params.cosThetaBar = 3./5.    # average zenith angle used in 2stream eqns
default_params.R_CO2 = phys.CO2.R     # non-condensible forcing component


# ---
# SCALING CONSTANTS
# picked to match 1D LBL calculations at RH=0.8, 400ppm CO2
# fns below expect values accessible in the form 'scaling_const.surf', scaling_const.h2o' etc

# a) consts. defined to match lambda_co2 + lambda_h2o at Ts=290K, lambda_surf at Ts=250K, lambda_cnt at Ts=330K in bulk-LR calculations
scaling_consts_bulkLR = Dummy()
scaling_consts_bulkLR.surf = 0.82
scaling_consts_bulkLR.co2 = 0.7
scaling_consts_bulkLR.h2o = 0.56
scaling_consts_bulkLR.cnt = 0.41

# b) consts. defined to match lambda_co2 + lambda_h2o at Ts=290K, lambda_surf at Ts=250K, lambda_cnt at Ts=330K in moist adiabatic calculations
scaling_consts_fullLR = Dummy()
scaling_consts_fullLR.surf = 0.81
scaling_consts_fullLR.co2 = 0.87
scaling_consts_fullLR.h2o = 0.98
scaling_consts_fullLR.cnt = 0.39

## 
default_scaling_consts = scaling_consts_fullLR  # set default
##

# ---
def get_default_params():
    return default_params

# Estimate gammaLR
def get_gammaLR(Ts,Tstrat,params=default_params):
    qsat_surf = params.R/params.Rv * params.esat(Ts)/params.ps_dry # dilute approx!
    #qsat_surf = (params.esat(Ts)/params.Rv) / (params.ps_dry/params.R + params.esat(Ts)/params.Rv) # non-dilute: use this when comparing with full LR calcs!?
    Tavg = 0.5*(Ts+Tstrat)
    gammaLR = params.R*Tavg*np.log(Ts/Tstrat) / (params.cp * (Ts-Tstrat) + params.Lvap*qsat_surf)
    return gammaLR


# -----
# ... define absorption crossections ...

# fitted functions for kappa 
def get_kappa_co2(n,p,T=None):
    ln0,n0 = co2_spec.ln0,co2_spec.n0
    kappa0 = co2_spec.kappa0
    p0 = co2_spec.p0

    kappa = kappa0 * (p/p0) * np.exp( -np.abs(n-n0)/ln0 )
    return kappa

def get_kappa_h2o(n,p,T=None,RH=1.):
    kappa0,kappa1 = h2o_spec.kappa0,h2o_spec.kappa1
    n0,n1 = h2o_spec.n0,h2o_spec.n1
    ln0,ln1 = h2o_spec.ln0,h2o_spec.ln1
    p0 = h2o_spec.p0

    kappa = np.maximum( kappa0*(p/p0)*np.exp( -np.abs(n-n0)/ln0 ), kappa1*(p/p0)*np.exp( -np.abs(n-n1)/ln1 ) )    # two bands -- NEW, correct
    return kappa

def get_kappa_selfcont(n,p,T,params=default_params,RH=1.,T0=300.):
    kappaX,TX = cntm_spec.kappa0, cntm_spec.T0
    esatX = params.esat(TX)
    
    gammaWV = params.Lvap/(params.Rv*T0)
    esat0 = params.esat(T0)
    e = RH *esat0 * (T/T0)**gammaWV

    kappa0 = kappaX * (esat0/esatX) * (T0/TX)**(-cntm_spec.a) #  if T0 for CC approx differs from cntm_spec.T0: properly rescale kappa0, so kappa0=kappa0(T0)    
    kappa = kappa0 * (e/esat0) * (T/T0)**(-cntm_spec.a) * np.ones_like(n)

    return kappa


# -----
# ... define radiating temperatures ...

# Absorption due to CO2 line wings
def get_Trad_co2(nu,Ts,gammaLR,pco2,params=default_params):
    tau0 = get_kappa_co2(nu,params.ps_dry,None) * params.ps_dry/(2.*params.g*params.cosThetaBar)  # kappa0=kappa0(ps)
    qco2 = convert_molar_to_mass_ratio(pco2,params.R_CO2,params.R)
    Trad = Ts * ( 1./(tau0 * qco2)  )**(gammaLR/2.)
    return Trad

# Absorption due to H2O line wings
def get_Trad_h2o(nu,Ts,gammaLR,RH,params=default_params,T0=300.):
    gammaWV = params.Lvap/(params.Rv * T0)
    kappa = get_kappa_h2o(nu,params.ps_dry,None,RH=1.)
    esat0 = params.esat(T0)
    tau_a = params.R/params.Rv * kappa*esat0/(params.g*params.cosThetaBar)
    Trad = T0 * ( (1.+gammaWV*gammaLR)/(tau_a * RH)  )**(gammaLR/(1.+gammaWV*gammaLR)) * (Ts/T0)**(1./(1.+gammaWV*gammaLR))
    return Trad

# Absorption due to H2O self-continuum
#  -> make sure to return correct array shape
def get_Trad_cntm(nu,Ts,gammaLR,RH,params=default_params,T0=300.):
    gammaWV = params.Lvap/(params.Rv * T0)
    kappa0 = get_kappa_selfcont(nu,None,T0,params,RH=1.,T0=T0)  # here: kappa0 defined at RH=1,T=T0
    esat0 = params.esat(T0)    
    tau_b = params.R/params.Rv * kappa0*esat0/(params.g*params.cosThetaBar)
    Trad = T0 * ( ((2.*gammaWV-cntm_spec.a)*gammaLR)/(tau_b * RH**2)  )**(1./(2.*gammaWV-cntm_spec.a))
    return Trad

# Overall radiating temp
def get_Trad_total(nu,Ts,Tstrat,gammaLR,RH,pco2,params=default_params,T0=300.):
    Tco2 = get_Trad_co2(nu,Ts,gammaLR,pco2,params=params)
    Th2o = get_Trad_h2o(nu,Ts,gammaLR,RH,params=params,T0=T0)
    Tcntm = get_Trad_cntm(nu,Ts,gammaLR,RH,params=params,T0=T0)
    Tatm = np.minimum(np.minimum(Tco2,Th2o),Tcntm)
    Trad = np.maximum(np.minimum(Ts,Tatm),Tstrat)
    return Trad


# --===================================
# ... define analytical feedbacks ...


# --
#     Surface feedback
#     -> treat lapse rate parameters gammaLR and dgammaLR/dTs as optional inputs. If not provided, use analytical bulk lapse rate scaling instead.
#        NOTE: surface feedback = surface kernel, so dgammaLR/dTs is *not* being used
#        NOTE: this function will only work well if q_co2 is constant for all Ts!
def get_lambda_surf(Ts,Tstrat,RH,params,ppv_co2,T0=300.,gammaLR=None,scaling_const=default_scaling_consts.surf):
    #
    Ts = np.atleast_1d(Ts)

    # get gammaLR
    if gammaLR is None:
        gammaLR = get_gammaLR(Ts,Tstrat,params)  # if not defined, use analytical approx
    else:
        pass   # rely on input...
    
    # ---
    # set spectroscopic parameters
    ln0,ln1 = h2o_spec.ln0,h2o_spec.ln1
    nu0,nu1 = h2o_spec.n0,h2o_spec.n1
    kappa0 = get_kappa_h2o(nu0,params.ps_dry,T=None,RH=1.)   # kappa_h2o_ref is defined at RH=1,p=p_ref (no explicit T-dependence)
    kappa1 = get_kappa_h2o(nu1,params.ps_dry,T=None,RH=1.)   # ..
    kappaCntm = get_kappa_selfcont(-1,None,T0,params,RH=1.,T0=T0)  # here: explicit T-dependence; no p-dependence
    aCntm = cntm_spec.a
    
    # set thermodynamic parameters
    gammaWV = params.Lvap/(params.Rv * T0)
    esat0 = params.esat(T0)
     
    # ...
    tau0_star = params.R/params.Rv * kappa0*esat0/(params.g*params.cosThetaBar)  # for rot. H2O band below ~700 cm-1
    tau1_star = params.R/params.Rv * kappa1*esat0/(params.g*params.cosThetaBar)  # for rot-vib. H2O band above ~1200 cm-1
    tau_cntm_star = params.R/params.Rv * kappaCntm*esat0/(params.g*params.cosThetaBar)
    
    # ---
    # Window width: here, due to H2O
    nuL_cold = nu0 + ln0 * np.log(tau0_star*RH/(1.+gammaLR*gammaWV)*(Ts/T0)**gammaWV)   # left edge: Th2o-Ts
    nuR_cold = nu1 - ln1 * np.log(tau1_star*RH/(1.+gammaLR*gammaWV)*(Ts/T0)**gammaWV)   # right edge: Th2o-Ts
    exp_fac = (1.+gammaWV*gammaLR)/((2.*gammaWV-aCntm)*gammaLR)
    nuL_hot = nu0 + ln0 * np.log(tau0_star*RH/(1.+gammaLR*gammaWV)*(T0/Ts)**(1./gammaLR)*( (2.*gammaWV-aCntm)*gammaLR/(RH**2*tau_cntm_star) )**exp_fac )   # left edge: Th2o-Tcnt
    nuR_hot = nu1 - ln1 * np.log(tau1_star*RH/(1.+gammaLR*gammaWV)*(T0/Ts)**(1./gammaLR)*( (2.*gammaWV-aCntm)*gammaLR/(RH**2*tau_cntm_star) )**exp_fac )   # right edge: Th2o-Tcnt
    nuL = np.minimum( nuL_cold, nuL_hot )
    nuR = np.maximum( nuR_cold, nuR_hot )
    
    # Window width: influence of CO2 blocking
    q_co2 = convert_molar_to_mass_ratio(ppv_co2,params.R_CO2,params.R)
    tau_co2_star = get_kappa_co2(co2_spec.n0,params.ps_dry) * params.ps_dry/(2.*params.g*params.cosThetaBar)   # CO2 in center of CO2 band ..
    if q_co2*tau_co2_star > 1:
        dnu_co2 = 2.*co2_spec.ln0*np.log(q_co2*tau_co2_star)
    else:
        dnu_co2 = 0.
        
    # ...
    nu_mid = 0.5*(nuR+nuL)
    dnu = np.maximum(0., np.maximum(0.,nuR-nuL) - dnu_co2 )

    # grey continuum transmission fn
    #      optical thickness evaluated at surface 
    trans_cntm = np.exp( -tau_cntm_star * RH**2 * 1./( (2.*gammaWV-aCntm)*gammaLR) *(Ts/T0)**(2.*gammaWV-aCntm) )

    # Surface feedback
    c = scaling_const
    lambda_surf = c*np.pi*Planck.dPlanckdT_n(nu_mid,Ts) * dnu * trans_cntm
    
    return lambda_surf



# --
# Non-Simpsonian H2O band feedback
#
# NOTES:
#     - What about CO2 blocking? In the range pCO2=0 versus pCO2=400ppm the impact is limited, so don't include here.
def get_lambda_h2o(Ts,Tstrat,RH,params,ppv_co2,T0=300.,gammaLR=None,dgammaLRdTs=None,scaling_const=default_scaling_consts.h2o):
    #
    Ts = np.atleast_1d(Ts)

    # ---
    # get gammaLR
    if gammaLR is None:
        gammaLR = get_gammaLR(Ts,Tstrat,params)  # if not defined, use analytical approx
    else:
        pass   # take as input...

    # get d(gammaLR)/dTs
    if dgammaLRdTs is None:
        dTs = 1.   # compute derivative numerically for now -> assumes a moist-adiabatic response!
        dgammaLRdTs = (get_gammaLR(Ts+dTs,Tstrat,params) - get_gammaLR(Ts,Tstrat,params))/dTs   # should be <0
    else:
        pass   # take as input...
    
    # ---
    # set spectroscopic parameters
    ln0,ln1 = h2o_spec.ln0,h2o_spec.ln1
    nu0,nu1 = h2o_spec.n0,h2o_spec.n1
    kappa0 = get_kappa_h2o(nu0,params.ps_dry,T=None,RH=1.)   # kappa_h2o_ref is defined at RH=1,p=p_ref (no explicit T-dependence)
    kappa1 = get_kappa_h2o(nu1,params.ps_dry,T=None,RH=1.)   # ..
    kappaCntm = get_kappa_selfcont(-1,None,T0,params,RH=1.,T0=T0)  # here: explicit T-dependence; no p-dependence
    aCntm = cntm_spec.a
    
    # set thermodynamic parameters
    gammaWV = params.Lvap/(params.Rv * T0)
    esat0 = params.esat(T0)
     
    # ...
    tau0_star = params.R/params.Rv * kappa0*esat0/(params.g*params.cosThetaBar)  # for rot. H2O band below ~700 cm-1
    tau_cntm_star = params.R/params.Rv * kappaCntm*esat0/(params.g*params.cosThetaBar)
    
    # ---
    # H2O band edges:
    # HERE, ONLY CONSIDER THE ROTATIONAL H2O BAND
    # - say left edge of band is always nu ~ 0
    # - at cold temperatures, right edge of H2O band is given by intersection of Th2o with Ts
    # - at hot temperatures, right edge of H2O band is given by intersection of Th2o with Tcnt
    # - notation: 'nuR' here is 'nuL' in surface & cntm feedback
    nuL = 0.
    
    nuR_cold = nu0 + ln0 * np.log(tau0_star*RH/(1.+gammaLR*gammaWV)*(Ts/T0)**gammaWV)   # left edge: Th2o-Ts
    exp_fac = (1.+gammaWV*gammaLR)/((2.*gammaWV-aCntm)*gammaLR)
    nuR_hot = nu0 + ln0 * np.log(tau0_star*RH/(1.+gammaLR*gammaWV)*(T0/Ts)**(1./gammaLR)*( (2.*gammaWV-aCntm)*gammaLR/(RH**2*tau_cntm_star) )**exp_fac )   # left edge: Th2o-Tcnt
    nuR = np.minimum( nuR_cold, nuR_hot )

    nu_mid = 0.5*(nuR+nuL)
    dnu = np.maximum(0.,nuR-nuL)

    # get Th2o plus derivatives
    Th2o = get_Trad_h2o(nu_mid,Ts,gammaLR,RH,params=params,T0=T0)   # evaluate at (Ts-varying) middle frequency of band
    dTh2odTs_ts = 1./(1. + gammaLR*gammaWV) * Th2o/Ts
    dTh2odTs_gamma = (gammaLR*gammaWV -gammaWV*np.log(Ts/T0) + np.log((1.+gammaLR*gammaWV)/(RH*tau0_star))) / ((1. + gammaLR*gammaWV)**2) * Th2o

    # H2O feedback
    c = scaling_const
    lambda_h2o = c*np.pi*Planck.dPlanckdT_n( nu_mid,Th2o ) * (dnu) * (dTh2odTs_ts + dTh2odTs_gamma * dgammaLRdTs)
    
    return lambda_h2o,Th2o



# --
# Non-Simpsonian H2O continuum feedback ...
# NOTES:
#    - here, include spectral blocking by CO2 -> same as for surface feedback
def get_lambda_cntm(Ts,Tstrat,RH,params,ppv_co2,T0=300.,gammaLR=None,dgammaLRdTs=None,scaling_const=default_scaling_consts.cnt):
    #
    Ts = np.atleast_1d(Ts)

    # set thermodynamic parameters, based on approximation of CC around T=T0
    gammaWV = params.Lvap/(params.Rv * T0)
    esat0 = params.esat(T0)
    
    # ---
    # get gammaLR
    if gammaLR is None:
        gammaLR = get_gammaLR(Ts,Tstrat,params)  # if not defined, use analytical approx
    else:
        pass   # take as input...

    # get d(gammaLR)/dTs
    if dgammaLRdTs is None:
        dTs = 1.   # compute derivative numerically for now -> assumes a moist-adiabatic response!
        dgammaLRdTs = (get_gammaLR(Ts+dTs,Tstrat,params) - get_gammaLR(Ts,Tstrat,params))/dTs   # should be <0
    else:
        pass   # take as input...

    # ---
    # set spectroscopic parameters
    ln0,ln1 = h2o_spec.ln0,h2o_spec.ln1
    nu0,nu1 = h2o_spec.n0,h2o_spec.n1
    kappa0 = get_kappa_h2o(nu0,params.ps_dry,T=None,RH=1.)   # kappa_h2o_ref is defined at RH=1,p=p_ref (no explicit T-dependence)
    kappa1 = get_kappa_h2o(nu1,params.ps_dry,T=None,RH=1.)   # ..
    kappaCntm = get_kappa_selfcont(-1,None,T0,params,RH=1.,T0=T0)  # here: explicit T-dependence; no p-dependence
    aCntm = cntm_spec.a
    
    # set thermodynamic parameters
    gammaWV = params.Lvap/(params.Rv * T0)
    esat0 = params.esat(T0)
     
    # ...
    tau0_star = params.R/params.Rv * kappa0*esat0/(params.g*params.cosThetaBar)  # for rot. H2O band below ~700 cm-1
    tau1_star = params.R/params.Rv * kappa1*esat0/(params.g*params.cosThetaBar)  # for rot-vib. H2O band above ~1200 cm-1
    tau_cntm_star = params.R/params.Rv * kappaCntm*esat0/(params.g*params.cosThetaBar)
    
    # ---
    # Window width: here, due to H2O
    nuL_cold = nu0 + ln0 * np.log(tau0_star*RH/(1.+gammaLR*gammaWV)*(Ts/T0)**gammaWV)   # left edge: Th2o-Ts
    nuR_cold = nu1 - ln1 * np.log(tau1_star*RH/(1.+gammaLR*gammaWV)*(Ts/T0)**gammaWV)   # right edge: Th2o-Ts
    exp_fac = (1.+gammaWV*gammaLR)/((2.*gammaWV-aCntm)*gammaLR)
    nuL_hot = nu0 + ln0 * np.log(tau0_star*RH/(1.+gammaLR*gammaWV)*(T0/Ts)**(1./gammaLR)*( (2.*gammaWV-aCntm)*gammaLR/(RH**2*tau_cntm_star) )**exp_fac )   # left edge: Th2o-Tcnt
    nuR_hot = nu1 - ln1 * np.log(tau1_star*RH/(1.+gammaLR*gammaWV)*(T0/Ts)**(1./gammaLR)*( (2.*gammaWV-aCntm)*gammaLR/(RH**2*tau_cntm_star) )**exp_fac )   # right edge: Th2o-Tcnt
    nuL = np.minimum( nuL_cold, nuL_hot )
    nuR = np.maximum( nuR_cold, nuR_hot )
    
    # Window width: CO2 blocking
    q_co2 = convert_molar_to_mass_ratio(ppv_co2,params.R_CO2,params.R)
    tau_co2_star = get_kappa_co2(co2_spec.n0,params.ps_dry) * params.ps_dry/(2.*params.g*params.cosThetaBar)   # in center of CO2 band ..
    if q_co2*tau_co2_star > 1:
        dnu_co2 = 2.*co2_spec.ln0*np.log(q_co2*tau_co2_star)
    else:
        dnu_co2 = 0.
        
    # ...
    nu_mid = 0.5*(nuR+nuL)
    dnu = np.maximum(0., np.maximum(0.,nuR-nuL) - dnu_co2 )
    
    # ---
    # get Tcnt
    Tcnt = np.minimum(Ts, get_Trad_cntm( 0.,Ts,gammaLR,RH,params=params,T0=T0))    # What to do at cold temps when Tcnt>Ts? Use Tcnt=min(Ts,Tcnt)
    dTcntdTs_gamma = Tcnt/( (2.*gammaWV-aCntm)*gammaLR )
    
    # grey continuum transmission fn
    #      optical thickness evaluated at surface 
    tau_cnt_surf = tau_cntm_star * RH**2 * 1./( (2.*gammaWV-aCntm)*gammaLR) *(Ts/T0)**(2.*gammaWV-aCntm)

    # Continuum feedback
    c = scaling_const
    lambda_cntm = c*np.pi*Planck.dPlanckdT_n( nu_mid,Tcnt ) *(dTcntdTs_gamma * dgammaLRdTs) *(dnu) *(1.-np.exp(-tau_cnt_surf))
    
    return lambda_cntm,Tcnt


#     CO2 feedback
#     -> treat lapse rate parameters gammaLR and dgammaLR/dTs as optional inputs. If not provided, use analytical bulk lapse rate scaling instead.
#     -> The way the fn is written assumes ppv_co2 is a constant! Don't pass ppv_co2 as an array input.
#        If want to explore variations in CO2, then loop over the function with scalar inputs instead.
#        E.g.: lambda_co2 = [get_lambda_co2(300.,200.,1.,ppv_co2) for ppv_co2 in [200e-6,400e-6,800e-6,...] ]
#
#     NOTES:
#        - at high Ts/low CO2, CO2 band center switches from being in stratosphere (present-day Earth) to being in troposphere.
#          if normalize_jump=True, add a constant 'b' to prevent the scaling from being discontinuous.
#     TO FIX:
#        - Currently normalize_jump=True only works if Tstrat is a scalar, because the newton's method func needs to return a scalar.
#          
def get_lambda_co2(Ts,Tstrat,RH,params,ppv_co2,gammaLR=None,dgammaLRdTs=None,T0=300.,scaling_const=default_scaling_consts.co2,normalize_jump=True,Ts0=310.):
    #
    Ts = np.atleast_1d(Ts)

    # ---
    # get gammaLR
    if gammaLR is None:
        gammaLR = get_gammaLR(Ts,Tstrat,params)  # if not defined, use analytical approx
    else:
        pass   # take as input...

    # get d(gammaLR)/dTs
    if dgammaLRdTs is None:
        dTs = 1.   # compute derivative numerically for now -> assumes a moist-adiabatic response!
        dgammaLRdTs = (get_gammaLR(Ts+dTs,Tstrat,params) - get_gammaLR(Ts,Tstrat,params))/dTs   # should be <0
    else:
        pass   # take as input...

    # ---
    # co2 parameters
    lk = co2_spec.ln0
    nu0 = co2_spec.n0
    q_co2 = convert_molar_to_mass_ratio(ppv_co2,params.R_CO2,params.R)
    tau_co2_star = get_kappa_co2(co2_spec.n0,params.ps_dry) * params.ps_dry/(2.*params.g*params.cosThetaBar)          # CO2 column optical thickness in center of CO2 band;

    # check if there even is a CO2 feedback. If CO2 concentration is so low that tau_co2 < 1 at all wavenrs terminate here. Else continue.
    if q_co2*tau_co2_star <= 1:
        return Ts*np.nan,Ts*np.nan,Ts*np.nan  # return output of correct shape
    else:
        pass
    
    # ---
    # get radiating temps -> just eval in center of CO2 band
    Trad_co2_nu0 = get_Trad_co2(nu0,Ts,gammaLR,ppv_co2,params=params)   # note: this can be much smaller than Tstrat; but at high Ts can also be lower than Tstrat!
    Trad_h2o_nu0 = get_Trad_h2o(nu0,Ts,gammaLR,RH,params=params)
    Trad_cntm = get_Trad_cntm(None,Ts,gammaLR,RH,params=params)          # indep of wavenr..

    # at edges of CO2 band
    Trad_h2o_nuL = get_Trad_h2o(nu0 - lk*np.log(q_co2*tau_co2_star),Ts,gammaLR,RH,params=params)

    # ---
    # CO2 feedback model that is based on a triangle/ditch.
    #   lambda_co2 = [ dB/dT(Tcold)*dTcold/dTs + dB/dT(Thot)*dThot/dTs ]*(nu_hot - nu_cold)
    #                + B(Thot)*( dnu_hot/dTs - dnu_cold/dTs )
    #                + B(Tcold)*( dnu_hot/dTs + dnu_cold/dTs )
    # Note the asymmetry in sign between B(Thot) and B(Tcold) terms. Due to the fact that emission inside stratosphere gets included in CO2 feedback term while
    # adding/removing emission outside CO2 band instead affects the surface/H2O feedback terms.

    # setup outputs ...
    lambda_co2 = np.zeros_like(Ts) + np.nan
    Thot = np.zeros_like(Ts) + np.nan
    Tcold = np.zeros_like(Ts) + np.nan

    # a) At cold temps assume: 
    #    - center of CO2 band in stratosphere, Tcold=Tstrat=const
    #    - Thot=Ts -> water vapor is completely transparent in CO2 band. -> dnuHot/dTs = 0

    mask = Ts <= Ts0
    
    Tcold = np.where(mask,np.zeros_like(Ts)+Tstrat,Tcold)
    Thot = np.where(mask,Ts,Thot)

    nuHot = np.zeros_like(Ts)                          # (needs correct array shape..)
    nuHot[:] = nu0 + lk *np.log( q_co2*tau_co2_star )
    nuCold = nu0 + lk *np.log( q_co2*tau_co2_star*(Tcold/Ts)**(2./gammaLR) )
    dnuCold_dTs = - 2.*lk/(Ts*gammaLR) + 2.*lk/(gammaLR**2)*np.log(Ts/Tcold)*dgammaLRdTs
    
    lambda_co2_cool = np.pi*Planck.dPlanckdT_n(nu0,Thot) *( nuHot - nuCold ) + \
                      ( np.pi*Planck.Planck_n(nu0,Thot) - np.pi*Planck.Planck_n(nu0,Tcold) ) *(-1) *dnuCold_dTs

    c = scaling_const
    lambda_co2[mask] = c*lambda_co2_cool[mask]
    
    # b) At high temps assume:
    #    - center of CO2 band in troposphere;
    #    - Thot=min[Th2o,Tcnt](nu0), dThot/dTs~0 (assume Simpsonian water vapor).

    mask = Ts > Ts0
    
    Tcold[mask] = Trad_co2_nu0[mask]
    Thot[mask] = np.minimum(Ts,np.minimum(Trad_h2o_nu0,Trad_cntm))[mask]

    nuHot = nu0 + lk *np.log( q_co2*tau_co2_star*(Thot/Ts)**(2./gammaLR) )
    nuCold = nu0

    dTcold_dTs = Tcold/Ts - 0.5 *Tcold *np.log(q_co2*tau_co2_star)*dgammaLRdTs

    lambda_co2_hot = np.pi*Planck.dPlanckdT_n(nu0,Tcold)*dTcold_dTs * (nuHot - nuCold)

    
    # OPTIONALLY: normalize lambda_co2 so there's no jump in lambda_CO2 when CO2 moves out of stratosphere?
    #             Ts0: surface temp at which CO2 moves out of stratosphere into troposphere
    #             Either just set this is a parameter or solve numerically using idealized band model:
    #                Trad_co2_nu0(Ts0,gammaLR(Ts0)) = Tstrat
    #                Tstrat - Tco2(Ts0,gammaLR(Ts0) = 0
    #             then evaluate lambda_co2_cool(Ts0), lambda_co2_hot(Ts0)
    #             -> b= lambda_co2_cool(Ts0) - lambda_co2_hot(Ts0)
    #
    b = np.zeros_like(Ts)
    
    if normalize_jump:
        #func = lambda ts: Tstrat - get_Trad_co2(nu0,ts,get_gammaLR(ts,Tstrat,params),ppv_co2,params=params)
        #Ts0 = scipy.optimize.newton(func,T0,tol=1e-3,maxiter=15)
        Ts0 = np.atleast_1d(Ts0)

        # ..
        gammaLR_at_Ts0 = get_gammaLR(Ts0,Tstrat,params)
        dTs = 1.
        dgammaLRdTs_at_Ts0 = (get_gammaLR(Ts0+dTs,Tstrat,params) - get_gammaLR(Ts0,Tstrat,params))/dTs

        # ..
        Thot0 = Ts0
        Tcold0 = np.zeros_like(Ts0)+Tstrat
        nuHot0 = np.zeros_like(Ts0)
        nuCold0 = np.zeros_like(Ts0)
        nuHot0[:] = nu0 + lk *np.log( q_co2*tau_co2_star )
        nuCold0[:] = nu0 + lk *np.log( q_co2*tau_co2_star*(Tcold0/Ts0)**(2./gammaLR_at_Ts0) )        
        dnuCold_dTs0 = - 2.*lk/(Ts0*gammaLR_at_Ts0) + 2.*lk/(gammaLR_at_Ts0**2)*np.log(Ts0/Tcold0)*dgammaLRdTs_at_Ts0

        lambda_cool_at_Ts0 = c*(
            np.pi*Planck.dPlanckdT_n(nu0,Thot0) *( nuHot0 - nuCold0 ) + \
            ( np.pi*Planck.Planck_n(nu0,Thot0) - np.pi*Planck.Planck_n(nu0,Tcold0) ) *(-1) *dnuCold_dTs0
            )

        # ..
        Tcold1 = get_Trad_co2(nu0,Ts0,gammaLR_at_Ts0,ppv_co2,params=params)

        Trad_h2o_nu0_at_Ts0 = get_Trad_h2o(nu0,Ts0,gammaLR_at_Ts0,RH,params=params)
        Trad_cntm_at_Ts0 = get_Trad_cntm(None,Ts0,gammaLR_at_Ts0,RH,params=params)
        Thot1 = np.minimum(Ts0,np.minimum(Trad_h2o_nu0_at_Ts0,Trad_cntm_at_Ts0))

        nuHot1 = np.zeros_like(Ts0)
        nuCold1 = np.zeros_like(Ts0)
        nuHot1[:] = nu0 + lk *np.log( q_co2*tau_co2_star*(Thot1/Ts0)**(2./gammaLR_at_Ts0) )
        nuCold1[:] = nu0

        dTcold_dTs1 = Tcold1/Ts0 - 0.5 *Tcold1 *np.log(q_co2*tau_co2_star)*dgammaLRdTs_at_Ts0

        lambda_hot_at_Ts0 = np.pi*Planck.dPlanckdT_n(nu0,Tcold1)*dTcold_dTs1 * (nuHot1 - nuCold1)

        # ..
        b[:] = lambda_cool_at_Ts0 - lambda_hot_at_Ts0
        print( "Normalization constant necessary to make lambda_co2 continuous at Tco2(nu0)=Tstrat: ", b )
    
    # ..
    lambda_co2[mask] = lambda_co2_hot[mask] + b[mask]

    return lambda_co2,Thot,Tcold

