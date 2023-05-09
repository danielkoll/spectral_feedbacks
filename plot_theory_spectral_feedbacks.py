import numpy as np
import theory
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# ---
# Plot theoretical spectral feedbacks as a function of Ts

# setup parameters ..
Tstrat = 200.
ppv_co2 = 400e-6 # here: want molar ratio, not mass ratio!
RH = 1.

params = theory.Feedback_scalings.default_params

# evaluate theory eqns
#   - if Ts is a 1d array, then outputs (lambda_i,...) will also be 1d arrays
#   - gammaLR, dgammaLR/dTs not specified, so computed using approximate moist adiabat
Ts_array = np.arange(250,320.1,1.)
lambda_surf = theory.Feedback_scalings.get_lambda_surf(Ts_array,Tstrat,RH,params,ppv_co2)
lambda_co2,Thot,Tcold = theory.Feedback_scalings.get_lambda_co2(Ts_array,Tstrat,RH,params,ppv_co2)
lambda_h2o,Th2o = theory.Feedback_scalings.get_lambda_h2o(Ts_array,Tstrat,RH,params,ppv_co2)
lambda_cntm,Tcntm = theory.Feedback_scalings.get_lambda_cntm(Ts_array,Tstrat,RH,params,ppv_co2)



# ---
# Make plot

norm = colors.Normalize(vmin=260, vmax=320)
my_cmap = cm.viridis

# ..
plt.figure()

plt.plot(Ts_array,Ts_array*0,"-",color="0.5")
p0,=plt.plot(Ts_array,lambda_surf+lambda_co2+lambda_h2o+lambda_cntm,"k-")
p1,=plt.plot(Ts_array,lambda_surf,"b-")
p2,=plt.plot(Ts_array,lambda_co2,"r-")
p3,=plt.plot(Ts_array,lambda_h2o,"c-")
p4,=plt.plot(Ts_array,lambda_cntm,"m-")
    
plt.xlabel("Surface temperature (K)")
plt.ylabel("Feedback (W/m$^2$/K)")
plt.xlim(Ts_array.min(),Ts_array.max())
plt.legend([p0,p1,p2,p3,p4],["$\\lambda_{net}$","$\\lambda_{surf}$","$\\lambda_{co2}$","$\\lambda_{h2o}$","$\\lambda_{cnt}$"],loc="best")

# ..
plt.show()



