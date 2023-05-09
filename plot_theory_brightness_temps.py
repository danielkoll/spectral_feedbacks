import numpy as np
import theory
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# ---
# Plot theoretical brightness temp as a function of wavenumber

# setup parameters ..
nu = np.linspace(1,1500,num=100)
Tstrat = 200.
gammaLR = 2./7.
pco2 = 400e-6 # here: want molar ratio, not mass ratio!
RH = 1.


# ---
# Make plot

norm = colors.Normalize(vmin=260, vmax=320)
my_cmap = cm.viridis

# ..
plt.figure()
for Ts in [260,280,300,320]:
    # compute Trad and plot directly..
    Trad = theory.Feedback_scalings.get_Trad_total(nu,Ts,Tstrat,gammaLR,RH,pco2)
    plt.plot(nu,Trad,label="T$_s$=%.fK" % Ts,color=my_cmap(norm(Ts)))
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.xlim(min(nu),max(nu))
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Emission temperature (K)")

# ..
plt.show()



