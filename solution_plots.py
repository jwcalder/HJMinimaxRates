import numpy as np 
import graphlearning as gl 
import matplotlib.pyplot as plt
from estimators import function, distance
from matplotlib import cm
import plots

def surface_plot(X,Y,Z):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},constrained_layout=True)
    surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm,linewidth=0,antialiased=True,rcount=300,ccount=300,shade=True)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


#eikonal
x = np.linspace(0,1,300)
y = np.linspace(0,1,300)
X, Y = np.meshgrid(x, y)
Z = function(X,Y,'eikonal')
plt.figure()
h = 0.04
contours = plt.contour(X, Y, Z, levels=np.arange(h,0.5,h), colors="black", linewidths=1)
plt.clabel(contours, inline=True, fontsize=8)
plt.axis('square')
plots.savefig('figures/eikonal.pdf',axis=True)

Z += np.sqrt((X-0.5)**2 + (Y-0.5)**2)
surface_plot(X,Y,Z)
plots.savefig('figures/eikonal_dpp.pdf',axis=True)
plots.savefig('figures/eikonal_dpp.png',axis=True)

#L1
Z = function(X,Y,'L1')
plt.figure()
h = 0.05
contours = plt.contour(X, Y, Z, levels=np.arange(h,1/np.sqrt(2),h), colors="black", linewidths=1)
plt.clabel(contours, inline=True, fontsize=8)
plt.axis('square')
plots.savefig('figures/L1.pdf',axis=True)

Z += np.maximum(np.abs(X-0.5),np.abs(Y-0.5))
surface_plot(X,Y,Z)
plots.savefig('figures/L1_dpp.pdf',axis=True)
plots.savefig('figures/L1_dpp.png',axis=True)

#Spatially varying
Z = function(X,Y,'x')
plt.figure()
h = 0.1
contours = plt.contour(X, Y, Z, levels=np.arange(h,2,h), colors="black", linewidths=1)
plt.clabel(contours, inline=True, fontsize=8)
plt.axis('square')
plots.savefig('figures/x.pdf',axis=True)

Z += 2*np.sqrt((np.sqrt(X)-np.sqrt(0.5))**2 + (np.sqrt(Y)-np.sqrt(0.5))**2)
surface_plot(X,Y,Z)
plots.savefig('figures/x_dpp.pdf',axis=True)
plots.savefig('figures/x_dpp.png',axis=True)

plt.show()
