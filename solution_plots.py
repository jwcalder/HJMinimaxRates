import numpy as np 
import graphlearning as gl 
import matplotlib.pyplot as plt
from estimators import function
import plots


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

x = np.linspace(-1,1,300)
y = np.linspace(-1,1,300)
X, Y = np.meshgrid(x, y)
Z = function(X+0.5,Y+0.5,'L1')
plt.figure()
h = 0.05
contours = plt.contour(X, Y, Z, levels=np.arange(h,1/np.sqrt(2),h), colors="black", linewidths=1)
plt.clabel(contours, inline=True, fontsize=8)
plt.axis('square')
plots.savefig('figures/L1.pdf',axis=True)

plt.show()
