import numpy as np 
import graphlearning as gl 
import matplotlib.pyplot as plt

def function(X,Y,equation):
    #General function interface
    if equation == 'eikonal':
        return np.minimum(np.minimum(X,Y),np.minimum(1-X,1-Y))
    else:
        print('Invalid choice!')

def estimator(x,y,uob,n,r,equation):
    #General estimator interface
    if equation == 'eikonal':
        return eikonal_estimator(x,y,uob,n,r)
    else:
        print('Invalid choice!')

def eikonal_estimator(x,y,uob,n,r):
    #Estimator for the eikonal equation
    eps = (1/n)**(2/5)
    t = (1/10)*(1/n)**(1/5)
    M = int(10/eps)
    theta = np.arange(0,1,1/M)*2*np.pi

    u_best = np.inf
    for th in theta:
        px = np.cos(th)
        py = np.sin(th)

        dot = px*x + py*y
        zx = x - px*dot
        zy = y - py*dot
        ind = (zx**2 + zy**2 < t**2) & (dot <= -r/2)
        if np.sum(ind)==0:
            print('Empty!')

        uest = np.mean(uob[ind])
        if uest < u_best:
            u_best = uest

    return u_best




