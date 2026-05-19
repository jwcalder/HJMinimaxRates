import numpy as np 
import graphlearning as gl 
import matplotlib.pyplot as plt

def function(X,Y,equation):
    #General function interface
    if equation == 'eikonal':
        return np.minimum(np.minimum(X,Y),np.minimum(1-X,1-Y))
    elif equation == 'L1':
        aX = np.abs(X - 0.45)
        aY = np.abs(Y - 0.45)
        norm = np.sqrt(aX**2 + aY**2)
        if isinstance(norm, np.ndarray):
            norm[norm < 1] = 1
        else:
            norm = max(norm,1)
        aX /= norm
        aY /= norm
        return (np.sqrt(2 - (aX - aY)**2) - (aX + aY))/2
    elif equation == 'x':
        aX = np.sqrt(X)
        aY = np.sqrt(Y)
        return 2*np.minimum(np.minimum(aX,aY),np.minimum(1-aX,1-aY))
    else:
        print('Invalid choice!')

def distance(X,equation):
    #GEneral distance interface
    if equation == 'eikonal':
        return np.linalg.norm(X,axis=1)
    elif equation == 'L1':
        return np.linalg.norm(X,axis=1,ord=np.inf)
    elif equation == 'x':
        Y = np.sqrt(X + [0.5,0.5]) - [np.sqrt(0.5),np.sqrt(0.5)]
        return 2*np.linalg.norm(Y,axis=1)
    else:
        print('Invalid choice!')

def estimator(x,y,uob,n,r,equation):
    #General estimator interface
    if equation == 'eikonal':
        return eikonal_estimator(x,y,uob,n,r)
    elif equation == 'L1':
        return L1_estimator(x,y,uob,n,r)
    elif equation == 'x':
        return x_estimator(x,y,uob,n,r)
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

def x_estimator(x,y,uob,n,r):

    #Estimator for the spatially dependent equation
    x = np.sqrt(x + 0.5) - np.sqrt(0.5)
    y = np.sqrt(y + 0.5) - np.sqrt(0.5)
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


def L1_estimator(x,y,uob,n,r):

    #Estimator for the L1 equation
    eps = (1/n)**(2/5)
    t = (1/2)*(1/n)**(1/3)
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





