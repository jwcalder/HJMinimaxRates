import numpy as np 
import graphlearning as gl 
import matplotlib.pyplot as plt
from estimators import function, estimator
import plots
import os

#Parameters
equation = 'eikonal' #Equation to test
r = 0.3 #Radius
T = 1000 #Number of trials

#Wrap function more easily
def f(X,Y):
    return function(X,Y,equation)

for sigma in [0,0.05]: #For noiseless and noisy

    fname = 'data/'+equation+'_sigma_%.2f_T_%d_r_%.2f.npz'%(sigma,T,r)

    N = [6400,12800,25600,51200,102400]

    if os.path.exists(fname):

        #Load experiment from file
        M = np.load(fname)
        all_umin_errors = M['umin']
        all_uavg_errors = M['uavg']
        all_uest_errors = M['uest'] 

    else:

        #Run experiment
        all_umin_errors = []
        all_uavg_errors = []
        all_uest_errors = []
        for t in range(T):
            umin_errors = []
            uavg_errors = []
            uest_errors = []
            for n in N:
                X = np.random.rand(n,2)
                u = f(X[:,0],X[:,1])
                u += sigma*np.random.randn(n)
                d = np.linalg.norm(X - [0.5,0.5],axis=1)
                uval = (u+d)[d<r]
                umin = np.min(uval)
                if sigma == 0:
                    uavg = np.mean(u[d<(np.log(n)/n)**(1/2)])
                else:
                    uavg = np.mean(u[d<(1/2)*(1/n)**(1/4)])
                uest = estimator(X[d<r,0]-0.5,X[d<r,1]-0.5,uval,n,r,equation)
                umin_errors += [abs(umin-f(0.5,0.5))]
                uavg_errors += [abs(uavg-f(0.5,0.5))]
                uest_errors += [abs(uest-f(0.5,0.5))]

            all_umin_errors += [umin_errors]
            all_uavg_errors += [uavg_errors]
            all_uest_errors += [uest_errors]

        all_umin_errors = np.array(all_umin_errors)
        all_uavg_errors = np.array(all_uavg_errors)
        all_uest_errors = np.array(all_uest_errors)

        np.savez_compressed(fname,umin=all_umin_errors,uavg=all_uavg_errors,uest=all_uest_errors)

    all_umin_errors = np.mean(all_umin_errors,axis=0)
    all_uavg_errors = np.mean(all_uavg_errors,axis=0)
    all_uest_errors = np.mean(all_uest_errors,axis=0)

    plt.figure()
    p,_=np.polyfit(np.log(N),np.log(all_uavg_errors),1)
    plt.loglog(N,all_uavg_errors,'s-',label=r'Lipschitz Estimator: $\mathcal{O}(n^{%.2f})$'%p)
    if sigma == 0:
        p,_=np.polyfit(np.log(N),np.log(all_umin_errors),1)
        plt.loglog(N,all_umin_errors,'o-',label=r'Minimax Estimator: $\mathcal{O}(n^{%.2f})$'%p)
    else:
        p,_=np.polyfit(np.log(N),np.log(all_uest_errors),1)
        plt.loglog(N,all_uest_errors,'o-',label=r'Minimax Estimator: $\mathcal{O}(n^{%.2f})$'%p)
    plt.legend()
    plt.xlabel('Number of samples $n$')
    plt.ylabel('Absolute error')
    plots.savefig('figures/'+equation+'_errors_sigma_%.2f.pdf'%sigma,axis=True,grid=True)
    plt.close()


