
import amitgroup as ag
import numpy as np
from time import time

def main():
    F, I = ag.io.load_example('two-faces')

    t1 = time()
    imdef, info = ag.stats.image_deformation(F, I, penalty=0.1, rho=2.0, last_level=3, tol=0.00001, maxiter=100, \
                                          start_level=1, wavelet='db8', debug_plot=False)
    t2 = time()
    Fdef = imdef.deform(F)
    
    print "Time:", t2-t1
    print "Cost:", info['cost']

    x, y = imdef.meshgrid() 
    Ux, Uy = imdef.deform_map(x, y)
    for i in range(Ux.shape[0]):
        for j in range(Ux.shape[1]):
            print Ux[i,j], Uy[i,j]

    print 'mins', Ux.min(), Uy.min()
    print 'maxs', Ux.max(), Uy.max()

    import matplotlib.pylab as plt
    plt.quiver(x, y, Ux, Uy)#, Ux, Uy)
    plt.show()

    PLOT = False 
    if PLOT:
        import matplotlib.pylab as plt
        ag.plot.deformation(F, I, imdef)

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('main()')
    main()
