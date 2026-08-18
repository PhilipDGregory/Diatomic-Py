import numpy 
import matplotlib.pyplot as plt
from matplotlib import gridspec
import Hamiltonian
from numpy.linalg import eigh
from math import isnan
from time import time
from scipy import constants
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, LinearSegmentedColormap

h = constants.h

############################### JQC Colours ############################################
JQC = {'red'     :(198.0/255.0, 62.0/255.0, 98.0/255.0), \
       'blue'    :(0.0/255.0, 70.0/255.0, 127.0/255.0), \
       'purple'  :(126.0/255.0, 29.0/255.0, 123.0/255.0), \
       'sand'    :(244.0/255.0, 234.0/255.0, 168.0/255.0)}

colour_dict_twk_blue = {
    "red" : [(0.0,244.0/255.0,244.0/255.0),
            (0.33,124.0/255.0,124.0/255.0),
            (0.66,0.0,0.0),
            (1.0,0.0,0.0)] ,
    "green" : [(0.0,234.0/255.0,234.0/255.0),
            (0.33,154.0/255.0,154.0/255.0),
            (0.66,70.0/255.0,70.0/255.0),
            (1.0,32.0/255.0,32.0/255.0)]  ,
    "blue" : [(0.0,168.0/255.0,168.0/255.0),
            (0.33,148.0/255.0,148/255.0),
            (0.66,127.0/255.0,127.0/255.0),
            (1.0,58.0/255.0,58.0/255.0)] ,
    "alpha" : [(0.0, 0.0,0.0),
                #   (0.25,1.0, 1.0),
                   (0.5, 1.0, 1.0),
                #   (0.75,1.0, 1.0),
                   (1.0, 1.0, 1.0)]
}

RbCs_map_twk_blue = LinearSegmentedColormap("RbCs_map_tweak_blue",colour_dict_twk_blue)
plt.register_cmap(cmap=RbCs_map_twk_blue, name='RbCs_map_blue')

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0,legend=False):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = numpy.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = numpy.array([z])
        
    z = numpy.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc


def GetZeeman(Consts, species):
    start = time()
    print(species)
    print("Building Hamiltonian...")
    
    Nmax=2
    H0,Hz,HDC,HAC = Hamiltonian.Build_Hamiltonians(Nmax,Consts,zeeman=True,EDC=True,AC=True)
    
    I = 0
    E = 0
    B = numpy.linspace(1, 600, int(60))*1e-4 #Set magnetic field range here
    
    H = H0[..., None]+\
        Hz[..., None]*B+\
        HDC[..., None]*E+\
        HAC[..., None]*I 
    H = H.transpose(2,0,1)
    print("Done: ", numpy.round(time()-start, 1), "seconds")
    start = time()
    
    
    
    print("Diagonalizing Hamiltonian...")
    
    energies, states = eigh(H)
    
    print("Done: ", numpy.round(time()-start, 1), "seconds")
    start = time()
    
    
    print("Generating labels...")
    i = numpy.argsort(energies[-1,:])
    
    #Find state labels
    N, I1,I2 = Hamiltonian.Generate_vecs(Nmax,Consts['I1'],Consts['I2'])
    
    N2 = Hamiltonian.vector_dot(N,N)
    Nlabels = numpy.einsum('lik,ij,ljk->lk',numpy.conj(states),N2,states)
    Nlabels = numpy.round((-1+numpy.sqrt(1+4*1*Nlabels))/2).real
    
    F = N + I1 + I2
    Fz = F[2]
    MFlabels = numpy.round(numpy.einsum('lik,ij,ljk->lk',
                                        numpy.conj(states),Fz,states),1).real                            
                                        
    labels = numpy.empty((3,numpy.shape(energies)[1]))
    labels[:] = numpy.NaN
    
    for i in range(numpy.shape(energies)[1]):
        k = 0
        for j in range(numpy.shape(energies)[1]):
            if labels[0,j] == Nlabels[-1,i] and labels[1,j] == MFlabels[-1,i]:
                k+=1
        labels[:,i] = numpy.array([Nlabels[-1,i], MFlabels[-1,i], k])

                     
                                   
    
    print("Done: ", numpy.round(time()-start, 1), "seconds")
    start = time()
    
    
    
    
    
    
    
    print("Sorting energies and states by label...")
    
    energies_sorted = numpy.empty(numpy.shape(energies))
    energies_sorted[:] = numpy.NaN
    states_sorted = numpy.empty(numpy.shape(states), dtype="complex")
    states_sorted[:] = numpy.NaN
    
    
    for i in range(numpy.shape(energies)[0]):
        for j in range(numpy.shape(energies)[1]):
            N, MF = Nlabels[i,j], MFlabels[i,j]
            trkr = 0
            for k in range(numpy.shape(energies)[1]):
               if labels[0,k] == N and labels[1,k] == MF:
                   if isnan(energies_sorted[i,k]) == True:
                       if labels[2,k] == trkr:
                           energies_sorted[i,k] = energies[i,j]
                           states_sorted[i,:,k] = states[i,:,j] #Not sure about first two indices here!
                   else:
                       trkr += 1
    
    energies = energies_sorted
    states = states_sorted
    
    print("Done: ", numpy.round(time()-start, 1), "seconds")
    start = time()
    
    
    print("Saving files...")
    #print(states.dtype)
    #states = numpy.ndarray.astype(states, 'float16')
    
    numpy.save("B_"+species, B)
    numpy.save("labels_"+species, labels)
    numpy.save("energies_"+species, energies)
    numpy.save("states_"+species, states)
    
    print("Done: ", numpy.round(time()-start, 1), "seconds")
    start = time()




if False:
    GetZeeman(Hamiltonian.Rb87Cs133, "Rb87Cs133")
    GetZeeman(Hamiltonian.K40Rb87, "K40Rb87")
    GetZeeman(Hamiltonian.Na23K40, "Na23K40")
    GetZeeman(Hamiltonian.Na23Rb87, "Na23Rb87")
    
else:
    fig = plt.figure(figsize=(12,5))
    
    
    #Specify grid - No. of rows, No. of columns
    gs = gridspec.GridSpec(2,4,
                           width_ratios=[1,1,1,1],
                           height_ratios=[1,1]
                           )
    
    
    
    ###########################################################################################
    
    plt.subplot(gs[0,0])
    
    plt.title("(a) $^{87}$Rb$^{133}$Cs", fontsize=14)
    
    Spec = 'Rb87Cs133'
    B = numpy.load('B_'+Spec+'.npy')
    labels = numpy.load('labels_'+Spec+'.npy')
    energies = numpy.load('energies_'+Spec+'.npy')
    states = numpy.load('states_'+Spec+'.npy')
    
    
    N, I1,I2 = Hamiltonian.Generate_vecs(2,Hamiltonian.Rb87Cs133['I1'],Hamiltonian.Rb87Cs133['I2'])#change
    MI1 = numpy.real(numpy.diag(I1[2]))
    MI2 = numpy.real(numpy.diag(I2[2]))
    x = numpy.intersect1d(numpy.where(abs(MI1-1.5)<0.1), numpy.where(abs(MI2-3.5)<0.1))
    
    plt.plot(B*1e4, energies/(1e6*h), linestyle='solid', color=JQC['sand'], zorder=0)
    for i in range(numpy.shape(energies)[1]):  
        #Colour lines with MF = +5 red...
        for y in x:
            cl=colorline(B*1e4,energies[:,i]/(1e6*h),z=abs(numpy.real(states[:,y,i])),cmap='RbCs_map_blue',norm=LogNorm(vmin=1e-2,vmax=1),linewidth=2.0)
        
    plt.ylim(2*Hamiltonian.Rb87Cs133['Brot']*1e-6/h-1.5, 2*Hamiltonian.Rb87Cs133['Brot']*1e-6/h+1.0)
    #plt.xlabel("Magnetic Field (G)")
    #plt.ylabel("Energy / $h$ (MHz)")
    
    plt.xticks([])
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    plt.xlim(0,600)
    plt.yticks(fontsize="12")
    
    #plt.axvline(181.5, color='k', linestyle='dotted')
    
    
    plt.subplot(gs[1,0])
    
    B = numpy.load('B_'+Spec+'.npy')
    labels = numpy.load('labels_'+Spec+'.npy')
    energies = numpy.load('energies_'+Spec+'.npy')
    
    for i in range(numpy.shape(energies)[1]):
    
        #Colour lines with MF = +5 red...
        if labels[1,i] == 5: 
            clr = "k"
            z = 100
        
        #Set all other lines to be gray, and to be behind the red ones
        else: 
            clr = JQC['sand']
            z = 0
            
        plt.plot(B*1e4, energies[:,i]/(1e6*h), linestyle='solid', color=clr, zorder=z)
        
    plt.ylim(-1.5, 1.)
    plt.xlim(0,600)
    plt.xlabel("Magnetic Field (G)", fontsize=14)
    plt.ylabel("                                   Energy / $h$ (MHz)", fontsize=14)
    plt.yticks(fontsize="12")
    plt.xticks(fontsize="12")
    
    #plt.axvline(181.5, color='k', linestyle='dotted')
    ###########################################################################################
    
    
    
    plt.subplot(gs[0,3])
    
    plt.title("(d) $^{23}$Na$^{40}$K", fontsize=14)
    
    Spec = "Na23K40"
    B = numpy.load('B_'+Spec+'.npy')
    labels = numpy.load('labels_'+Spec+'.npy')
    energies = numpy.load('energies_'+Spec+'.npy')
    states = numpy.load('states_'+Spec+'.npy')
    
    N, I1,I2 = Hamiltonian.Generate_vecs(2,Hamiltonian.Na23K40['I1'],Hamiltonian.Na23K40['I2'])#change
    MI1 = numpy.real(numpy.diag(I1[2]))
    MI2 = numpy.real(numpy.diag(I2[2]))
    x = numpy.intersect1d(numpy.where(abs(MI1-1.5)<0.1), numpy.where(abs(MI2-4)<0.1))
    
    plt.plot(B*1e4, energies/(1e6*h), linestyle='solid', color=JQC['sand'], zorder=0)
    for i in range(numpy.shape(energies)[1]):  
        #Colour lines with MF = +5 red...
        for y in x:
            cl=colorline(B*1e4,energies[:,i]/(1e6*h),z=abs(numpy.real(states[:,y,i])),cmap='RbCs_map_blue',norm=LogNorm(vmin=1e-2,vmax=1),linewidth=2.0)
        
    plt.ylim(2*Hamiltonian.Na23K40['Brot']*1e-6/h-1.5, 2*Hamiltonian.Na23K40['Brot']*1e-6/h+1.)
    plt.xlim(0,600)
    #plt.xlabel("Magnetic Field (G)")
    #plt.ylabel("Energy / $h$ (MHz)")
    
    plt.xticks([])
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    plt.yticks(fontsize="12")
    
    #plt.axvline(216.6, color='k', linestyle='dotted')
    
    
    plt.subplot(gs[1,3])
    
    B = numpy.load('B_'+Spec+'.npy')
    labels = numpy.load('labels_'+Spec+'.npy')
    energies = numpy.load('energies_'+Spec+'.npy')
    
    for i in range(numpy.shape(energies)[1]):
        #Colour lines with MF = +5 red...
        if i == x[0]:
            clr = "k"
            z = 100
            
        
        #Set all other lines to be gray, and to be behind the red ones
        else: 
            clr = JQC['sand']
            z = 0
            
        plt.plot(B*1e4, energies[:,i]/(1e6*h), linestyle='solid', color=clr, zorder=z)
        
    plt.ylim(-1.5, 1.)
    plt.xlim(0,600)
    plt.xlabel("Magnetic Field (G)", fontsize=14)
    plt.yticks(fontsize="12")
    plt.xticks(fontsize="12")
    #plt.ylabel("Energy / $h$ (MHz)")
    
    #plt.axvline(216.6, color='k', linestyle='dotted')
    
    ###########################################################################################
    
    plt.subplot(gs[0,1])
    
    plt.title("(b) $^{40}$K$^{87}$Rb", fontsize=14)
    
    Spec = "K40Rb87"
    B = numpy.load('B_'+Spec+'.npy')
    labels = numpy.load('labels_'+Spec+'.npy')
    energies = numpy.load('energies_'+Spec+'.npy')
    states = numpy.load('states_'+Spec+'.npy')
    
    N, I1,I2 = Hamiltonian.Generate_vecs(2,Hamiltonian.K40Rb87['I1'],Hamiltonian.K40Rb87['I2'])#change
    MI1 = numpy.real(numpy.diag(I1[2]))
    MI2 = numpy.real(numpy.diag(I2[2]))
    x = numpy.intersect1d(numpy.where(abs(MI1-4)<0.1), numpy.where(abs(MI2-1.5)<0.1))
    
    plt.plot(B*1e4, energies/(1e6*h), linestyle='solid', color=JQC['sand'], zorder=0)
    for i in range(numpy.shape(energies)[1]):  
        #Colour lines with MF = +5 red...
        for y in x:
            cl=colorline(B*1e4,energies[:,i]/(1e6*h),z=abs(numpy.real(states[:,y,i])),cmap='RbCs_map_blue',norm=LogNorm(vmin=1e-2,vmax=1),linewidth=2.0)
        
    plt.ylim(2*Hamiltonian.K40Rb87['Brot']*1e-6/h-1.5, 2*Hamiltonian.K40Rb87['Brot']*1e-6/h+1.)
    plt.xlim(0,600)
    #plt.xlabel("Magnetic Field (G)")
    #plt.ylabel("Energy / $h$ (MHz)")
    
    plt.xticks([])
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    plt.yticks(fontsize="12")
    
    #plt.axvline(545.9, color='k', linestyle='dotted')
    
    
    plt.subplot(gs[1,1])
    
    B = numpy.load('B_'+Spec+'.npy')
    labels = numpy.load('labels_'+Spec+'.npy')
    energies = numpy.load('energies_'+Spec+'.npy')
    
    k = 0
    for i in range(numpy.shape(energies)[1]):
        #Colour lines with MF = +5 red...
        if i == x[0]: 
            clr = "k"
            z = 100
            k += 1
        
        #Set all other lines to be gray, and to be behind the red ones
        else: 
            clr = JQC['sand']
            z = 0
            
        plt.plot(B*1e4, energies[:,i]/(1e6*h), linestyle='solid', color=clr, zorder=z)
        
    plt.ylim(-1.5, 1.)
    plt.xlim(0,600)
    plt.xlabel("Magnetic Field (G)", fontsize=14)
    plt.yticks(fontsize="12")
    plt.xticks(fontsize="12")
    #plt.ylabel("Energy / $h$ (MHz)")
    
    #plt.axvline(545.9, color='k', linestyle='dotted')

    
    
    
    
    ###########################################################################################
    
    plt.subplot(gs[0,2])
    
    plt.title("(c) $^{23}$Na$^{87}$Rb", fontsize=14)
    
    Spec = "Na23Rb87"
    B = numpy.load('B_'+Spec+'.npy')
    labels = numpy.load('labels_'+Spec+'.npy')
    energies = numpy.load('energies_'+Spec+'.npy')
    states = numpy.load('states_'+Spec+'.npy')
    
    N, I1,I2 = Hamiltonian.Generate_vecs(2,Hamiltonian.Na23Rb87['I1'],Hamiltonian.Na23Rb87['I2'])#change
    MI1 = numpy.real(numpy.diag(I1[2]))
    MI2 = numpy.real(numpy.diag(I2[2]))
    x = numpy.intersect1d(numpy.where(abs(MI1-1.5)<0.1), numpy.where(abs(MI2-1.5)<0.1))
    
    plt.plot(B*1e4, energies/(1e6*h), linestyle='solid', color=JQC['sand'], zorder=0)
    for i in range(numpy.shape(energies)[1]):  
        #Colour lines with MF = +5 red...
        for y in x:
            cl=colorline(B*1e4,energies[:,i]/(1e6*h),z=abs(numpy.real(states[:,y,i])),cmap='RbCs_map_blue',norm=LogNorm(vmin=1e-2,vmax=1),linewidth=2.0)
        
    plt.ylim(2*Hamiltonian.Na23Rb87['Brot']*1e-6/h-1.5, 2*Hamiltonian.Na23Rb87['Brot']*1e-6/h+1.0)
    plt.xlim(0,600)
    #plt.xlabel("Magnetic Field (G)")
    #plt.ylabel("Energy / $h$ (MHz)")
    
    plt.xticks([])
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    plt.yticks(fontsize="12")
    
    #plt.axvline(335.2, color='k', linestyle='dotted')
    
    plt.subplot(gs[1,2])
    
    B = numpy.load('B_'+Spec+'.npy')
    labels = numpy.load('labels_'+Spec+'.npy')
    energies = numpy.load('energies_'+Spec+'.npy')
    
    for i in range(numpy.shape(energies)[1]):
        #Colour lines with MF = +5 red...
        if i == x[0]:
            clr = "k"
            z = 100
        
        
        #Set all other lines to be gray, and to be behind the red ones
        else: 
            clr = JQC['sand']
            z = 0
            
        plt.plot(B*1e4, energies[:,i]/(1e6*h), linestyle='solid', color=clr, zorder=z)
        
    plt.ylim(-1.5, 1.0)
    plt.xlim(0,600)
    plt.xlabel("Magnetic Field (G)", fontsize=14)
    plt.yticks(fontsize="12")
    plt.xticks(fontsize="12")
    
    #plt.axvline(335.2, color='k', linestyle='dotted')
    #plt.ylabel("Energy / $h$ (MHz)")


    cax = plt.axes([0.92, 0.11, 0.015, 0.83])
    cbar = plt.colorbar(cl, cax=cax)
    cbar.set_label('Relative Transition Strength', fontsize=14)
    
    

    
    
    
    
    
    plt.subplots_adjust(wspace = 0.4 , hspace =0.1,right =0.90,top=0.94,left=0.07,bottom=0.11)
    plt.savefig("Zeeman.pdf", dpi=600)
    plt.show()
