import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

### Plot de la reconstruction 3D

def Affichage(Corresp,CalM,imsize,NFA_th=1,max_it=100):
    
    M=np.shape(Corresp)[0]//2
    N=np.shape(Corresp)[1]
    
    #On effectue le filtrage sur 3 images
    inliers, Sol, ransac_th = AC_RANSAC_filtre(np.take(Corresp,[0,1,2,3,4,5],axis=0),np.take(CalM,[0,1,2,3,4,5,6,7,8],axis=0),imsize)
    outliers=[i for i in range(N)]
    for i in inliers:
        outliers.remove(i)
    
    #Calcul de la triangulation
    [Sol1,Sol2] = Triangulation_Ortho(np.take(Corresp,inliers,axis=1),CalM)
    
    #Solution 1
    Sol=-Sol1[2]
    fig1=plt.figure()
    ax1=fig1.add_subplot(111,projection='3d')
    PointCloud=-Sol
    ax1.scatter(PointCloud[0,:],PointCloud[1,:],PointCloud[2,:],s=15,c='k')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    #Solution 2
    Sol=-Sol2[2]
    fig2=plt.figure()
    ax2=fig2.add_subplot(111,projection='3d')
    PointCloud=-Sol
    ax2.scatter(PointCloud[0,:],PointCloud[1,:],PointCloud[2,:],s=15,c='k')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    return(Sol1,Sol2)

    
        
        
            
    
