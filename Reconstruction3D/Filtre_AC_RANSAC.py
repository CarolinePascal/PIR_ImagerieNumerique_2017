import numpy as np

###filtre_AC_RANSAC
    
"""
Inputs

- Corresp : Matrice de taille 6xN contenant dans chaque colonne les 3 projections du même point dans les 3 images.
- CalM : Matrice de taille 9x3 contant les 3 matrices 3x3 (concaténées) de calibration des 3 caméras.
- imsize : Vecteur indiquant la taille de l'image.
- NFA_th : Niveau pour le NFA pour établie la validité d'un modèle (Par défaut, 1).
- max_it : Nombre maximum d'itérations pour le RANSAC (par défaut, 100).

                                                      
Outputs

- inliers : Indices des points retenus (inliers).
- Sol : Sol = {Rot,Trans,Reconst,R,T} => Orientation finale en considérant les inliers. Output du même genre que celui de la fonction OrthographicPoseEstimation.
- ransac_th : Niveau utilisé par le RANSAC, donné par la méthode AC et utilisé pour le modèle final et le calcul des iliners.

"""

def filtre_AC_RANSAC(Corresp,CalM,imsize,NFA_th=1,max_it=100):
    
    N=np.shape(Corresp)[1]
    n_sample=4 #Nombre minimal de correspondances de points pour l'estimation de pose
    d=2 
    n_out=2 #Nombre possible d'orientations
    a=np.pi/(imsize[0]*imsize[1]) #Probabilité d'avoir une erreur (Pour une hypthèse nulle, 1)
    
    k_inliers=n_sample+1 #Nombre maximum d'inliers trouvés
    inliers=[] #Liste des inliers
    ransac_th=float('inf') #Seuil RANSAC
    
    it=0
    max_old=max_it
    
    while it<max_it:
        it=it+1
        print(it)
        
        #Echantillon choisi au hasard parmi les points correspondants
        sample=np.random.choice(np.arange(N),n_sample,replace=False) #On constuit un vecteur de taille n_sample à partir de valeur comprises entre 1 et N
        #Calcul de l'orientation avec le modèle orthographique à partir de cet échantillon (si la fonction echoue, on commence une nouvelle itération)
        try:
            [Sol1,Sol2]=Triangulation_Orthographique(np.take(Corresp,sample,axis=1),CalM)
        except:
            if max_it<2*max_old:
                max_it=max_it+1
            continue
        #Calcul de l'erreur résiduelle pour les deux solutions et choix de celle avec l'erreur la plus faible
        R=Sol1[3]
        T=Sol1[4]
        err_min=float('inf')
        Sol_it=Sol1
        
        for j in range(2):
            ind=[[0,1,2,3],[4,5],[0,1,4,5],[2,3],[2,3,4,5],[0,1]]
            err=np.zeros(N)
            for k in range(3):
                #Reconstruction 3D à partir d'une paire de vues
                p3D=np.dot(np.linalg.pinv(np.take(R,ind[2*k],axis=0)),np.take(Corresp,ind[2*k],axis=0)-np.tile(np.take(T,ind[2*k],axis=0),(1,N)))
                #Reprojection de l'autre vue et erreur
                error=np.sqrt(np.sum(np.power(np.dot(np.take(R,ind[2*k+1],axis=0),p3D)+np.tile(np.take(T,ind[2*k+1],axis=0),(1,N))-np.take(Corresp,ind[2*k+1],axis=0),2),0))
                #On prend le max de l'erreur
                for i in range(len(err)):
                    err[i]=max(err[i],error[i])
            if sum(err)<err_min:
                vec_errors=err
                err_min=sum(err)
                if j==1:
                    Sol_it=Sol2
            R=Sol2[3]
            T=Sol2[4]

        #Autres points que ceux d l'échantillon utilisé
        tab=[i for i in range(N)]
        nosample=np.setdiff1d(tab,sample)
        #Tri de la liste des erreurs
        v=np.take(vec_errors,nosample,axis=0)
        ind_sorted=np.argsort(v)
        
        #Recherche du minimum de NFA(model,k)
        NFA_min=NFA_th
        k_min=0
        err_threshold=float('inf')
        factor=n_out*np.arange(N-n_sample,N+1).prod()/np.math.factorial(n_sample)
        for k in range(n_sample,N):
            factor=factor*((N-k)/(k-n_sample+1))*a
            NFA=factor*(vec_errors[nosample[ind_sorted[k-n_sample]]])**(d*(k-n_sample+1))
            if NFA<=NFA_min:
                NFA_min=NFA
                k_min=k+1
                err_threshold=vec_errors[nosample[ind_sorted[k-n_sample]]]
                
        #Si le modèle trouvé a plus d'inliers ou le même nombre avec moins d'erreur que le précédent, on le garde
        if ((k_min>k_inliers) or (k_min==k_inliers and err_threshold<ransac_th)):
            k_inliers=k_min
            tab=[]
            for k in range(k_inliers-n_sample):
                tab.append(k)
            A=np.take(nosample,np.take(ind_sorted,tab,axis=0),axis=0)
            A.shape=(1,np.size(A))
            inliers=np.concatenate((np.reshape(sample,(1,-1)),A),axis=1)[0]
            ransac_th=err_threshold
            Sol=Sol_it
            inliers.sort()
    return(inliers,Sol,ransac_th)

        
    
        
        
            
    
