import numpy as np
import scipy.linalg as sl
    
###Reconstrucion 3D

"""
Inputs

- Corresp : Matrice de taille 2MxN contenant dans chaque colonne les M projections du même point dans les M images.
- CalM : Matrice de taille 3Mx3 contant les M matrices 3x3 (concaténées) de calibration des 3 caméras.

Outputs

-> Les deux solutions possibles Sol1 et Sol2 sous le format d'un tableau 1x5 Sol=[Rot,Trans,Reconst,R,T} où :

- Rot : Matrice de taille 3Mx3 contenant les M matrices 3x3 de rotation concaténées pour chaque caméra. La première sera toujours la matrice identité.
- Trans : Vecteur de taille 3Mx1 contenant les M vecteurs de translation concaténés pour chaque caméra.
- Reconst : Matrice de taille 3xN contenant la reconstitution 3D des correspondances.
- R,T : Matrice de mouvement de taille 2Mx3 et matrice de translation de taille 2Mx1 dans le cadre du modèle orthographique.
                                                      
"""

def Triangulation_Orthographique(Corresp,CalM):
    N=len(Corresp[0])
    M=len(Corresp)//2
    if N<4 or M<3:
         print("Pas assez de caméras ou points")
         return(None)

    """Longueurs focales et points principaux"""
    
    focalL=np.take(np.take(CalM,[3*k for k in range(M)],axis=0),[0],axis=1)
    ppalP=np.take(np.take(CalM,np.setdiff1d([i for i in range(3*M)],[3*k+2 for k in range(M)]),axis=0),[2],axis=1)

    """Centrage des points de l'image en soustrayant les points principaux et calcul de la moyenne"""
    
    W=Corresp-np.tile(ppalP,(1,N))
    T=np.mean(W,axis=1)
    T.shape=(np.size(T),1)
    
    """Soustraction de la moyenne pour obtenir W*"""
    
    W_star=W-np.tile(T,(1,N))
    
    """Calcul de la décomposition en valeurs singulières de W*"""
    
    U,d,Vh=np.linalg.svd(W_star,False)
    D=np.diag(d)
    V=np.transpose(Vh)

    """Imposer la déficience du rang et calcul de la factorisation en rang"""
    
    R_aux=np.dot(np.take(U,[0,1,2],axis=1),sl.sqrtm(np.take(np.take(D,[0,1,2],axis=0),[0,1,2],axis=1)))
    S_aux=np.dot(sl.sqrtm(np.take(np.take(D,[0,1,2],axis=0),[0,1,2],axis=1)),np.transpose(np.take(V,[0,1,2],axis=1)))
    
    """Pour trouver QQ' tq R=R_aux*Q et S=Q*S_aux on résout les système linéaire homogène M*coef(QQ')=0"""    
    
    SystM=np.zeros((2*M,6))
    for i in range(M):
        m=R_aux[2*i,:]
        n=R_aux[2*i+1,:]
        
        #Contrainte en norme
        A=2*(np.dot(np.reshape(m,(-1,1)),np.reshape(m,(1,-1))) - np.dot(np.reshape(n,(-1,1)),np.reshape(n,(1,-1))))
        SystM[2*i,:]=[0.5*A[0,0],0.5*A[1,1],0.5*A[2,2],A[1,0],A[2,0],A[2,1]]
        
        #Contraintes de perpendicularité
        A=np.dot(np.reshape(m,(-1,1)),np.reshape(n,(1,-1))) + np.dot(np.reshape(n,(-1,1)),np.reshape(m,(1,-1)))
        SystM[2*i+1,:]=[0.5*A[0,0],0.5*A[1,1],0.5*A[2,2],A[1,0],A[2,0],A[2,1]]
        
    """Résolution du système"""
    
    
    v1,v2,vh=np.linalg.svd(SystM,False)
    v=np.transpose(vh)
    coefQQ=v[:,-1] * np.sign(v[0,-1]) 
    QQ=np.array([[coefQQ[0],coefQQ[3],coefQQ[4]] ,[coefQQ[3],coefQQ[1],coefQQ[5]],[coefQQ[4],coefQQ[5],coefQQ[2]]])
    Q=np.transpose(np.linalg.cholesky(QQ))
    Q=np.transpose(Q)
     
    R=np.dot(R_aux,Q) 
    S=np.linalg.lstsq(Q,S_aux)
    S=S[0]
    
    """Paramètres de rotation"""
                   
    norms=np.sqrt(np.sum(np.power(R,2),1))
    norms.shape=(np.size(norms),1)
    Rot=np.zeros((3*M,3))
    tab1=np.setdiff1d([i for i in range(3*M)],[3*k+2 for k in range(M)])
    n=0
    for i in tab1:
        Rot[i,:]=np.divide(R,np.tile(norms,(1,3)))[n,:] 
        n+=1
    tab2=[2+3*i for i in range(M)]
    tab3=[1+3*i for i in range(M)]
    tab4=[3*i for i in range(M)]
    n=0
    for i in tab2:
        Rot[i,:]=np.cross(np.take(Rot,tab4,axis=0),np.take(Rot,tab3,axis=0),1)[n,:]
        n+=1
    
    """Paramètres de translation"""
    
    Trans=np.zeros((3*M,1))
    Trans[np.setdiff1d(tab1,tab2)]=T
    Trans[tab2]=focalL  
    s=np.reshape(norms,(M,2))
    s=np.transpose(s)
    s=np.sum(s,0)/2
    s=np.tile(s,(3,1))
    s=np.transpose(s)
    s=np.reshape(s,(-1,1))
    
    Trans=np.divide(Trans,s)

    """Seonde solution -> Ambiguité de profondeur"""
    
    a= np.array((1,1,-1))
    A_=np.diag(np.tile(a,(1,M))[0])
    Rot2=np.dot(np.dot(A_,Rot),np.diag(a))
    S2=np.dot(np.diag(a),S)
    R2=np.dot(R,np.diag(a))
    T2=T 
    Trans2=Trans
    Reconstr=S
    Reconstr2=S2
    
    """Post-traitement pour la reconstruction de l'échantillon"""
    
    """Translation et rotation pour ramener le centre de la première caméra à l'origine"""
    """
    Reconstr=np.dot(np.take(Rot,[0,1,2],axis=0),Reconstr) + np.tile(np.take(Trans,[0,1,2],axis=0),(1,N))
    Reconstr2=np.dot(np.take(Rot2,[0,1,2],axis=0),Reconstr2) + np.tile(np.take(Trans2,[0,1,2],axis=0),(1,N)) 
    R=((np.linalg.lstsq(np.transpose(np.take(Rot,[0,1,2],axis=0)),np.transpose(R))))
    R2=((np.linalg.lstsq(np.transpose(np.take(Rot2,[0,1,2],axis=0)),np.transpose(R2))))
    R=R[0]
    R2=R2[0]
    R=np.transpose(R)
    R2=np.transpose(R2)
    T=T-np.dot(R,np.take(Trans,[0,1,2],axis=0)) 
    T2=T2-np.dot(R2,np.take(Trans2,[0,1,2],axis=0))   
    
    Rot=np.transpose(np.linalg.lstsq(np.transpose(np.take(Rot,[0,1,2],axis=0)),np.transpose(Rot)))
    Rot2=np.transpose((np.linalg.lstsq(np.transpose(np.take(Rot2,[0,1,2],axis=0)),np.transpose(Rot2))))
    Rot=Rot[0]
    Rot2=Rot2[0]
    Rot=np.transpose(Rot)
    Rot2=np.transpose(Rot2)
    Trans=Trans-np.dot(Rot,np.take(Trans,[0,1,2],axis=0))
    Trans2=Trans2-np.dot(Rot2,np.take(Trans2,[0,1,2],axis=0))
    """
    """Mise à l'échelle de sorte que la distance de la première caméra à la deuxième est unitaire"""
    """
    alpha =1/np.linalg.norm(np.take(Trans,[3,4,5],axis=0))
    alpha2=1/np.linalg.norm(np.take(Trans2,[3,4,5],axis=0))
    Trans=alpha*Trans
    Trans2=alpha2*Trans2
    R =(1/alpha)*R
    R2=(1/alpha2)*R2
    Reconstr =alpha*Reconstr
    Reconstr2=alpha2*Reconstr2
    """
    
    Sol1=(Rot ,Trans ,Reconstr ,R ,T+ppalP)
    Sol2=(Rot2,Trans2,Reconstr2,R2,T2+ppalP)
    
    return [Sol1,Sol2]
