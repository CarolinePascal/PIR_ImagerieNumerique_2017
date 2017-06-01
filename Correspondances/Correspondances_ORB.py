import numpy as np
import math
import cmath 
import scipy.misc as spm
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('carre 2.png',1) 
img2 = cv2.imread('carre 22.png',1) 

def aleaGauss(sigma):   #Simulation d'une VAR Gaussienne centrée sur 0.
    U1 = np.random.random()
    U2 = np.random.random()
    k = sigma*math.sqrt(-2*math.log(U1))*math.cos(2*math.pi*U2)
    return(k)
    
def bruitgimage(I,sigma):
    m,n,k=np.shape(I)
    for i in range(m):
        for j in range(n):
            for z in range(k):
                I[i,j,z]+=aleaGauss(sigma)
    return(I)
    
img2=bruitgimage(img2,0.5)
# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)
img3=img2
A=[]
for m in matches:
    x=kp1[m.queryIdx].pt
    y=kp2[m.trainIdx].pt
    A.append(y)
# Draw first 10 matches.
img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[0:10],img3,flags=2)

plt.imshow(img3),plt.show()

def rot(z,a,alpha):
    return (z-a)*cmath.rect(1,alpha)+a
G=[]

for m in matches:
    x=kp1[m.queryIdx].pt
    x1=x[0]
    y1=x[1]
    m=complex(x1,y1)
    a=complex(125,125)
    m1=rot(m,a,math.pi/2)
    m1x=m1.real
    m1y=m1.imag
    m2x=250-m1x
    m2y=250-m1y
    G.append([m2x,m2y])
    
c=0
for i in range(len(G)):
    a=(G[i][0]-A[i][0])**2
    b=(G[i][1]-A[i][1])**2
    if a<4 and b<4:
        c+=1

k2=float(c)/float(len(G))
print(k2)