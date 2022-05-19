import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import random
import math


@jit
def laplacian(ix, iy, s):#ラプラシアンを求める
    ts = 0.0
    S=int(np.sqrt(s.size)-1)
    if ix==0 and iy==0:#左上
        ts += (s[S,0]+s[ix+1, iy]+s[0,S]+s[ix,iy+1]-4*s[ix, iy])##ix-1,iy-1
    if ix==0 and iy==S:#右上
        ts += (s[S,S]+s[ix+1,iy]+s[0,0]+s[ix,iy+1]-4*s[ix, iy])##ix-1,iy+1
    if ix==S and iy==0:#左下
        ts += (s[ix-1,iy]+s[0,0]+s[S,S]+s[ix,iy+1]-4*s[ix, iy])##ix+1,iy-1
    if ix==S and iy==S:#右下
        ts += (s[ix-1, iy]+s[0,S]+s[ix,iy-1]+s[S,0]-4*s[ix, iy])##ix+1,iy+1
    if ix==0 and iy!=0 and iy!=S:#上
        ts += (s[S,iy]+s[ix+1, iy]+s[ix, iy-1]+s[ix,iy+1]-4*s[ix, iy])##ix-1
    if iy==0 and ix!=0 and ix!=S:#左
        ts += (s[ix-1,iy]+s[ix+1, iy]+s[ix,S]+s[ix, iy+1]-4*s[ix, iy])##iy-1
    if ix==S and iy!=0 and iy!=S:#下
        ts += (s[ix-1,iy]+s[0,iy]+s[ix,iy-1]+s[ix, iy+1]-4*s[ix, iy])##ix+1
    if iy==S and ix!=0 and ix!=S:#右
        ts += (s[ix-1, iy]+s[ix+1, iy]+s[ix, iy-1]+s[ix, 0]-4*s[ix, iy])##iy+1
    if ix!=0 and ix!=S and iy!=0 and iy!=S:
        ts += (s[ix-1, iy]+s[ix+1, iy]+s[ix, iy-1]+s[ix, iy+1]-4*s[ix, iy])
    return ts


@jit
def calc(a, h, a2, h2):
    (L,L) = a.shape
    dt=0.1
    Dh=0.5
    ca=0.08
    ch=0.11
    da=0.08
    #dh=0
    mua=0.03
    muh=0.12
    #aとhの密度が0.1になるように設定
    #roa=0.003
    #roh=0.001
    roa=(da+mua-ca)/10
    roh=(muh-ch)/10
    #roa=mua/10
    #roh=muh/10
    fa=ca-mua
    fh=-da
    ga=ch
    gh=-muh
    Da=0.02
    #Da=-Dh*fa/gh
    #Da=(Dh*(fa*gh-2*fh*ga)-2*Dh*np.sqrt(fh*ga*fh*ga-fh*ga*fa*gh))/(gh*gh)
    mina=0
    minh=0
    maxa=1
    maxh=1
    la = np.zeros((L, L))
    lh = np.zeros((L, L))
    
    for ix in range(L):
        for iy in range(L):
            la[ix, iy] = Da * laplacian(ix, iy, a)##拡散項
            lh[ix, iy] = Dh * laplacian(ix, iy, h)       
    sa = (ca*a)-(da*h)+roa-mua*a ##反応項
    sh = (ch*a)+roh-muh*h  
            
    for i in range(L):
        for j in range(L):
            a2[i,j] = (a[i,j]+(la[i,j]+sa[i,j]) * dt) #-mua*a[i,j]
            h2[i,j] = (h[i,j]+(lh[i,j]+sh[i,j]) * dt) # -muh*h[i,j]
            if a2[i,j]<mina:
                a2[i,j]=mina
            if h2[i,j]<minh:
                h2[i,j]=minh
            if a2[i,j]>maxa:
                a2[i,j]=maxa
            if h2[i,j]>maxh:
                h2[i,j]=maxh
                

@jit
def main():
    L = 100 #128*128ビット
    u = np.zeros((L, L)) 
    u2 = np.zeros((L, L)) 
    v = np.zeros((L, L))+0.1
    v2 = np.zeros((L, L))
    h =L//2##正方形を作成
    u[h-2:h+2, h-2:h+2]=0.1
    #u[h,h]=0.5
    #u[h,h]=u[h,h-1]=u[h,h+1]=u[h-1,h]=u[h+1,h]=1
    #for i in range(1,L-1):#初期状態をランダムに設定
        #for j in range(1,L-1):
            #u[i,j] = random.uniform(0,0.1)
            #v[i,j] = random.uniform(0,0.1)
    time=100000
    for i in range(time):
        if i % 2 == 0:
            calc(u, v, u2, v2)#現在のステップの状態u,vから次のステップの状態u2,v2を計算する
        else:
            calc(u2, v2, u, v)#現在のステップの状態u2,v2から次のステップの状態u,vを計算する
            
    for j in range(L):
        for k in range(L):
            u[j,k]=round(u[j,k],2)
            v[j,k]=round(v[j,k],2)
    print("maxu",np.max(u),"minu",np.min(u),"maxv",np.max(v),"minv",np.min(v))
    return u,np.min(u),np.max(u),v,np.min(v),np.max(v) 
c=main()
fig, ax = plt.subplots()
ax.set_title('activator')
aximg = ax.imshow(c[0],cmap="binary",vmin=0,vmax=1)#vmin=c[1],vmax=c[2] #vmin=0,vmax=1
fig.colorbar(aximg, ax=ax)
plt.show()
fig, ax = plt.subplots()
ax.set_title('inhibitor')
aximg = ax.imshow(c[3],cmap="binary",vmin=0,vmax=1)#vmin=c[4],vmax=c[5] #vmin=0,vmax=1
fig.colorbar(aximg, ax=ax)
plt.show()
