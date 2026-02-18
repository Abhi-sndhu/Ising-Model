import matplotlib.pyplot as plt
import numpy as np
plt.ion()
fig,ax = plt.subplots(figsize=(8,8))

scale=2

grid = np.random.choice([-1,-1], size=(41*scale,41*scale))
Nx, Ny = grid.shape
N=Nx*Ny
cax = ax.imshow(grid, cmap='gray',vmin=-1,vmax=1)
plt.pause(1e-10)

##########################################################################################################################################################################################
#Parameters deciding update rule:
#Ising model (spin=+1/-1): Spin flipping depending on temperature, external field, coupling with neighbours 
kB=1#1.380649e-23 #Boltzmann Constant kB​=1.380649×10−23J/K
g=1#2
muB=1 #9.274* 1e-24 #Bohr Magneton: 9.274×10−24J/T
hcut=1

J=1#1,0,-1    #Coupling strength, Ferro: 1e−20,Para: 1e-24, Antiferro: -1e-21
B=0#External field, J dominates:0, WeakB: 0.01J, Strong B: 0.1J
T=1.5#In natural units 1 to 3, Tc = 2.27    #(J/kB)*1e-100 #Temperature in Kelvin: (Transition : kB Tc ~ J)

##########################################################################################################################################################################################
def deltaE(i,j): #Change in energy if flip, deltaE=J(s)(sum of spins of all neighbours)-2B(s)
    sum_nspins = grid[(i-1)%Nx,j%Ny]+ grid[i%Nx,(j-1)%Ny]+grid[i%Nx,(j+1)%Ny]+ grid[(i+1)%Nx,j%Ny] 
    delE=2*J*grid[i,j]*sum_nspins - 2*g*muB*B*grid[i,j]
    return(delE)

def Ener(grid):
    TotE=0
    for i in range(Nx):
        for j in range(Ny):
            sum_nspins = grid[(i-1)%Nx,j%Ny]+ grid[i%Nx,(j+1)%Ny]
            TotE+=-J*grid[i,j]*sum_nspins + g*muB*B*grid[i,j]
    return(TotE)

def Prob(i,j): #Stochastic (Glauber) update according to Temp and dE
    Prob= 1/( 1 + np.exp(deltaE(i,j)/(kB*T)) )
    return (Prob)
 
def result(grid,steps):
   n=0
   Mag=[]
   Mag2=[]
   E=[]
   E2=[]
   while n<steps:
       for _ in range (N):
           i = np.random.randint(0, grid.shape[0])
           j = np.random.randint(0, grid.shape[1])
           if np.random.rand() < Prob(i,j):  #Stochastic Update
              grid[i,j] *= -1 #flip spin
    
    
       n+=1
       M=-np.mean(grid) #Mag=-g*mub*sigma Si/hcut
       #M=np.abs(np.mean(grid))
       Mag.append( M ) #/(grid.shape[0]*grid.shape[1])
       Mag2.append( M**2 )
       ener=Ener(grid)
       E.append(ener)
       E2.append(ener**2)
       cax.set_data(grid)
       ax.set_title(f"{n}sweep, M={np.mean(Mag):.2f}, B={B:.2f},T={T:.2f}")
       plt.draw()
       plt.pause(1e-10)
       mean_mag_abs=np.mean(np.abs(Mag))
       mean_mag=np.mean(Mag)
       Chi=N*(np.mean(Mag2)-(mean_mag)**2)/(kB*T) #N/kbT(<M^2>-<M>^2), if M= mag per spin
       Energy=np.mean(E)
       Cv= (np.mean(E2) -Energy**2)/(N*kB* T**2)
   return(mean_mag,Chi,Energy,Cv,mean_mag_abs )

##########################################################################################################################################################################################


Mlst,Tlst,Chilst,Elst,Cvlst=[],[],[],[],[]
for i in range(31):
    cax.set_data(grid)
    ax.set_title(f"B={B:.2f},T={T:.2f}")
    plt.draw()
    plt.pause(1e-10)
    Mag,Chi,Energy,Cv,Mag_abs=result(grid,100)
    Mlst.append(Mag_abs)
    Tlst.append(T)
    Chilst.append(Chi)
    Elst.append(Energy)
    Cvlst.append(Cv)
    T+=0.05
    print('Step:',i,'/ 30')


##########################################################################################################################################################################################
Blst,M2lst=[],[]
T=2
grid = np.random.choice([-1,1], size=(41*scale,41*scale))
cax.set_data(grid)
ax.set_title(f"Hysteresis")
plt.draw()
plt.pause(1)
B=-0.005
for i in range(76):
    if i<15:
       B+=0.03
    elif i<45:
       B-=0.03
    elif i<75:
        B+=0.03
    Mag,Chi,Energy,Cv,Mag_abs=result(grid,20)
    M2lst.append(Mag)
    Blst.append(B)
    print('Step:',i,'/ 75')

##########################################################################################################################################################################################
plt.figure()
plt.plot(Blst,M2lst,marker='o',linestyle='-')
plt.xlabel('H')
plt.ylabel('M')
plt.figure()
plt.plot(Tlst,Mlst,marker='o',linestyle='-')
plt.xlabel('Temp')
plt.ylabel('Magnetization')
plt.figure()
plt.plot(Tlst,Chilst,marker='o',linestyle='-')
plt.xlabel('Temp')
plt.ylabel('Susceptibility')
plt.figure()
plt.plot(Tlst,Elst,marker='o',linestyle='-')
plt.xlabel('Temp')
plt.ylabel('Energy')
plt.figure()
plt.plot(Tlst,Cvlst,marker='o',linestyle='-')
plt.xlabel('Temp')
plt.ylabel('Specific Heat Capacity')
plt.show()
