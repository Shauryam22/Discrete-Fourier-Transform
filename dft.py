import numpy as np
import matplotlib.pyplot as plt
def f(t):
    return 5 + 2 * np.cos(2 * np.pi * t - np.pi/2) + 3 * np.cos(4 * np.pi * t - np.pi/2)
def f_trans(f,N,T0):
    ft = list()
    T = T0/N
    w = np.exp(2*np.pi*1j/N)
    j,k = np.meshgrid(np.arange(N),np.arange(N))
    dft = np.power(w,j*k)
    for i in range(N):
        t = i*T
        ft.append(f(t))
    ft1 = np.array(ft)
    ft1 = ft1.reshape(-1,1)
    t = np.arange(N)*T
    #print(ft1)
    mult = np.matmul(dft,ft1)
    #print(dft)
    f_hat = mult/T0   # F(w)
    return t,f_hat,dft,T
    #print("F(w)",np.round(f_hat,2))
    #res = res.reshape(1,-1)
    #print(res)
#fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#print("DFT",f_trans(gaussian,4,1)[2])
plt.figure(figsize=(10,8))
N = 10
plt.subplot(2,2, 1)
x = np.linspace(0,1,100)
plt.plot(x,f(x),linestyle="dotted",label = "original function")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.grid()
plt.title(f"N = {N}")
#plt.show()
plt.subplot(2,2, 2)
#print(np.meshgrid(np.arange(4),np.arange(4)))

t,f_hat,dft,T = f_trans(f,N,1)
plt.scatter(t,f(t),color = "red",label = "sample points")
plt.stem(t,f(t),markerfmt='ro')
x = np.linspace(0,1,100)
f_val = f(x)
#print(f_val)
plt.plot(x,f_val,linestyle = 'dotted',label = "original signal")
plt.grid()
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
#plt.show()
plt.title(f"N = {N}")
#print(dft)



# Inverse DFT
inv_dft = np.linalg.inv(dft)
#print(inv_dft)
res_matrix = inv_dft @ f_hat*T
print("Resultant matrix",res_matrix)


# DFT Response plot
plt.subplot(2,2, 3)
n = np.arange(N)
factor = n/(N * T)

plt.plot(factor,f_hat)
plt.scatter(factor,f_hat,color = "seagreen")
plt.grid()
plt.xlabel("w/2*pi")
plt.ylabel("F(w)")
#plt.show()
plt.title(f"N = {N}")
# Signal Strength Plot
plt.subplot(2,2, 4)
n = np.arange(N)
w = n*2*np.pi/(N*T)
plt.plot(w,np.abs(f_hat))
plt.scatter(w,np.abs(f_hat),color="green")
plt.xlabel("w")
plt.ylabel("|F(w)|")
plt.grid()
plt.tight_layout()
plt.title(f"N = {N}")
plt.show()

# Error plots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
N = 2
x = np.linspace(0,1,100)
t,f_hat,dft,T = f_trans(f,N,1)
axs[0, 0].scatter(t, f(t), color='b',label='sample points')
axs[0,0].stem(t,f(t),markerfmt='bo')
axs[0,0].plot(x,f(x),linestyle = 'dotted',label = "original signal")
axs[0,0].grid()
axs[0,0].set_xlabel("t")
axs[0,0].set_ylabel("f(t)")
axs[0, 0].set_title(f'N={N}')
axs[0,0].legend()
N = 3
t,f_hat,dft,T = f_trans(f,N,1)
x = np.linspace(0,1,100)
axs[0, 1].scatter(t, f(t), color='b',label='sample points')
axs[0,1].stem(t,f(t),markerfmt='bo')
axs[0,1].plot(x,f(x),linestyle = 'dotted',label = "original signal")
axs[0,1].grid()
axs[0, 1].set_title(f'N={N}')
axs[0,1].set_xlabel("t")
axs[0,1].set_ylabel("f(t)")
axs[0,1].legend()
N = 4
x = np.linspace(0,1,100)
t,f_hat,dft,T = f_trans(f,N,1)
axs[1, 0].scatter(t, f(t), color='b',label='sample points')
axs[1,0].stem(t,f(t),markerfmt='bo')
axs[1,0].plot(x,f(x),linestyle = 'dotted',label = "original signal")
axs[1,0].grid()
axs[1,0].legend()
axs[1,0].set_xlabel("t")
axs[1,0].set_ylabel("f(t)")

axs[1,0].set_title(f'N = {N}')
N = 500
t,f_hat,dft,T = f_trans(f,N,1)
x = np.linspace(0,1,100)
axs[1, 1].scatter(t, f(t), color='b',label='sample points')
axs[1,1].stem(t,f(t),markerfmt='bo')
axs[1,1].plot(x,f(x),linestyle = 'dotted',label = "original signal")
axs[1,1].grid()
axs[1,1].set_title(f'N = {N}')
axs[1,1].legend()
axs[1,1].set_xlabel("t")
axs[1,1].set_ylabel("f(t)")
plt.tight_layout()
plt.suptitle('Aliasing', fontsize=12)
plt.show()


# F(w)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
N = 2
x = np.linspace(0,1,100)
n = np.arange(N)
w = n*2*np.pi/(N*T)
t,f_hat,dft,T = f_trans(f,N,1)
axs[0, 0].scatter(w,np.abs(f_hat))
#axs[0,0].stem(t,f(t),markerfmt='bo')
axs[0,0].plot(w,np.abs(f_hat))
axs[0,0].grid()
axs[0,0].set_xlabel("w")
axs[0,0].set_ylabel("|F(w)|")
axs[0, 0].set_title(f'N={N}')
axs[0,0].legend()

N = 3
x = np.linspace(0,1,100)
n = np.arange(N)
w = n*2*np.pi/(N*T)
t,f_hat,dft,T = f_trans(f,N,1)
axs[0, 1].scatter(w,np.abs(f_hat))
#axs[0,0].stem(t,f(t),markerfmt='bo')
axs[0,1].plot(w,np.abs(f_hat))
axs[0,1].grid()
axs[0,1].set_xlabel("w")
axs[0,1].set_ylabel("|F(w)|")
axs[0,1].set_title(f'N={N}')
axs[0,1].legend()

N = 8
x = np.linspace(0,1,100)
n = np.arange(N)
w = n*2*np.pi/(N*T)
t,f_hat,dft,T = f_trans(f,N,1)
axs[1, 0].scatter(w,np.abs(f_hat))
#axs[0,0].stem(t,f(t),markerfmt='bo')
axs[1,0].plot(w,np.abs(f_hat))
axs[1,0].grid()
axs[1,0].set_xlabel("w")
axs[1,0].set_ylabel("|F(w)|")
axs[1, 0].set_title(f'N={N}')
axs[1,0].legend()

N = 50
x = np.linspace(0,1,100)
n = np.arange(N)
w = n*2*np.pi/(N*T)
t,f_hat,dft,T = f_trans(f,N,1)
axs[1, 1].scatter(w,np.abs(f_hat))
#axs[0,0].stem(t,f(t),markerfmt='bo')
axs[1,1].plot(w,np.abs(f_hat))
axs[1,1].grid()
axs[1,1].set_xlabel("w")
axs[1,1].set_ylabel("|F(w)|")
axs[1,1].set_title(f'N={N}')
axs[1,1].legend()

plt.tight_layout()
plt.suptitle('Aliasing', fontsize=12)
plt.show()



# Application

def gaussian(t,s=1):
    return (1/((2*np.pi*s*s)**.5))*np.exp((-t*t/(2*s*s))) 
def ft(f,t):
    w = 2*np.pi
    arg = complex(0,w*t)
    return f(t)*np.exp(-arg)
    
def simpson(f,a,b,n):
    h = abs((b-a)/n)
    Sum = 0
    for i in range(1,n):
        x = a + i*h
        if i%2==0:
            Sum += 2*f(x)
        else:
            Sum += 4*f(x)
    result = h*(f(a)+f(b))/3 + h*Sum/3
    return result

def Gauss_Simp(w):
    v = lambda t: np.exp(-1j*w*t)
    f = lambda t: gaussian(t)*v(t)
    integral = simpson(f,-500,500,10000)
    return integral/np.sqrt(2*np.pi)
print("ft",Gauss_Simp(1))
print(gaussian(1))
# Square wave
def sq_wave(t):
    if type(t)== np.ndarray:
        l = []
        for i in range(len(t)):
            if t[i]!= 0:
                l.append(abs(np.sin(t[i]))/np.sin(t[i]))
            else:
                l.append(0)
        return np.array(l)
    elif t != 0:
        return abs(np.sin(t))/np.sin(t)
    else:
        return 0
def Gauss_w(w):
    sig = 1
    return np.exp(-w*w*sig*sig/2)/np.sqrt(2*np.pi)

def Gauss_inv(t):
    v = lambda w: np.exp(1j*w*t)
    f = lambda w: Gauss_w(w)*v(w)
    return simpson(f,-500,500,10000)/np.sqrt(2*np.pi)
print('Check')
print('Inv transform: ',Gauss_inv(0))
print('Original: ',gaussian(0))

# Non-periodic wave
def non_wave(t):
    return t**2*np.sin(t)

# Plotting of Gaussian
P = 1
w0 = 80
N = 5
w = np.arange(-10,N,1)

plt.suptitle('Fourier transform of Gaussian Pulse using Simpson')
ax1 = plt.subplot(221)
#t = np.arange(-10,10,0.5)
t_sam = np.arange(-N,N,1/N)
ax1.plot(t_sam,gaussian(t_sam))#linestyle='dashed')
ax1.scatter(t_sam,gaussian(t_sam),color='red',label='sample points')
ax1.legend()
ax1.set_title('Guassian curve')
ax1.grid()

#ax2 = plt.subplot(232)
#ax2.scatter(w,Gauss_w(w),color='orange')
#ax2.set_xlabel('F(w) Analytical vs w/(2*pi)')
#ax2.grid()

#F_dft = f_trans(gaussian,12,-0.01)[1]
#w1 = np.arange(-0.01,0.01,0.02/N)
w_2pi = np.arange(-N,N,1/N)
ax3 = plt.subplot(222) 
ax3.scatter(w_2pi,Gauss_Simp(w_2pi).real,color = 'brown')
ax3.plot(w_2pi,Gauss_Simp(w_2pi))
ax3.set_ylabel('F(w)')
ax3.set_xlabel('w')
ax3.grid()

#ax4 = plt.subplot(223)
#ax4.scatter(w1,F_dft.real)
#ax4.set_title('F(w) DFT')
#ax4.grid()

ax5 = plt.subplot(223)
ax5.scatter(w_2pi,abs(Gauss_Simp(w_2pi)),color ='red')
ax5.plot(w_2pi,abs(Gauss_Simp(w_2pi)))
ax5.set_ylabel('|F_w|')

ax5.grid()

#ax6 = plt.subplot(236)
#t = np.arange(-5,5,0.05)
#t1 = np.arange(-5,5,0.5)
#ax6.plot(t,gaussian(t))
#ax6.scatter(t1,Gauss_inv(t1).real,color='green')
#ax6.set_title('Aliasing')
#ax6.grid()
plt.tight_layout()
plt.show()

# Plotting Square Wave
def f_tran(f,T,N,t0=0):
    # Twiddle Matrix
    del_t = T/N
    arg = complex(0,-2*np.pi/N)
    w = np.exp(arg)
    j,k = np.meshgrid(np.arange(N),np.arange(N))
    Twiddle = np.power(w,j*k)
    # f_n matrix
    
    f_s = []
    for v in range(N):
        f_s.append(f(t0+v*del_t))
    f_n = np.transpose(np.array(f_s))
    F_transform = np.dot(Twiddle,f_n)
    return Twiddle,f_n,F_transform
# Non-periodic wave
def non_wave(t):
    return t**2*np.sin(t)
# Inverse DFT
def dft_inv(f,F,T,N,t0=0):
    twiddle,f_n,F_trans = F(f,T,N,t0)
    twi = np.conj(twiddle)
    f_t = np.dot(twi,F_trans)/N
    return f_t
# Plotting Square Wave
T = 2*np.pi
N = 20
Fsq = f_tran(sq_wave,T,N)[2]
t = np.linspace(0,T,200)
t_pts = np.arange(0,T,T/N)
plt.suptitle('Square wave')
ax1 = plt.subplot(231)
ax1.plot(t,sq_wave(t),linestyle = 'dashed')
ax1.scatter(t_pts,sq_wave(t_pts),color='red',label='sample points')
ax1.legend()
ax1.set_title('Plotting of curve and sampling')
ax1.grid()
print("w",w)
print(Fsq)
ax2 = plt.subplot(232)
w0 = 1
w = np.linspace(0,N,N)
ax2.scatter(w*w0/(2*np.pi),Fsq.real)
ax2.set_title('F(w) vs w/(2*pi)')
ax2.grid()

ax3 = plt.subplot(233)
ax3.stem(w,abs(Fsq))
ax3.set_title('|F(w)| vs w')
ax3.grid()
plt.show()

# Plotting non-wave
T_nw = 5
N = 20
F_nwv = f_tran(non_wave,T_nw,N,-2.5)[2]
t = np.linspace(-2.5,-2.5+T_nw,100)
t_pt = np.arange(-2.5,-2.5+T_nw,T_nw/N)
plt.suptitle('Non-wave')
ax1 = plt.subplot(231)
ax1.plot(t,non_wave(t),linestyle = 'dashed')
ax1.scatter(t_pt,non_wave(t_pt),color='red',label='sample points')
ax1.legend()
ax1.set_title('Plotting of curve and sampling')
ax1.grid()

ax2 = plt.subplot(232)
w0 = 2*np.pi/T_nw
w = np.linspace(0,N,N)
ax2.scatter(w*w0/(2*np.pi),F_nwv.real)
ax2.set_title('F(w) vs w/(2*pi)')
ax2.grid()

ax3 = plt.subplot(233)
ax3.stem(w,abs(F_nwv))
ax3.set_title('|F(w)| vs w')
ax3.grid()
plt.show()