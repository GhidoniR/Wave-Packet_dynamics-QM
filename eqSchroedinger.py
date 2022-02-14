#!/usr/bin/env python
# coding: utf-8

# In[1]:


#numpy serve per le funzioni matematiche
import numpy as np

#matplotlib serve per i grafici
import matplotlib.pyplot as plt

#time serve per misurare il tempo di esecuzione del codice
import time

#exit serve per provocare l'uscita forzata da un codice
from sys import exit


# In[2]:


#Leggiamo dal file i valori delle variabili
f = open('testoprove.txt', 'r')

var=f.readline()
L=int(var)     #Lunghezza della scatola

var=f.readline()
T=int(var)     #Intervallo temporale

var=f.readline()
N=int(var)     #Numero di punti nello spazio
if N<500:     #Controlliamo che la discretizzazione spaziale sia abbastanza accurata
    print('il numero di punti nello spazio è troppo piccolo')
    exit()
    
var=f.readline()
M=int(var)    #Numero di punti nel tempo
if M<500:    #Controlliamo che la discretizzazione temporale sia abbastanza accurata
    print('il numero di punti nello spazio è troppo piccolo')
    exit()
    
var=f.readline()
k0=int(var)    #Velocità di gruppo del pacchetto d'onda

var=f.readline()
E1=int(var)    #Altezza della prima barriera di potenziale

var=f.readline()
E2=int(var)    #Altezza della seconda barriera di potenziale

var=f.readline()
L1=int(var)    #Spessore della prima barriera di potenziale

var=f.readline()
L2=int(var)    #Spessore della seconda barriera di potenziale

var=f.readline()
sigma=int(var) #Varianza

var=f.readline()
freq=int(var)  #Frequenza (mi dice ogni quanto salvare la traiettoria)

f.close
l1=int(L1/400*N)
l2=int(L2/400*N)

#definiamo altre variabili importanti
dx=L/N         #Discretizzazione spaziale

dt=T/M         #Discretizzazione temporale

x0=L/8        #Valore medio della gaussiana iniziale

x=0            #Estremo spaziale iniziale (lo spazio va da 0 a L)

t=0            #Tempo iniziale


# In[3]:


#Definiamo la matrice nella quale salvare la traiettoria per ogni istante di tempo
psir=np.zeros( (N,M+1), dtype=complex )


# In[4]:


#Definiamo la forma iniziale del pacchetto d'onda (si tratta di una gaussiana)
iniz=np.zeros( (N), dtype=complex)

cost= 1/(2*np.pi*sigma*sigma)**(1/4)

for n in range(0,N):
    a=complex(0,k0*x)
    psir[n,0]=cost*((np.e)**(-((x-x0)**2)/(4*sigma*sigma)))*(np.e**(a))
    x=x+dx


# In[5]:



#Definiamo la forma del potenziale
def pot(N,E1,l1,E2,l2):
    V=np.zeros(N)
    for n in range (0,N):
        if n>=N/3 and n<=N/3+l1:
            V[n]=E1
        elif n>=N/2 and n<=N/2+l2:
            V[n]=E2
        else:
            V[n]=0

    return V

#Controlliamo che il potenziale abbia la forma desiderata di 2 barriere di potenziale distinte
if N/3+l1>=N/2:
    print('Le barriere di potenziale si sovrappongono')
    exit()

#Richimiamo il potenziale
V=pot(N,E1,l1,E2,l2)



#Definiamo G1
b=np.zeros(N, dtype=complex)

G1=np.zeros(N, dtype=complex)

for n in range(0,N):
    b[n]=complex(0,-dt/2*V[n])
    G1[n]=(np.e)**(b[n])
    
plt.plot(V)


# In[6]:


#Definamo G2
k=np.fft.fftfreq(N,2*np.pi/L) #Punti dello spazio dei momenti

a=np.zeros(N, dtype=complex)  

G2=np.zeros(N, dtype=complex)
for n in range(0,N):
    a[n]=complex(0,(dt/2)*k[n]**2)
    G2[n]=(np.e)**(a[n])


# In[7]:


#Facciamo evolvere il sistema
start = time.time()


A=np.zeros(N, dtype=complex)

C=np.zeros(N, dtype=complex)
for m in range (0,M):
    for n in range(0,N):
        A[n]=G1[n]*psir[n,m]
    B=np.fft.fft(A) 
    for n in range(0,N):
        C[n]=G2[n]*B[n]
    D=np.fft.ifft(C)
    for n in range(0,N):
        psir[n,m+1]=G1[n]*D[n]
    area = np.trapz((abs(psir[:,m])**2), dx=dx)
    if abs(1-area)>0.005:
        print('la norma non si conserva')
        print('la variazione della norma non è più trascurabile a partire dal tempo', dt*m)
        exit()
    
    
end = time.time()

elapsed = end - start
print(elapsed)


# In[8]:


#Controlliamo la conservazione della norma (che deve essere unitaria)
Err_max=0

i=0
norm_iniz=np.trapz((abs(psir[:,m])**2), dx=dx)
print(norm_iniz)
for m in range (0,M):
    norm = np.trapz((abs(psir[:,m])**2), dx=dx)
    Err=abs((norm_iniz-norm)/norm_iniz)
    if Err>Err_max:
        Err_max=Err
        i=m 
    
print('errore massimo percentile=', Err_max*100,'al tempo',i*dt)


# In[9]:


#Salviamo i risultati ottenuti su un file di tipo testo (TASK 1)
file=open('traiettoria.txt','w')

for m in range(0,M):
    if m%freq==0:
        file.write('tempo:')
        file.write(str(m*dt) )
        file.write(' posizione:')
        file.write(str(abs(psir[:,m])**2))
        file.write('\n')
        
file.close


# In[10]:


#Grafichiamo il pacchetto d'onda a 3 istanti distinti (servirà per verificare l'accurtezza in funzione di dx/dt)
ascissa=np.arange(0,L,dx)

f = plt.figure(figsize=(14,14))


#Grafico al tempo zero
plt.subplot(331) 
plt.title("FUNZIONE D'ONDA")
plt.plot(ascissa,(abs(psir[:,0]))**2) 
plt.plot(ascissa,V)
plt.ylim(0,0.055)
plt.xlabel('L')
plt.ylabel('psi_modulo_quadro')


#Grafico a metà dell'intervallo temporale
plt.subplot(332)   
plt.title("FUNZIONE D'ONDA")
plt.plot(ascissa,(abs(psir[:,int(M/8)]))**2) 
plt.plot(ascissa,V)
plt.ylim(0,0.055)
plt.xlabel('L')
plt.ylabel('psi_modulo_quadro')


#Grafico al tempo finale
plt.subplot(333) 
plt.title("FUNZIONE D'ONDA")
plt.plot(ascissa,(abs(psir[:,int(M/4)]))**2) 
plt.plot(ascissa,V)
plt.ylim(0,0.055)
plt.xlabel('L')
plt.ylabel('psi_modulo_quadro')

#Grafico al tempo zero
plt.subplot(334) 
plt.title("FUNZIONE D'ONDA")
plt.plot(ascissa,(abs(psir[:, int(M/8*3)]))**2) 
plt.plot(ascissa,V)
plt.ylim(0,0.055)
plt.xlabel('L')
plt.ylabel('psi_modulo_quadro')


#Grafico a metà dell'intervallo temporale
plt.subplot(335)   
plt.title("FUNZIONE D'ONDA")
plt.plot(ascissa,(abs(psir[:,int(M/2)]))**2) 
plt.plot(ascissa,V)
plt.ylim(0,0.055)
plt.xlabel('L')
plt.ylabel('psi_modulo_quadro')


#Grafico al tempo finale
plt.subplot(336) 
plt.title("FUNZIONE D'ONDA")
plt.plot(ascissa,(abs(psir[:,int(M/8*5)]))**2) 
plt.plot(ascissa,V)
plt.ylim(0,0.055)
plt.xlabel('L')
plt.ylabel('psi_modulo_quadro')

#Grafico al tempo zero
plt.subplot(337) 
plt.title("FUNZIONE D'ONDA")
plt.plot(ascissa,(abs(psir[:,int(M/8*6)]))**2) 
plt.plot(ascissa,V)
plt.ylim(0,0.055)
plt.xlabel('L')
plt.ylabel('psi_modulo_quadro')


#Grafico a metà dell'intervallo temporale
plt.subplot(338)   
plt.title("FUNZIONE D'ONDA")
plt.plot(ascissa,(abs(psir[:,int(M/8*7)]))**2) 
plt.plot(ascissa,V)
plt.ylim(0,0.055)
plt.xlabel('L')
plt.ylabel('psi_modulo_quadro')


#Grafico al tempo finale
plt.subplot(339) 
plt.title("FUNZIONE D'ONDA")
plt.plot(ascissa,(abs(psir[:,M]))**2) 
plt.plot(ascissa,V)
plt.ylim(0,0.055)
plt.xlabel('L')
plt.ylabel('psi_modulo_quadro')



#plt.subplots_adjust(left=0.01, right=1.6, wspace=0.9, hspace=5)

#plt.show()
plt.savefig('graf123.png')


# In[ ]:




