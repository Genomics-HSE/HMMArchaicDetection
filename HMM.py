import msprime
import tskit
import random
import numpy as np
from copy import deepcopy


random.seed()  


states = (0,1,2)

def initS(a) -> np.array:
    S = np.zeros(3)
    S[0]=1-2*a
    S[1]=a
    S[2]=a
    return S
    
def initA(Ti,Tn,r,L,a) -> np.array:
    A = np.zeros((3,3))
    A[0][1]=Ti*r*L*a
    A[0][2]=Tn*r*L*a
    A[1][0]=Ti*r*L*(1-2*a)
    A[1][2]=Tn*r*L*a
    A[2][0]=Tn*r*L*(1-2*a)
    A[2][1]=Tn*r*L*a
    A[0][0]=1-A[0][1]-A[0][2]
    A[1][1]=1-A[1][0]-A[1][2]
    A[2][2]=1-A[2][0]-A[2][1]
    return A

#Tn : Split time between the two Nd families
#Tea:  Split time between Asia and Europe
def initB(m,L,Ti,Ta,Tea,Tn) -> np.array: 
    B = np.empty(shape=(3,30,30))
    meani = m*L*Ti
    meana = m*L*Ta
    meanea = m*L*Tea
    meann = m*L*Tn
    Pea = np.empty(30)
    Pi_ea = np.empty(30)
    Pa_ea = np.empty(30)
    Pn=np.empty(30)
    Pa_n=np.empty(30)
    Pea[0]=np.exp(-meanea)
    Pi_ea[0]=np.exp(-(meani-meanea))
    Pa_ea[0]=np.exp(-(meana-meanea))
    Pn[0]=np.exp(-(meann))
    Pa_n[0]=np.exp(-(meana-meann))
    sumea=0
    sumi_ea=0
    suma_ea=0
    sumn=0
    suma_n=0
    for i in range(1,30):
        Pea[i]=Pea[i-1]*meanea/i
        Pi_ea[i]=Pi_ea[i-1]*(meani-meanea)/i
        Pa_ea[i]=Pa_ea[i-1]*(meana-meanea)/i
        Pn[i]=Pn[i-1]*meann/i
        Pa_n[i]=Pa_n[i-1]*(meana-meann)/i
        sumea=sumea+Pea[i]
        sumi_ea=sumi_ea+Pi_ea[i]
        suma_ea=suma_ea+Pa_ea[i]
        sumn=sumn+Pn[i]
        suma_n=suma_n+Pa_n[i]

    Pea[0]=1-sumea
    Pi_ea[0]=1-sumi_ea
    Pa_ea[0]=1-suma_ea
    Pn[0]=1-sumn
    Pa_n[0]=1-suma_n
  
    for i in range(30):
        for j in range(30):
            if j<=i:
                B[0][i][j]=Pea[j]*Pi_ea[i-j]
                B[1][i][j]=Pea[j]*Pa_ea[i-j]
                B[2][i][j]=Pn[j]*Pa_n[i-j]
            else:
                B[0][i][j]=0
                B[1][i][j]=0
                B[2][i][j]=0
    return B


def print_dptable(V):
    print("      ",)
    for i in range(len(V)): print("%8d" % i,)
    print()

    for y in V[0].keys():
        print("%7.7s: " % y,)
        for t in range(len(V)):
            print("%.8s" % ("%f" % V[t][y]),)
        print()


            
def viterbi(V, initial_distribution, a, b):
    T = len(V)
    M = a.shape[0]
 
    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0][0],V[0][1]])
 
    prev = np.zeros((T - 1, M))
 
    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t][0], V[t][1]])
 
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)
 
            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
 
    # Path Array
    S = np.zeros(T)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)
 
    # Convert numeric values to actual hidden states
 
    result = []
    for s in S:
        if s == 0:
            result.append(0)
        elif s == 1:
            result.append(1)
        elif s == 2:
            result.append(2)
 
    return result


def posterior(sequence,start,transitions,emissions):
    res = initialize_matrix(len(states),len(sequence))
    f=forwardlog(sequence,start,transitions,emissions)
    b=backwardlog(sequence,start,transitions,emissions)
    for i in range(0,len(states)):
        for j in range(0,len(sequence)):
            res[i][j]=np.exp(f[i][j+1]+b[i][j+1]-f[-1][-1])
    return res
            

def forwardlog(sequence,start,transitions,emissions):
    F = initialize_matrix(len(states),len(sequence)+2)
    F[0][0] = 1
    for i in range(0,len(states)):
        F[i][1] = np.log(start[states[i]])+np.log(emissions[states[i]][sequence[0]])
    for j in range(2,len(sequence)+1):
        for i in range(0,len(states)):
            p_sum = 0
            sA = []
            for k in range(0,len(states)):
                if  transitions[(states[k],states[i])]*emissions[states[i]][sequence[j-1]]==0:
                     p_sum +=0
                else:
                    sA.append(F[k][j-1]+np.log(transitions[(states[k],states[i])])+np.log(emissions[states[i]][sequence[j-1]]))
            for  q in range(0,len(sA)):     
                p_sum += np.exp(sA[q]-max(sA))                      
            F[i][j] = np.log(p_sum)+max(sA)
    p_sum = 0
    sA = []
    for k in range(0,len(states)):
        sA.append(F[k][len(sequence)])
    for  q in range(0,len(sA)):     
        p_sum += np.exp(sA[q]-max(sA))  
    F[-1][-1] = np.log(p_sum)+max(sA)
    return F

def backwardlog(sequence,start,transitions,emissions):
    F = initialize_matrix(len(states),len(sequence)+1)
    for i in range(0,len(states)):
        F[i][-1] = 0
    for j in range(len(sequence)-1,0,-1): 
        for i in range(0,len(states)):
            p_sum = 0
            sA=[]
            for k in range(0,len(states)):
                if transitions[(states[i],states[k])]*emissions[states[k]][sequence[j]] ==0:
                    p_sum=0
                else:
                    sA.append(F[k][j+1]+np.log(transitions[(states[i],states[k])])+np.log(emissions[states[k]][sequence[j]]))
            for  q in range(0,len(sA)):     
                p_sum += np.exp(sA[q]-max(sA))                      
            F[i][j] = np.log(p_sum)+max(sA)      
    p_sum = 0
    sA=[]
    for k in range(0,len(states)):
        if start[states[k]]*emissions[states[k]][sequence[0]] ==0:
            p_sum +=0
        else:
            sA.append(F[k][1]+np.log(start[states[k]]*emissions[states[k]][sequence[0]]))
    for  q in range(0,len(sA)):     
         p_sum += np.exp(sA[q]-max(sA)) 
    F[0][0] = np.log(p_sum)+max(sA)
    return F




def training( obs_seq,S,A,B, maxIter=100000) -> (np.array,np.array,np.array):

        N = len(S)
        M = 30
        T = len(obs_seq)
        print("Ajustement du HMM par rapport Ã  %d observations" % T)
        iters = 0
        oldLogProb = float("-inf")
        logProb = 0.0
        while iters < maxIter:
            iters=iters+1
            print("ItÃ©ration %d, Ã©tat du HMM:" % iters)
            #self.impress()
            #the alpha-pass
            alpha = []
            c=[]
            #init
            c.append(0)
            alpha.append([])
            for i in range(N):
                alpha[0].append(S[i]*B[i,obs_seq[0][0],obs_seq[0][1]])
                c[0]=c[0]+alpha[0][i]
            c[0]=1.0/c[0]
            for i in range(N):
                alpha[0][i]=c[0]*alpha[0][i]
            
            #compute
            for t in range(1,T):
                c.append(0.0)
                alpha.append([])
                for i in range(N):
                    alpha[t].append(0.0)
                    for j in range(N):
                        alpha[t][i]=alpha[t][i]+alpha[t-1][j]*A[i][j]
                    alpha[t][i]=alpha[t][i]*B[i,obs_seq[t][0],obs_seq[t][1]]
                    c[t]=c[t]+alpha[t][i]
                c[t]=1.0/c[t]
                for i in range(N):
                    alpha[t][i]=c[t]*alpha[t][i]

            #Compute log[P(Obs_seq)]
            logProb=0.0
            for i in range(T):
                logProb=logProb+np.log(c[i])
            logProb=-logProb
            print("Log[P(ObsSeq)] = %f\n" % logProb)
            if logProb > oldLogProb:
                oldLogProb=logProb
            else:
                break

            #beta-pass
            beta = []
            beta.append([])
            for i in range(N):
                beta[0].append(c[T-1])
            for t in range(T-2,-1,-1):
                beta.insert(0,[])
                for i in range(N):
                    beta[0].append(0.0)
                    for j in range(N):
                        beta[0][i]=beta[0][i]+A[i][j]*B[j,obs_seq[t+1][0],obs_seq[t+1][1]]*beta[1][j]
                    beta[0][i]=c[t]*beta[0][i]

            #compute gama-ti and gama-tij
            gamaTi = []
            gamaTij = []
            for t in range(T-1):
                denom=0.0
                for i in range(N):
                    for j in range(N):
                        denom=denom+alpha[t][i]*A[i][j]*B[j,obs_seq[t+1][0],obs_seq[t+1][1]]*beta[t+1][j]
                gamaTi.append([])
                gamaTij.append([])
                for i in range(N):
                    gamaTi[t].append(0.0)
                    gamaTij[t].append([])
                    for j in range(N):
                        gamaTij[t][i].append((alpha[t][i]*A[i][j]*B[j,obs_seq[t+1][0],obs_seq[t+1][1]]*beta[t+1][j])/denom)
                        gamaTi[t][i] = gamaTi[t][i] + gamaTij[t][i][j]

            #Save start, trans, emission probabilities
            oldStart_p = deepcopy(S)
            oldTrans_p = deepcopy(A)
            oldEmission_p = deepcopy(B)


            #Re-estimate start, transition and emission probabilities
            #Start
            for i in range(N):
                S[i]=gamaTi[0][i]

            #Transition
            for i in range(N):
                for j in range(N):
                    numer = 0.0
                    denom = 0.0
                    for t in range(T-1):
                        numer = numer+gamaTij[t][i][j]
                        denom = denom + gamaTi[t][i]
                    A[i][j]=numer/denom

            #Emission
            for i in range(N):
                for j in range(M):
                    for k in range(M):
                        numer = 0.0
                        denom = 0.0
                        for t in range(T-1):
                            if obs_seq[t][0] == j and obs_seq[t][1] == k:
                                numer = numer+gamaTi[t][i]
                            denom=denom+gamaTi[t][i]
                        B[i,j,k]=numer/denom

        S = oldStart_p
        A = oldTrans_p
        B = oldEmission_p
        return (S,A,B)
    
def initialize_matrix(dim1,dim2,value=0):
    F = []
    for i in range(0,dim1):
        F.append([])
        for j in range(0,dim2):
            F[i].append(value)
    return F