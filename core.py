import msprime
import tskit
import random
import time

import numpy as np



random.seed()  


        

#Get all tracts that migrated in pop.
#Run faster than get_migrating_tracts_ind, but does not distinguish between individuals

def get_migrating_tracts(ts,pop):
    neanderthal_id = [p.id for p in ts.populations() if p.metadata['name']==pop][0]
    migrating_tracts = []
    for migration in ts.migrations():
        if migration.dest == neanderthal_id:
            migrating_tracts.append((migration.left, migration.right))
    return np.array(migrating_tracts)


# Input:
#      pop: a population
#      ind: id of the node corresponding to the individual
# Output: 
#      all tracts from ind that migrated in pop
        
def get_migrating_tracts_ind(ts,pop,ind,size=1000):
    print(" --- Start get_migrating_tracts_ind ---")
    pop_id = [p.id for p in ts.populations() if p.metadata['name']==pop][0]
    mig = ts.tables.migrations
    migrating_tracts = []
    for tree in ts.trees():
        u = ind
        flag=True
        while u != tskit.NULL and flag:
            migs = np.where(mig.node == u)[0]
            for cur_mig in migs:
                if flag:
                    cur_mig = mig[cur_mig]
                    if(cur_mig.dest==pop_id and cur_mig.left<=tree.interval.left and cur_mig.right>=tree.interval.right):
                        flag=False
                        if(len(migrating_tracts)>0 and tree.interval.left==migrating_tracts[len(migrating_tracts)-1][1]):
                            migrating_tracts[len(migrating_tracts)-1][1]=tree.interval.right                     
                        else:
                            migrating_tracts.append([tree.interval.left,tree.interval.right])

            u = tree.parent(u)
    #Clean duplicates
    tracts = migrating_tracts
    flag = True
    while(flag):
        i=0
        flag=False
        while i < len(tracts) and not flag:
            l=[]
            j=i+1
            while j < len(tracts):
                if tracts[i][0]==tracts[j][0] and tracts[i][1]==tracts[j][1]:
                    l.append(j)
                    flag=True
                j=j+1
            cnt=0
            for j in l:
                tracts.pop(j-cnt)
                cnt=cnt+1
            i=i+1
    migrating_tracts = clean_tracts(migrating_tracts,size)
    print(" --- End get_migrating_tracts_ind ---")
    return migrating_tracts



def createSeqObs(ts,cut,ind,p1,p2,p3) -> list:
    print(" --- Start createSeqObs ---")
    tables = ts.dump_tables()
    nodes = tables.nodes
    seq = np.zeros((int(ts.sequence_length/cut),2),dtype=int) #List of seq, a seq is a list with nb of mutations
    african_id = [p.id for p in ts.populations() if p.metadata['name']=="Af"][0]
    asian_id = [p.id for p in ts.populations() if p.metadata['name']=="As2"][0]
    
    start_time = time.time()
    pop=[]
    r=0
    for v in ts.variants():
        i=int(v.site.position/cut)
        
        r=r+1
        b = False
        c = False
        d = False
        j=0
        x=0
        while x < len(v.genotypes):
            if(nodes[x].individual==ind):
                if(v.genotypes[x]==1):
                    b = True
                elif not b and x==p1-1:
                    x=p1+p2+p3
            elif(b and nodes[x].population==african_id):
                if(v.genotypes[x]==1):
                    c = True
                    x=p1+p2-1
                    
            elif(b and nodes[x].population==asian_id):
                if(v.genotypes[x]==1):
                    d = True 
                    x=p1+p2+p3-1
            j=j+1
            x=x+1
        if b and not c :
            seq[i][0]=seq[i][0]+1
        if b and not c and not d :
            seq[i][1]=seq[i][1]+1
    print(" --- End createSeqObs --- ")

    return seq

# Input:
#   ts:  A Tree Sequence
#   cut: The length of the tracts
#   ind: An individual id
#   pop: A population name
#Output:
#   For each tract, the number of mutation in ind that is not found in pop


def createSeqObsSingle(ts,cut,ind,pop,max_sample=1000) -> list:
    tables = ts.dump_tables()
    nodes = tables.nodes
    seq = np.zeros(int(ts.sequence_length/cut),dtype=int)  #The list which will contain the result of our function.
    pop_id = [p.id for p in ts.populations() if p.metadata['name']==pop][0] 
    start=-1
    end=-1
    for i in range(len(nodes)):
        if nodes[i].population==pop_id and start ==-1:
            start = i
        if nodes[i].population!=pop_id and start !=-1:
            end = i-1
            break
    if end-start > max_sample:
        end = start+max_sample-1

    for v in ts.variants():
        i=int(v.site.position/cut)
        c = False
        x=0
        while x <= end: 
            if(nodes[x].individual==ind): # Check is this position corresponds to individual 'ind'
                b=v.genotypes[x]
                x=start
            elif(nodes[x].population==pop_id): # Check is this position corresponds to an individual in 'pop'
                if(v.genotypes[x]==b): 
                    c = True # The mutation found in 'ind' is also present in the population 'pop' 
                    x = end # We can skip all the remaining individuals
            x=x+1
            
            
        if not c : # If c is False it means that the mutation in 'ind' has not been found in the population 'pop' 
            seq[i]=seq[i]+1
    return seq


# Returns the list of tracts from the output of the HMM
def get_HMM_tracts(seq):
    migrating_tracts = []
    maxi = 0
    for e in seq:
        if e>maxi:
            maxi=e
            
    for i in range(maxi+1):
        migrating_tracts.append([])
    start=0
    for i in range(1,len(seq)):
        if seq[i]!=seq[i-1]:
            migrating_tracts[seq[i-1]].append([start,i-1])
            start=i
    migrating_tracts[seq[len(seq)-1]].append([start,len(seq)-1])
    return migrating_tracts

# Remove duplicates, merge adjacent tracts, and sort them.
def clean_tracts(tractInit,size=1000):
    tract = np.copy(tractInit)
    tract = tract/size
    tract=tract.astype(int)
    flag = True
    while(flag):
        flag=False
        for i in range(len(tract)):
            for j in range(len(tract)):
                if not flag and tract[i,0]==tract[j,1]:
                    tract[j,1]=tract[i,1]
                    tract = np.delete(tract,i,0)
                    flag=True
    flag = True
    while(flag):
        flag=False
        for i in range(len(tract)):
            for j in range(i+1,len(tract)):
                if tract[i,0]>tract[j,0]:
                    save0=tract[i,0]
                    save1=tract[i,1]
                    tract[i,0]=tract[j,0]
                    tract[i,1]=tract[j,1]
                    tract[j,0]=save0
                    tract[j,1]=save1
                    flag=True
    return tract


def inTracts(pos,tract):
    for i in range(len(tract)):
        if pos>=tract[i][0] and pos <= tract[i][1]:
            return True
    return False
            

def get_tract_length(tracts):
    l=0
    for t in tracts:
        l=l+t[1]-t[0]+1
    return l


#Return the tracts that are in both input
def intersect_tracts(tracts1,tracts2):
    max = 0
    res =[]
    for t in tracts1:
        if t[1]>max:
            max=t[1]
    for t in tracts2:
        if t[1]>max:
            max=t[1]
    b1=False
    b2=False
    inside=False
    start=-1
    for i in range(max+1):
        if (inTracts(i,tracts1)):
            b1=True
        else:
            b1=False
        if (inTracts(i,tracts2)):
            b2=True
        else:
            b2=False
        if (b1 and b2 and not inside):
            inside = True
            start = i
        elif( not(b1 and b2) and inside):
            inside = False
            res.append([start,i])
    return res

#Return the first tract minus the second
def substract_tracts(tracts1,tracts2):
    maxi = 0
    res =[]
    for t in tracts1:
        if t[1]>maxi:
            maxi=t[1]
    for t in tracts2:
        if t[1]>maxi:
            maxi=t[1]
    b1=False
    b2=True
    inside=False
    start=-1
    for i in range(maxi+1):
        if (inTracts(i,tracts1)):
            b1=True
        else:
            b1=False
        if (inTracts(i,tracts2)):
            b2=False
        else:
            b2=True
        if (b1 and b2 and not inside):
            inside = True
            start = i
        elif( not(b1 and b2) and inside):
            inside = False
            res.append([start,i])

    if(b1 and b2 and inside):
        res.append([start,maxi])
    return res
        

def detected_tracts(tracts1,tracts2):
    tractsC1 = np.copy(tracts1)
    tractsC2 = np.copy(tracts2)
    isDetected = np.zeros(len(tractsC2))
    count=0
    acc=0
    for i in range(len(tractsC1)):
        j=0
        flag=True
        while flag and j < len(tractsC2):
            if tractsC1[i,0]<tractsC2[j,1] and tractsC1[i,1]>tractsC2[j,0]:
                diff = abs(tractsC1[i,0]-tractsC2[j,0])+abs(tractsC1[i,1]-tractsC2[j,1])
                common = min(tractsC1[i,1],tractsC2[j,1])-max(tractsC1[i,0],tractsC2[j,0])
                if common>4*diff:
                    flag=False
                    count=count+1
                    acc=acc+(common/(common+diff))
                    isDetected[j]=1
                    
            j=j+1
    acc=acc/count
    lenD=0
    lenND=0
    for i in range(len(tractsC2)):
        if isDetected[i]==1:
            lenD=lenD+tractsC2[i,1]-tractsC2[i,0]
        else:
            lenND=lenND+tractsC2[i,1]-tractsC2[i,0] 
        
    return(count,acc,len(tractsC1)-count,len(tractsC2)-count)

# Input:
#    seq: The result of the HMM algorithm
#    tracts: The actual tracts, in the same order as the states. Eg. if the state 0 correspond to non Archaic, the first tracts should correspond to non Archaic tracts.
def confusionMatrix(seq, tracts):
    nbState =len(tracts)
    M = np.zeros((nbState,nbState),dtype=int)
    for j in range(nbState):
        for t in tracts[j]:
            for i in range(t[0],t[-1]):
                M[seq[i],j]+=1
    return M

