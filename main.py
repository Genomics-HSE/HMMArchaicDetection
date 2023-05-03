import msprime
import tskit
import core
import parse
import HMM
import HMMS
import HMMSND
import random
import numpy as np
import gtts
import time
import sys
from importlib import reload
from IPython.display import SVG, display
from playsound import playsound


from gtts import gTTS

np.set_printoptions(threshold=sys.maxsize)


reload(core)
reload(HMM)

random.seed()

print("C'est le debut")
start_time = time.time()

pop=1

demography = msprime.Demography()
demography.add_population(name="O", initial_size=100000*pop) # archaic humans
demography.add_population(name="S", initial_size=100000*pop) # homo sapiens
demography.add_population(name="ND", initial_size=100000*pop) # Neandertal + Denisovian
demography.add_population(name="Ne", initial_size=1000000*pop) # Neandertal
demography.add_population(name="Na", initial_size=1000000*pop)
demography.add_population(name="Ng", initial_size=1000000*pop)  #test 
demography.add_population(name="N", initial_size=1000000*pop)
demography.add_population(name="D", initial_size=100000*pop)
demography.add_population(name="Af", initial_size=1000000*pop)
demography.add_population(name="EAs", initial_size=100000*pop)
demography.add_population(name="EAs2", initial_size=100000*pop)
demography.add_population(name="E", initial_size=1000000*pop)
demography.add_population(name="As", initial_size=1000000*pop)
demography.add_population(name="E2", initial_size=1000000*pop) # Modern europeans
demography.add_population(name="As2", initial_size=1000000*pop)


demography.add_admixture(time=1000, derived="E2", ancestral=["E", "Ne"], proportions=[0.95, 0.05])#Hyp 1 - 1000?
demography.add_admixture(time=1000, derived="As2", ancestral=["As", "Na"], proportions=[0.95,0.05])#Hyp 1
demography.add_population_split(time=1100, derived=["E", "As"], ancestral="EAs2") 
demography.add_admixture(time=1850, derived="EAs2", ancestral=["EAs","Ng"], proportions=[0.95, 0.05]) 
demography.add_population_split(time=3700, derived=["Af", "EAs"], ancestral="S")
demography.add_population_split(time=8000, derived=["Ne", "Na","Ng"], ancestral="N") #Hyp 1 - 3701?
demography.add_population_split(time=16000, derived=["N", "D"], ancestral="ND")
demography.add_population_split(time=24500, derived=["S", "ND"], ancestral="O")

# demography.add_population_split(time=1100, derived=["E2", "As2"], ancestral="EAs") 
# demography.add_mass_migration(time=1850, source='EAs', dest='N', proportion=0.05)
# demography.add_population_split(time=3700, derived=["Af", "EAs"], ancestral="S")
# demography.add_population_split(time=24500, derived=["S", "N"], ancestral="O")



nbAf = 250
nbAs2 = 250
nbN = 0

ts = msprime.sim_ancestry({"E2": 1,"Af":nbAf,"As2": nbAs2, "N": nbN },     ploidy=1,    sequence_length=20000000,recombination_rate=2.5e-9, demography=demography,record_migrations=True, random_seed=34299)
#Rec rate 2.5e-9

ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=4319)    


# for v in ts.variants():
#     site = v.site
#     alleles = np.array(v.alleles)
#     print(f"Site {site.position} {int(ts.tables.mutations[site.id].time)} (ancestral state '{site.ancestral_state}')",  alleles[v.genotypes])
#     if site.id >= 8:  # only print up to site ID 4
#         print("...")
#         break 




#  -------- With conditional probabilities ----------

seq=core.createSeqObs(ts,1000,0,1,nbAf,nbAs2)

S = HMM.initS(0.05)
A = HMM.initA(1850,1000,2.5e-9,1000,0.05)
B = HMM.initB(1.25e-8,1000,3700,24500,1100,1850)
states = (0,1,2)


resV =  HMM.viterbi(seq, S, A,B)


tractsHMM2 = core.get_HMM_tracts(resV)

tractsNe = core.clean_tracts(core.get_migrating_tracts(ts,'Ne'),1000)

tractsNg = core.get_migrating_tracts_ind(ts,"Ng",0,1000)



#  ----------- Without conditional probabilites ------------
# print("Without conditional")


# seqAf = core.createSeqObsS(ts,1000,0,"Af",nbAf)
# seqAs = core.createSeqObsS(ts,1000,0,"As2",nbAs2)


# S = HMM.initS(0.05)
# A = HMM.initA(1850,1000,2.5e-9,1000,0.05)
# B = HMM.initBNotCond(1.25e-8,1000,3700,24500,1100,8000)
# states = (0,1,2)
    
# seq=np.column_stack((seqAf,seqAs))

# res =  HMM.viterbi(seq, S, A,B)

# tractsHMMnotCond = core.get_HMM_tracts(res)



# tracts = core.get_migrating_tracts(ts,'N')
# tractsNe = core.clean_tracts(core.get_migrating_tracts(ts,'Ne'),1000)

# tracts2 = core.get_migrating_tracts_ind_all(ts,['Ng','Ne'],0)


#print(core.get_tract_length(tractsHMM2[0])/(30000))
#print(core.get_tract_length(tractsHMM2[1])/(30000))
#print(core.get_tract_length(tractsHMM2[2])/(30000))



# ------------  HMMS ----------------

# states = [0,1]
# S = HMMS.initS(0.95)
# A = HMMS.initA(1850,2.5e-9,1000,0.05)
# B = HMMS.initB(1.25e-8,1000,3700,24500)

# seqAf = core.createSeqObsS(ts,1000,0,"Af",nbAf)
# res = HMMS.viterbi(seqAf, S, A, B)
# tractsS = core.get_HMM_tracts(res)

# ratioS = core.get_tract_length(tractsS[1])/(core.get_tract_length(tractsS[0])+core.get_tract_length(tractsS[1]))


# ------------  HMMS ----------------

# ------------  HMMSND ----------------

# states = [0,1]
# S = HMMSND.initS(0.95)
# A = HMMSND.initA(1850,2.5e-9,1000,0.05)
# B = HMMSND.initB(1.25e-8,1000,3700,24500)

# seqNd = core.createSeqObsS(ts,1000,0,"N",nbN)
# seq=np.column_stack((seqAf,seqNd))
# res = HMMSND.viterbi(seq, S, A, B)
# tracts = core.get_HMM_tracts(res)

# ratio = core.get_tract_length(tracts[1])/(core.get_tract_length(tracts[0])+core.get_tract_length(tracts[1]))



# ------------  HMMSND ----------------

# tractsND = core.get_migrating_tracts(ts,"N")
# tractsC = core.clean_tracts(tractsND,1000)

# seqAf = core.createSeqObsS(ts,1000,0,"Af",nbAf)
# tracts2 = core.get_migrating_tracts_ind_all(ts,['Ng','Ne'],0)
# allMut =0
# allMut2 =0
# for t2 in tracts2:
#     for t in t2:
#         for i in range(t[0],t[1]):
#             allMut=allMut+seqAf[i]

#     allMut2 =allMut2+ core.get_tract_length(t2)*1.25e-8*24500*1000

# res2=HMM.training(seqAll)
# resV2 = HMM.viterbi(states, seqAll, res[0], res[1], res[2])
mytext="Hello"
language='en'
myobj=gTTS(text=mytext,lang=language,slow=True)
myobj.save("welcome1.mp3")
playsound("welcome1.mp3")


