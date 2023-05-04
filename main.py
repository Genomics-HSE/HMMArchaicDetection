import msprime
import tskit
import core
import HMM
import random
import numpy as np
import gtts
import time
import sys
from importlib import reload
from IPython.display import SVG, display
from gtts import gTTS

pop=1

demography = msprime.Demography()
demography.add_population(name="O", initial_size=100000*pop) # archaic humans
demography.add_population(name="S", initial_size=100000*pop) # homo sapiens
demography.add_population(name="ND", initial_size=100000*pop) # Neandertal + Denisovian
demography.add_population(name="N", initial_size=1000000*pop) # Neandertal
demography.add_population(name="D", initial_size=100000*pop) # Denisovian
demography.add_population(name="Ne", initial_size=1000000*pop) # We assume a Neandertal split into three groups. Ne will introgress into Europe,
demography.add_population(name="Na", initial_size=1000000*pop) # Na into Asia,
demography.add_population(name="Ng", initial_size=1000000*pop) # Ng into both Europe and Asia
demography.add_population(name="Af", initial_size=1000000*pop) # Africa
demography.add_population(name="EAs", initial_size=100000*pop) # Eurasians  before Neandertal (Ng) introgression 
demography.add_population(name="EAs2", initial_size=100000*pop) # Eurasians  after Neandertal (Ng) introgression 
demography.add_population(name="E", initial_size=1000000*pop) # Europeans before Neandertal (Ne) introgression 
demography.add_population(name="As", initial_size=1000000*pop) # Asians before Neandertal (Na) introgression 
demography.add_population(name="E2", initial_size=1000000*pop) # Europe after Neandertal (Ne) introgression 
demography.add_population(name="As2", initial_size=1000000*pop) # Asians after Neandertal (Na) introgression 


demography.add_admixture(time=1000, derived="E2", ancestral=["E", "Ne"], proportions=[0.97, 0.03])
demography.add_admixture(time=1000, derived="As2", ancestral=["As", "Na"], proportions=[0.97,0.03])
demography.add_population_split(time=1100, derived=["E", "As"], ancestral="EAs2") 
demography.add_admixture(time=1850, derived="EAs2", ancestral=["EAs","Ng"], proportions=[0.97, 0.03]) 
demography.add_population_split(time=3700, derived=["Af", "EAs"], ancestral="S")
demography.add_population_split(time=8000, derived=["Ne", "Na","Ng"], ancestral="N") 
demography.add_population_split(time=16000, derived=["N", "D"], ancestral="ND")
demography.add_population_split(time=24500, derived=["S", "ND"], ancestral="O")


nbAf = 250 #Number of African samples that are going to be used in the HMM
nbAs2 = 250 #Number of Asian samples that are going to be used in the HMM
nbN = 0  #Number of Neanderthal samples that are going to be used in the HMM
seq_len=20000000

ts = msprime.sim_ancestry({"E2": 1,"Af":nbAf,"As2": nbAs2, "N": nbN }, ploidy=1, sequence_length=seq_len,recombination_rate=2.5e-9, demography=demography,record_migrations=True, random_seed=34299)

ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=4319)    


# --- Run the HMM ---

seq=core.createSeqObs(ts,1000,0,1,nbAf,nbAs2) #Compute the observations

S = HMM.initS(0.03)
A = HMM.initA(1850,1000,2.5e-9,1000,0.03)
B = HMM.initB(1.25e-8,1000,3700,24500,1100,1850)
states = (0,1,2)


resViterbi =  HMM.viterbi(seq, S, A,B)


tractsHMM = core.get_HMM_tracts(resViterbi) 

tractsNe = core.clean_tracts(core.get_migrating_tracts(ts,'Ne'),1000)
tractsNg = core.get_migrating_tracts_ind(ts,"Ng",0,1000)
tractsAf = core.substract_tracts([[0,seq_len//1000]],np.concatenate((tractsNg, tractsNe)))

M = core.confusionMatrix(resViterbi,[tractsAf,tractsNg,tractsNe])

