import numpy as np
from models.model import Model
from utils import rand_gen

# Peter Duggins
# August 11, 2016
# ISC Model

import numpy as np
class agent:

	def __init__(self,iden,xpos,ypos,o,intol,sus,con,radius):
		self.iden = iden
		self.x = xpos
		self.y = ypos
		self.O = o
		self.E = o
		self.int = intol
		self.sus = sus
		self.con = con
		self.commit = 1.0
		self.radius=radius
		self.network=[]

	def set_E(self,elist):
		if len(elist) == 0.0: posturing = 0.0 #if nobody has yet spoken in dialogue, no falsification
		else: posturing = np.mean(elist) - self.O #O_j is a constant, so pull out of sum
		self.E = self.O + self.con/self.commit * posturing 			#'''EQUATION 1'''
		if self.E < 0.0: self.E = 0.0 #truncate below 0 or above 100
		elif self.E > 100.0: self.E = 100.0

	def set_commit(self):
		self.commit = 1.0 + self.sus * (abs(50.0 - self.O) / 50.0) 	#'''EQUATION 2'''

	def set_influence(self,elist,wlist):
		if len(self.network) == 0.0: return 0.0 #stop if nobody in network
		sum_influence,sum_weight = 0.0,0.0
		for j in range(len(elist)):
			sum_influence += wlist[j] * (elist[j] - self.O) 		#'''EQUATION 3'''
			sum_weight += abs(wlist[j])
		if sum_weight != 0.0:
			influence = sum_influence / sum_weight
		else: influence = 0.0
		return influence

	def set_w(self,E_j):
		return 1 - self.int * abs(E_j - self.O)/50.0 				#'''EQUATION 4'''

	def set_O(self,influence):
		self.O = self.O + 0.1 * influence / self.commit 			#'''EQUATION 5'''
		#the 0.1 above slows the rate of opinion change but can be omitted
		if self.O < 0.0: self.O = 0.0 #truncate opinions below 0 or over 100
		elif self.O > 100.0: self.O = 100.0

	def addtonetwork(self,other):
		self.network.append(other)

	def hold_dialogue(self,rng):
		elist=[]
		wlist=[]
		elist.append(self.E) #i initiates by speaking his true opinion
		wlist.append(1.0) #placekeeper
		self.set_commit() #calculate i's susceptibility
		rng.shuffle(self.network) #randomize order of speaking each dialogue
		for j in self.network:
			j.set_E(elist) #each member of the dialogue calculates expressed
			elist.append(j.E) #expressed is spoken to the dialogue
			wlist.append(self.set_w(j.E)) #i calculates interagent weight
		influence=self.set_influence(elist[1:],wlist[1:]) #calculate dialogue's influence
		self.set_O(influence) #update opinion after dialogue

def create_agents(P,rng):
	#initialize agents' internal parameters: initial opinion, intolerance, susceptibility,
	#conformity, social reach. Truncate parameters below zero (or over 100 for O_i)
	# from agent import agent
	agentdict={}
	for i in range(P['popsize']):
		x=rng.uniform(0,P['gridsize'])
		y=rng.uniform(0,P['gridsize'])
		if P['std_init_opinion'] != 0: #opinion
			o_i=rng.normal(P['mean_init_opinion'],P['std_init_opinion'])
			if o_i<0: o_i=0
			if o_i>100: o_i=100
		else: o_i=P['mean_init_opinion']
		if P['std_intolerance'] != 0: #intolerance
			t_i=rng.normal(P['mean_intolerance'],P['std_intolerance'])
			if t_i<0: t_i=0
		else: t_i=P['mean_intolerance']
		if P['std_susceptibility'] != 0: #susceptibility
			s_i=rng.normal(P['mean_susceptibility'],P['std_susceptibility'])
			if s_i<0: s_i=0
		else: s_i=P['mean_susceptibility']
		if P['std_conformity'] != 0: #conformity
			s_i=rng.normal(P['mean_conformity'],P['std_conformity'])
			#if s_i<0: s_i=0 #negative implies anticonformity / distinctiveness
			#if s_i>1: s_i=1 #over 1 implies overshooting the group norm in the effort to conform
		else: s_i=P['mean_conformity']
		if P['std_social_reach'] != 0: #social reach
			r_i=rng.normal(P['mean_social_reach'],P['std_social_reach'])
			if r_i<0: r_i=0
		else: r_i=P['mean_social_reach']
		#create an agent with these parameters and add it to the agent dictionary
		agentdict[i]=agent(i,x,y,o_i,t_i,s_i,s_i,r_i)
	return agentdict	

def network_agents(agentdict):
	#create social networks: if euclidian distance sqrt(dx^2+dy^2)<r_i, add to network
	for i in agentdict.itervalues():
		for j in agentdict.itervalues():
			if i != j and ((j.x - i.x)**2 + (j.y - i.y)**2)**(0.5) < min(i.radius,j.radius):
				i.addtonetwork(j)
				
class DugginsModel(Model):

    def __init__(self, params=None):
        super().__init__(params)
		
    def create_agents(self, seed=None):
        self.rng = np.random.RandomState(seed=seed) #set the simulation seed
        self.agentdict = create_agents(self.params, self.rng)
        network_agents(self.agentdict)
    
    def run(self, input):
        """
        """
        
        p = self.params
		
        for i in range(p['iters']):
            order=np.array(self.agentdict.keys())
            self.rng.shuffle(order) #randomize order of dialogue initiation
            for i in order:
                self.agentdict[i].hold_dialogue(self.rng)
			
        return
    
    def get_random_params(self):
        """Get random feasible parameters for the model."""
        return {
            'shift_amount': np.random.uniform(0, 0.06),
            'flip_prob': np.random.uniform(0, 0.08),
            'mob_min': np.random.uniform(0, 0.10),
            'mob_max': np.random.uniform(0.15, 0.30),
            'iterations': np.random.randint(1000, 5000)
        }
    
    def get_opinion_range(self):
        """Get the opinion range of the model. ie. the range of possible opinion values."""
        return (-1, 1)
    
    def set_normalized_params(self, params):
        """
        The optimizer will return values between 0 and 1.
        This function will convert them to the actual parameter values.
        """
        self.params = {
            'shift_amount': 0.06 * params['shift_amount'],
            'flip_prob': 0.08 * params['flip_prob'],
            'mob_min': 0.10 * params['mob_min'],
            'mob_max': 0.15 * params['mob_max'] + 0.15,
            'iterations': int(5000 * params['iterations'])
        }