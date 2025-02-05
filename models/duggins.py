import numpy as np
from models.model import Model
from utils import rand_gen

# Peter Duggins
# August 11, 2016
# ISC Model

POP_SIZE = 1000
GRID_SIZE = 1000
ITERS = 10

def keep_in_range(value):
    """
    Keep a value in the opinion range [0, 100] by reflecting out-of-range values back into the range.
    """
    if value < 0:
        return (-value)
    if value > 100:
        return 200 - value
    return value

import numpy as np
class agent:

	def __init__(self,iden,xpos,ypos,o,intol,sus,con,radius):
		self.iden = iden
		self.x = xpos
		self.y = ypos
		self.O = o
		# self.E = o # Since it needs to be calculated
		self.int = intol
		self.sus = sus
		self.con = con
		self.commit = 1.0
		self.radius = radius
		self.network = []

	def get_E(self,elist):
		if len(elist) == 0.0: posturing = 0.0 #if nobody has yet spoken in dialogue, no falsification
		else: posturing = np.mean(elist) - self.O #O_j is a constant, so pull out of sum
		E = self.O + self.con/self.commit * posturing 			#'''EQUATION 1'''
		return keep_in_range(E) #keep E in the opinion range [0,100]

	def set_commit(self):
		self.commit = 1.0 + self.sus * (abs(50.0 - self.O) / 50.0) 	#'''EQUATION 2'''

	def get_influence(self,elist,wlist):
		if len(self.network) == 0.0: return 0.0 #stop if nobody in network
		sum_influence,sum_weight = 0.0,0.0
		for j in range(len(elist)):
			sum_influence += wlist[j] * (elist[j] - self.O) 		#'''EQUATION 3'''
			sum_weight += abs(wlist[j])
		if sum_weight != 0.0:
			influence = sum_influence / sum_weight
		else: influence = 0.0
		return influence

	def get_w(self, E_j):
		return 1 - self.int * abs(E_j - self.O)/50.0 				#'''EQUATION 4'''

	def set_O(self, influence):
		self.O = self.O + 0.1 * influence / self.commit 			#'''EQUATION 5'''
		#the 0.1 above slows the rate of opinion change but can be omitted
		self.O = keep_in_range(self.O) #keep O in the opinion range [0,100]

	def addtonetwork(self,other):
		self.network.append(other)

	def hold_dialogue(self, rng):
		elist = []
		wlist = []
		elist.append(self.O)  # i initiates by speaking his true opinion USED TO BE E BUT O MAKES MORE SENSE
		wlist.append(1.0)  # placekeeper
		self.set_commit()  # calculate i's susceptibility
		rng.shuffle(self.network)  # randomize order of speaking each dialogue
		for j in self.network:
			E = j.get_E(elist)  # each member of the dialogue calculates expressed
			elist.append(E)  # expressed is spoken to the dialogue
			wlist.append(self.get_w(E))  # i calculates interagent weight
		influence = self.get_influence(elist[1:], wlist[1:])  # calculate dialogue's influence
		self.set_O(influence)  # update opinion after dialogue

def create_agents(P, initial_opinions, x_positions, y_positions, rng):
    agentdict = {}
    
    # Use initial_opinions as is
    opinions = np.array(initial_opinions)

    # Sample other parameters in bulk
    intolerances = np.maximum(rng.normal(P['mean_intolerance'], P['std_intolerance'], P['popsize']), 0)
    susceptibilities = np.maximum(rng.normal(P['mean_susceptibility'], P['std_susceptibility'], P['popsize']), 0)
    conformities = rng.normal(P['mean_conformity'], P['std_conformity'], P['popsize'])  # Conformity can be negative
    social_reaches = np.maximum(rng.normal(22.0, 4.0, P['popsize']), 0)

    # Create agents
    for i in range(P['popsize']):
        agentdict[i] = agent(
            iden=i,
            xpos=x_positions[i],
            ypos=y_positions[i],
            o=opinions[i],
            intol=intolerances[i],
            sus=susceptibilities[i],
            con=conformities[i],
            radius=social_reaches[i]
        )
    
    return agentdict

from scipy.spatial import cKDTree

def network_agents(agentdict):
    positions = np.array([[agent.x, agent.y] for agent in agentdict.values()])
    radii = np.array([agent.radius for agent in agentdict.values()])
    
    # Build a k-d tree for efficient neighbor lookup
    tree = cKDTree(positions)
    
    for i, agent in enumerate(agentdict.values()):
        neighbors_idx = tree.query_ball_point(positions[i], agent.radius)
        agent.network = [agentdict[j] for j in neighbors_idx if j != i]

def update_agents(agentdict, initial_opinions, P, rng):

    # Use initial_opinions as is
    opinions = np.array(initial_opinions)

    # Sample other parameters in bulk
    intolerances = np.maximum(rng.normal(P['mean_intolerance'], P['std_intolerance'], P['popsize']), 0)
    susceptibilities = np.maximum(rng.normal(P['mean_susceptibility'], P['std_susceptibility'], P['popsize']), 0)
    conformities = rng.normal(P['mean_conformity'], P['std_conformity'], P['popsize'])  # Conformity can be negative

    # Update agents
    for i in range(P['popsize']):
        agentdict[i].o = opinions[i]
        agentdict[i].intol = intolerances[i]
        agentdict[i].sus = susceptibilities[i]
        agentdict[i].con = conformities[i]
    
    return agentdict


# def network_agents(agentdict):
# 	#create social networks: if euclidian distance sqrt(dx^2+dy^2)<r_i, add to network
# 	for i in agentdict.values(): # agentdict.itervalues():
# 		for j in agentdict.values(): # agentdict.itervalues():
# 			# if i != j and ((j.x - i.x)**2 + (j.y - i.y)**2)**(0.5) < min(i.radius,j.radius):
# 			if i != j and ((j.x - i.x)**2 + (j.y - i.y)**2)**(0.5) < i.radius: # changed to i.radius
# 				i.addtonetwork(j)
				
class DugginsModel(Model):

    def __init__(self, params=None):
        super().__init__(params)
        self.agentdict = None
		
    def set_positions(self, x_positions, y_positions):
        self.x_positions = x_positions
        self.y_positions = y_positions
		
    def create_agents(self, initial_opinions):
        self.rng = np.random.RandomState(seed=None)
        self.agentdict = create_agents(self.params, initial_opinions, self.x_positions, self.y_positions, self.rng)
        network_agents(self.agentdict)
        return self.agentdict

    def update_agents(self, agentdict, initial_opinions):   
        if agentdict is None:
             agentdict = self.agentdict
        self.rng = np.random.RandomState(seed=None)
        self.agentdict = update_agents(agentdict, initial_opinions, self.params, self.rng)
			
    # def set_agentdict(self, agentdict):
    #     self.agentdict = agentdict
    
    def run(self, input):
        
        p = self.params
		
        # Make sure agents have correct opinions
        for i, o in enumerate(input):
            self.agentdict[i].O = o
		
        for i in range(p['iters']):
			
            # order = np.array(self.agentdict.keys())
            order = np.array(list(self.agentdict.keys()))
			
            self.rng.shuffle(order) #randomize order of dialogue initiation
            for i in order:
                self.agentdict[i].hold_dialogue(self.rng)
		
        return [agent.O for agent in self.agentdict.values()]
    
    def get_random_params(self):
        """Get random feasible parameters for the model."""
        return {
			'popsize': POP_SIZE,
			'gridsize': GRID_SIZE,
			'iters': ITERS,
			# 'mean_init_opinion': 50,
            # 'std_init_opinion': 50,
            'mean_intolerance': np.random.uniform(0.4, 1.2), # 0.8,
            'mean_susceptibility': np.random.uniform(2.5, 7.5), # 5.0,
            'mean_conformity': np.random.uniform(0.15, 0.45), # 0.3,
            'std_intolerance': np.random.uniform(0.15, 0.45), # 0.3,
            'std_susceptibility': np.random.uniform(0.35, 1.05), # 0.7,
            'std_conformity': np.random.uniform(0.15, 0.45), # 0.3,
            # 'mean_social_reach': np.random.uniform(10.0, 30.0), # 22.0,
            # 'std_social_reach': np.random.uniform(2, 6), # 4
        }
    
    def get_opinion_range(self):
        """Get the opinion range of the model. ie. the range of possible opinion values."""
        return (0, 100)
    
    def set_normalized_params(self, params):
        """
        The optimizer will return values between 0 and 1.
        This function will convert them to the actual parameter values.
        """
        self.params = {
            'popsize': POP_SIZE,
			'gridsize': GRID_SIZE,
			'iters': ITERS,
			# 'mean_init_opinion': 50,
            # 'std_init_opinion': 50,
            'mean_intolerance': 0.4 + 0.8 * params['mean_intolerance'],
            'mean_susceptibility': 2.5 + 5.0 * params['mean_susceptibility'],
            'mean_conformity': 0.15 + 0.3 * params['mean_conformity'],
            'std_intolerance': 0.15 + 0.3 * params['std_intolerance'],
            'std_susceptibility': 0.35 + 0.7 * params['std_susceptibility'],
            'std_conformity': 0.15 + 0.3 * params['std_conformity'],
            # 'mean_social_reach': 15.0 + 20.0 * params['mean_social_reach'],
            # 'std_social_reach': 2 + 4 * params['std_social_reach']
        }