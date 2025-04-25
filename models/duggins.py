import numpy as np
from models.model import Model
from utils import rand_gen
from scipy.spatial import cKDTree
import copy
from hyperopt import hp

# Peter Duggins
# August 11, 2016
# ISC Model

GRID_SIZE = 1000
ITERS = 10

def create_agents(n):
    """Create agents with random positions and social reaches. These will stay constant with the model."""

    agents = []
    x_positions = np.random.uniform(0, GRID_SIZE, n)
    y_positions = np.random.uniform(0, GRID_SIZE, n)
    # social_reaches = np.maximum(np.random.normal(22.0, 4.0, n), 0)

    # Create agents
    for i in range(n):
        agents.append(agent(
            iden=i,
            xpos=x_positions[i],
            ypos=y_positions[i],
            # radius=social_reaches[i]
        ))
    
    return agents

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
     
    def __init__(self,iden,xpos,ypos):
        # Created upon initialization
        self.iden = iden
        self.x = xpos
        self.y = ypos
        # self.radius = radius 

        # Not created upon initialization
        self.O = None
        self.int = None
        self.sus = None
        self.con = None
        self.radius = None
        self.commit = 1.0  
        self.network = []
        # self.E = o # Since it needs to be calculated

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

    def hold_dialogue(self):
        elist = []
        wlist = []
        elist.append(self.O)  # i initiates by speaking his true opinion USED TO BE E BUT O MAKES MORE SENSE
        wlist.append(1.0)  # placekeeper
        self.set_commit()  # calculate i's susceptibility
        np.random.shuffle(self.network)  # randomize order of speaking each dialogue
        for j in self.network:
            E = j.get_E(elist)  # each member of the dialogue calculates expressed
            elist.append(E)  # expressed is spoken to the dialogue
            wlist.append(self.get_w(E))  # i calculates interagent weight
        influence = self.get_influence(elist[1:], wlist[1:])  # calculate dialogue's influence
        self.set_O(influence)  # update opinion after dialogue
				
class DugginsModel(Model):

    def __init__(self, params=None, agents=None, n=1000):
        super().__init__(params)
        self.agents = agents
        # print(f"Duggins model created with parameters {self.params}")
        if agents is None:
            # print("No agents provided, creating new agents.")
            self.agents = create_agents(n)
    
    def sample_isc_for_agents(self, initial_opinions):

        # Use initial_opinions as is
        opinions = np.array(initial_opinions)

        p = self.params
        n = len(initial_opinions)

        p['std_conformity'] = 0.3
        p['std_intolerance'] = 0.3
        p['std_susceptibility'] = 0.7
        p['std_social_reach'] = 4.0

        # print(p)

        # Sample other parameters in bulk
        intolerances = np.maximum(np.random.normal(p['mean_intolerance'], p['std_intolerance'], n), 0)
        susceptibilities = np.maximum(np.random.normal(p['mean_susceptibility'], p['std_susceptibility'], n), 0)
        conformities = np.random.normal(p['mean_conformity'], p['std_conformity'], n)  # Conformity can be negative
        social_reaches = np.maximum(np.random.normal(p['mean_social_reach'], p['std_social_reach'], n), 0)

        # Update agents
        for i in range(n):
            self.agents[i].o = opinions[i]
            self.agents[i].int = intolerances[i]
            self.agents[i].sus = susceptibilities[i]
            self.agents[i].con = conformities[i]
            self.agents[i].radius = social_reaches[i]
            self.agents[i].commit = 1.0
            self.agents[i].network = []
        
        self.network_agents()

    def network_agents(self):
        #create social networks: if euclidian distance sqrt(dx^2+dy^2)<min(r_i,r_j), add to network
        for i in self.agents: # agentdict.itervalues():
            for j in self.agents: # agentdict.itervalues():
                if i != j and ((j.x - i.x)**2 + (j.y - i.y)**2)**(0.5) < min(i.radius,j.radius):
                    i.addtonetwork(j)
    
    def run(self, input):
		
        # Make sure agents have correct opinions
        for i, o in enumerate(input):
            self.agents[i].O = o
		
        for i in range(ITERS):
			
            # order = np.array(self.agentdict.keys())
            order = np.arange(len(self.agents))
			
            np.random.shuffle(order) #randomize order of dialogue initiation
            for i in order:
                self.agents[i].hold_dialogue()
		
        return np.array([agent.O for agent in self.agents])
    
    def get_random_params(self):
        """Get random feasible parameters for the model."""
        return {
            'mean_intolerance': np.random.uniform(0.7, 1.00),
            'mean_susceptibility':np.random.uniform(1, 5),
            'mean_conformity': np.random.uniform(0.1, 0.5),
            # 'std_intolerance': np.random.uniform(0.15, 0.45), # 0.3,
            # 'std_susceptibility': np.random.uniform(0.35, 1.05), # 0.7,
            # 'std_conformity': np.random.uniform(0.15, 0.45), # 0.3,
            'mean_social_reach': np.random.uniform(15.0, 30.0), # 22.0,
            # 'std_social_reach': np.random.uniform(2, 6), # 4
        }
    
    @staticmethod
    def get_model_name():
        """Return the name of the model."""
        return "duggins"
    
    @staticmethod
    def get_opinion_range():
        """Get the opinion range of the model. ie. the range of possible opinion values."""
        return (0, 100)
    
    def set_normalized_params(self, params):
        """
        The optimizer will return values between 0 and 1.
        This function will convert them to the actual parameter values.
        """
        self.params = {
            'mean_intolerance': 0.3 * params['mean_intolerance'] + 0.7,
            'mean_susceptibility': 4 * params['mean_susceptibility'] + 1,
            'mean_conformity': 0.4 * params['mean_conformity'] + 0.1,
            # 'std_intolerance': 0.15 + 0.3 * params['std_intolerance'],
            # 'std_susceptibility': 0.35 + 0.7 * params['std_susceptibility'],
            # 'std_conformity': 0.15 + 0.3 * params['std_conformity'],
            'mean_social_reach': 30.0 * params['mean_social_reach'],
            # 'std_social_reach': 2 + 4 * params['std_social_reach']
        }

    def get_cleaned_agents(self):
        agents_copy = copy.deepcopy(self.agents)
        for i in range(len(agents_copy)):
            agents_copy[i].o = None
            agents_copy[i].int = None
            agents_copy[i].sus = None
            agents_copy[i].con = None
            agents_copy[i].radius = None
            agents_copy[i].network = []
            agents_copy[i].commit = 1.0

        return agents_copy
    
    def create_fresh_duggins_model(self, initial_opinions):
        model = DugginsModel(params=self.params, agents=self.get_cleaned_agents())
        model.sample_isc_for_agents(initial_opinions)
        return model