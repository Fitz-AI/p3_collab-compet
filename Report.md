
# Tony Quertier - Project 3: Collaboration and Competition


## Examine the State and Action Spaces

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.his yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


##  Train the agent


### Algorithm 1

To train the agent, we use the ddpg algorithm form DDPG_continuous_control notebook. 
DDPG (Lillicrap, et al., 2015), short for Deep Deterministic Policy Gradient, is a model-free off-policy actor-critic algorithm, combining DPG with DQN.The original DQN works in discrete space, and DDPG extends it to continuous space with the actor-critic framework while learning a deterministic policy.

In order to do better exploration, an exploration policy μ’ is constructed by adding noise N :μ′(s)=μθ(s)+N.

In addition, DDPG does soft update on the parameters of both actor and critic, with τ≪1: θ′←τθ+(1−τ)θ′. In this way, the target network values are constrained to change slowly, different from the design in DQN that the target network stays frozen for some period of time.


![alt text](https://github.com/Quertier/p3_collab-compet/blob/master/pics/ddpg_algo.jpg)


### Algorithm 2

Multi-agent DDPG (MADDPG) (Lowe et al., 2017)extends DDPG to an environment where multiple agents are coordinating to complete tasks with only local information. In the viewpoint of one agent, the environment is non-stationary as policies of other agents are quickly upgraded and remain unknown. MADDPG is an actor-critic model redesigned particularly for handling such a changing environment and interactions between agents.

In summary, MADDPG added three additional ingredients on top of DDPG to make it adapt to the multi-agent environment:

   -Centralized critic + decentralized actors;
   
   -Actors are able to use estimated policies of other agents for learning;
   
   -Policy ensembling is good for reducing variance.


![alt text](https://github.com/Quertier/p3_collab-compet/blob/master/pics/maddpg_algo.jpg)

### Hyper parameters for DDPG

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 256        
GAMMA = 0.99            
TAU = 5e-3              
LR_ACTOR = 1e-3         
LR_CRITIC = 1e-3        
WEIGHT_DECAY = 0.0
EPSILON = 1.0 
EPSILON_DECAY = 1e-6

### Hyper parameters for MADDPG

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 256        
GAMMA = 0.99           
TAU = 5e-3              
LR_ACTOR = 1e-3         
LR_CRITIC = 1e-3        
WEIGHT_DECAY = 0.0 
EPSILON = 1.0 
EPSILON_DECAY = 1e-6

### Neural Networks for DDPG

Actor and Critic network models were defined in model.py.

The Actor networks utilised two fully connected layers with 256 and 128 units with relu activation and tanh activation for the action space. 
The Critic networks utilised two fully connected layers with 256 and 128 units with relu activation. 

### Neural Networks for MADDPG

Actor and Critic network models were defined in maddpg_model.py.

The Actor networks utilised two fully connected layers with 256 and 256 units with relu activation and tanh activation for the action space. 
The Critic networks utilised two fully connected layers with 256 and 256 units with relu activation. 



## Performance of the agent

For DDPG, we have :

![Alt text](https://github.com/Quertier/p3_collab-compet/blob/master/pics/p3_score.PNG)

For MADDPG, we want to have a higher score then we fix a threshold of 2.0. :

![Alt text](https://github.com/Quertier/p2_continuous-control/blob/master/p3_maddpg.PNG)


## Future Improvements

As future works, we could replace DDPG with Distributed Distributional Deterministic Policy Gradients (D4PG) [https://arxiv.org/pdf/1804.08617.pdf]. We could also implement Rainbow Algorithm [https://arxiv.org/pdf/1710.02298.pdf] to compare the results with MADDPG.



