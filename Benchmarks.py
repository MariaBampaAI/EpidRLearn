import os
import matplotlib.pyplot as plt
import numpy as np
import env_realCase
import math 
if not os.path.exists('Figures'):
    os.mkdir("Figures")    


"""

Function that defines the deterministic policies to be compared with the AI policies


"""

def policy_Sweden(actions, week, state):

    """
    Sweden ECDC: mild lockdown policy
    """

    free_to_move = actions[0]
    _75_lockdown = actions[1]
    _50_lockdown = actions[2]
    _25_lockdown = actions[3]


    #if(week <= 22) :
        #print("Action chosen: _25_lockdown ")
        #return _25_lockdown

    if(week<5):
        #print("Action chosen: _50_lockdown")
        return _25_lockdown
    if(week>=5):
        #print("Action chosen: _50_lockdown")
        return _50_lockdown


def policy_Sweden_second(actions, week, state):

    """
    Sweden Fitted: the action here is always free_to_move
    as this refers to the fitted version and the actions are derived from the contact reduction matrix
    
    """
    free_to_move = actions[0]
    _75_lockdown = actions[1]
    _50_lockdown = actions[2]
    _25_lockdown = actions[3]


    if(week>=0):
        #print("Action chosen: _50_lockdown")
        return free_to_move




def policy_Greece(actions, week, state):

    """
    Greece ECDC: strict lockdown policy
    """

    free_to_move = actions[0]
    _75_lockdown = actions[1]
    _50_lockdown = actions[2]
    _25_lockdown = actions[3]

    
    if(week<2):
        return _50_lockdown


    if(week>=2):
        return _75_lockdown




def evaluate_deterministic_model(reward_id, problem_id, policy_chosen, period='autumn'):
    

    problems = [0]
    rewards_per_problem =  []
    for problem in problems:
    #envr = env_realCase.Epidemic(problem_id = problem, reward_id=reward_id)
        if period == 'autumn':
            #print('autUMNN')
            envr = env_realCase.Epidemic(problem_id = problem_id, reward_id=reward_id, period = 'autumn')
        elif period == 'FOHM':
            #print('fohm')
            envr = env_realCase.Epidemic(problem_id = problem_id, reward_id=reward_id, period = 'FOHM')


        #print("Observations/States: " + str(envr.observation_space))
        #print("Actions: " + str(envr.action_space))
        #print("Rewards: " + str(envr.reward_range))
        
        states = []
        rewards = []
        done = False
        
        state = envr.reset()
        """
        print("\nInitial state: " + "\nSusceptible: " + str(state[0]) + "\nExposed: " + str(state[1])
              + "\nInfected: " + str(state[2]) + "\nRecovered: " + str(state[3]) + "\nDead: " + str(state[4]))
        """
        states.append(state)
        
        actions = [0, 1, 2, 3]

        week = 0
        while not done:
            if policy_chosen == 'policy_Sweden':
                action = policy_Sweden(actions,week,state)
                state,r,done,i= envr.step(action = (action))
                states.append(state)

                rewards.append(r)
                week+=1
            
            elif policy_chosen == 'policy_Sweden_second':

                action = policy_Sweden_second(actions,week,state)
                
                state,r,done,i= envr.step(action = (action))

                states.append(state)

                rewards.append(r)
                week+=1
            
            elif policy_chosen == 'policy_Greece':

                action = policy_Greece(actions,week,state)
                state,r,done,i= envr.step(action = (action))
                states.append(state)
                rewards.append(r)
                week+=1


        #print('Sum of rewards reward', np.sum(rewards))
        rewards_per_problem.append(np.sum(rewards))
    return rewards_per_problem, rewards, states


