# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            val = self.values.copy()
            for state in self.mdp.getStates():
                dict = {}
                for action in self.mdp.getPossibleActions(state):
                    sample = 0
                    for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
                        rewards = self.mdp.getReward(state, action, nextState)
                        value = val[nextState]
                        sample =sample+ prob * (rewards + self.discount * value) #calculating qvalue and so in final calculating max of this to get value

                    dict[action]=sample
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    self.values[state] = max(dict.values())


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Qval = 0
        for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
            rewards = self.mdp.getReward(state, action, nextState)
            value = self.values[nextState]
            Qval =Qval+ prob * (rewards + self.discount * value) #calculating qvalue of current state by taking all nextstate and transition values
        return Qval


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        act=self.mdp.getPossibleActions(state)
        if len(act)==0:
            return None
        dict={}
        for action in act:
            dict[action]=self.computeQValueFromValues(state,action)#calculating all qvalues and putting it ini dictionary with its actions so that for computing action we will take max of qvalue

        maxval=max(dict.values())
        for k in dict.keys():
            if dict[k]==maxval: #getting action where qvalue is maximum
                return k
        return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
