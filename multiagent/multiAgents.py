# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
from searchAgents import mazeDistance

import random
import util
import math

from game import Agent


class ReflexAgent(Agent):

    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        # for index, score in enumerate(scores):
        #   print score, legalMoves[index]
        # print "===>", legalMoves[chosenIndex]

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()


        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        ghostDistance = min(
            [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

        foodList = newFood.asList()
        foodDistance = 0
        if len(foodList) > 0:
            foodDistance = min(
                [manhattanDistance(newPos, food) for food in foodList])

        if ghostDistance < 3:
            return successorGameState.getScore() - ghostDistance
        else:
            return 2 * successorGameState.getScore() - foodDistance


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):

    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isMaximizing(self, agent):
        return agent == self.index

    def nextAgent(self, depth, agent):
        agent += 1

        if agent == self.numAgents:
            agent = self.index
            depth += 1

        return (depth, agent)


class MinimaxAgent(MultiAgentSearchAgent):

    """
      Your minimax agent (question 2)
    """

    def minimax(self, gameState, currentDepth=0, currentAgent=0):

        # terminal state
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        scores = []

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(currentAgent)

        # evaluate current state or go deeper if possible
        if currentDepth > self.depth:
            scores = [
                self.evaluationFunction(gameState) for action in legalMoves]
        else:
            nextAgent = currentAgent + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = self.index
                currentDepth += 1
            scores = [self.minimax(gameState.generateSuccessor(
                currentAgent, action), currentDepth, nextAgent) for action in legalMoves]

        # print currentDepth, currentAgent, actionScores

        # Choose action depending on agent
        if currentAgent == 0:  # pacman
            return max(scores)
        else:
            return min(scores)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        legalMoves = gameState.getLegalActions(self.index)

        actionScores = [(self.minimax(gameState.generateSuccessor(
            self.index, action), 1, 1), action) for action in legalMoves]

        return max(actionScores)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):

    """
      Your minimax agent with alpha-beta pruning (question 3)
    """




    def isTopLevel(self, depth, agent):
        return depth == 1 and agent == self.index

    def alphaBeta(self, gameState, depth, agent, alpha, beta):

        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)

        (nextDepth, nextAgent) = self.nextAgent(depth, agent)
        isTopLevel = self.isTopLevel(depth, agent)
        bestAction = None

        if self.isMaximizing(agent):

            bestValue = float("-inf")

            for action in gameState.getLegalActions(agent):
                state = gameState.generateSuccessor(agent, action)

                value = self.alphaBeta(state, nextDepth, nextAgent, alpha, beta)

                if value > bestValue:
                    bestValue = value
                    bestAction = action
                    if bestValue > alpha:
                        alpha = value

                #print "MAX", action, state.state, alpha

                if beta < alpha:
                    break

            if isTopLevel:
                return bestAction
            else:
                return bestValue

        else:

            bestValue = float("inf")

            for action in gameState.getLegalActions(agent):
                state = gameState.generateSuccessor(agent, action)

                value = self.alphaBeta(state, nextDepth, nextAgent, alpha, beta)

                if value < bestValue:
                    bestValue = value

                    if bestValue < beta:
                        beta = value
                #print "MIN", action, state.state, beta

                if beta < alpha:
                    break

            return bestValue

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction

        """

        self.numAgents = gameState.getNumAgents()

        return self.alphaBeta(gameState, 1, self.index, float("-inf"), float("inf"))



class ExpectimaxAgent(MultiAgentSearchAgent):

    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, gameState, depth, agent):

        # terminal state
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)

        scores = []

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(agent)

        (nextDepth, nextAgent) = self.nextAgent(depth, agent)
        scores = [self.expectimax(gameState.generateSuccessor(
            agent, action), nextDepth, nextAgent) for action in legalMoves]


        if self.isMaximizing(agent):

            return max(scores)
        else:
            return sum(scores) / float(len(legalMoves))


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        self.numAgents = gameState.getNumAgents()

        legalMoves = gameState.getLegalActions(self.index)

        actionScores = [(self.expectimax(gameState.generateSuccessor(
            self.index, action), 1, 1), action) for action in legalMoves]

      #  print actionScores

        return max(actionScores)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newCapsules = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
 #  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # food is worth 10
    # win is worth 500
    # eating a ghost: +200

    # parameters
    foodsToMeasure = 2
    tastyGhostDistance = 10
    nearbyCapsuleThreshold = 6

    # reward for eating nearby capsules
    nearbyCapsuleBonus = 0
    if len(newCapsules) > 0:
      nearestCapsuleDistance = min([manhattanDistance(newPos, capsule) for capsule in newCapsules])
      if nearestCapsuleDistance < nearbyCapsuleThreshold:
          nearbyCapsuleBonus = 6 - nearestCapsuleDistance

    # reward for being nearby scared ghosts
    nearbyGhostBonus = 0
    for ghost in newGhostStates:
        ghostDistance = manhattanDistance(newPos, ghost.getPosition())

        # is it close, and will it still be tasty?
        if ghostDistance < tastyGhostDistance and ghostDistance <= ghost.scaredTimer:
            nearbyGhostBonus = max(nearbyGhostBonus, tastyGhostDistance - ghostDistance)


    # reward for having a short path to the next x foods.
    foodList = newFood.asList()
    nearestFoodDistance = 0
    foodDistances = [(manhattanDistance(newPos, food), food) for food in foodList]
    if len(foodList) > 0:
        for food in sorted(foodDistances):

            nearestFoodDistance += mazeDistance(newPos, food[1], currentGameState)
            newPos = food[1]

            foodsToMeasure -= 1
            if foodsToMeasure < 1:
                break

    # scaling notes:
    # nearest food is slightly more important than pellets
    # nearby ghost bonus is a value 1-tastyGhostdistance where 5 could lead to 200.
    # nearby pellet is the most basic bonus
    return currentGameState.getScore() - 2 * nearestFoodDistance + 10 * nearbyGhostBonus + nearbyCapsuleBonus


# Abbreviation
better = betterEvaluationFunction
