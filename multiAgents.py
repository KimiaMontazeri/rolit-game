from Agents import Agent
import util
import random

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.index = 0 # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', **kwargs):
        self.index = 0 # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
    

    def minValue(self, state, depth, currentAgent):
        value = float("inf")
        legalMoves = state.getLegalActions(currentAgent)
        for action in legalMoves:
            successor = state.generateSuccessor(currentAgent, action)
            successorValue = self.value(successor, depth, currentAgent+1)
            value = min(successorValue, value)
        return value


    def maxValue(self, state, depth, currentAgent):
        value = float("-inf")
        legalMoves = state.getLegalActions(currentAgent)
        for action in legalMoves:
            successor = state.generateSuccessor(currentAgent, action)
            successorValue = self.value(successor, depth, currentAgent+1)
            value = max(successorValue, value)

            # In Minimax, we should store the first action that made us reach
            # the best possible state (since minimax only returns the best value!)
            if value == successorValue and depth == 1:
                self.action = action
        return value


    def value(self, state, depth=0, currentAgent=0):
        # Ensuring that agent's index is not out of range
        currentAgent = currentAgent % state.getNumAgents()

        # We reach a terminal state or max depth
        if depth == self.depth or state.isGameFinished():
            return self.evaluationFunction(state)

        # Actual Minimax Search
        if currentAgent == self.index: # It's my turn
            return self.maxValue(state, depth+1, currentAgent)
        else:
            return self.minValue(state, depth+1, currentAgent)


    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        But before getting your hands dirty, look at these functions:

        gameState.isGameFinished() -> bool
        gameState.getNumAgents() -> int
        gameState.generateSuccessor(agentIndex, action) -> GameState
        gameState.getLegalActions(agentIndex) -> list
        self.evaluationFunction(gameState) -> float
        """
        "*** YOUR CODE HERE ***"
        self.value(state)
        return self.action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    
    def minValue(self, state, depth, currentAgent, alpha, beta):
        value = float("inf")
        legalMoves = state.getLegalActions(currentAgent)
        for action in legalMoves:
            successor = state.generateSuccessor(currentAgent, action)
            successorValue = self.value(successor, depth, currentAgent+1, alpha, beta)
            value = min(successorValue, value)

            if value < alpha:
                return value
            
            beta = min(beta, value)
        return value


    def maxValue(self, state, depth, currentAgent, alpha, beta):
        value = float("-inf")
        legalMoves = state.getLegalActions(currentAgent)
        for action in legalMoves:
            successor = state.generateSuccessor(currentAgent, action)
            successorValue = self.value(successor, depth, currentAgent+1, alpha, beta)
            value = max(successorValue, value)

            # In Minimax, we should store the first action that made us reach
            # the best possible state (since minimax only returns the best value!)
            if value == successorValue and depth == 1:
                self.action = action
            
            if value > beta:
                return value
            
            alpha = max(alpha, value)
        return value


    def value(self, state, depth=0, currentAgent=0, alpha=float("-inf"), beta=float("inf")):
        # Ensuring that agent's index is not out of range
        currentAgent = currentAgent % state.getNumAgents()

        # We reach a terminal state or max depth
        if depth == self.depth or state.isGameFinished():
            return self.evaluationFunction(state)

        # Actual Minimax Search
        if currentAgent == self.index: # It's my turn
            return self.maxValue(state, depth+1, currentAgent, alpha, beta)
        else:
            return self.minValue(state, depth+1, currentAgent, alpha, beta)


    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        "*** YOUR CODE HERE ***"
        self.value(gameState)
        return self.action
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    Here are also some functions you will need to use:
    
    gameState.getPieces(index) -> list
    gameState.getCorners() -> 4-tuple
    gameState.getScore() -> list
    gameState.getScore(index) -> int

    """
    
    "*** YOUR CODE HERE ***"

    # parity

    # corners

    # mobility

    # stability
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction