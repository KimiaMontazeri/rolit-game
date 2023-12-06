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

    
    def expectValue(self, state, depth, currentAgent):
        value = 0
        legalMoves = state.getLegalActions(currentAgent)
        # We take average on all the legal moves
        legalMoveProbability = 1.0 / len(legalMoves)
        for action in legalMoves:
            successor = state.generateSuccessor(currentAgent, action)
            successorValue = self.value(successor, depth, currentAgent+1)
            value += legalMoveProbability * successorValue
        return value


    def maxValue(self, state, depth, currentAgent):
        value = float("-inf")
        legalMoves = state.getLegalActions(currentAgent)
        for action in legalMoves:
            successor = state.generateSuccessor(currentAgent, action)
            successorValue = self.value(successor, depth, currentAgent+1)
            value = max(successorValue, value)

            # In Expectimax, we should store the first action that made us reach
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
            return self.expectValue(state, depth+1, currentAgent)


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.value(gameState)
        return self.action
    

def calcParity(state):
    """
    Returns the difference between each player's score
    """
    parity = 0
    numOfAgents = state.getNumAgents()
    if numOfAgents == 2:
        parity = state.getScore(0) - state.getScore(1)
    elif numOfAgents == 4:
        parity = state.getScore(0) - (state.getScore(1) + state.getScore(2) + state.getScore(3))

    return parity


def calcCorners(state):
    """
    There is a high correlation between the number of corners captured by a player
    and the player winning the game.
    Capturing a majority of the corners, allows for greater stability to be built. 
    Corners are 3 types: captured, potential, unlikely.

    The corner heuristic involves counting the number of discs owned by the player 
    in the corner squares in current and next state.
    """
    cornerValues = state.getCorners()
    corners = 0
    for value in cornerValues:
        if value == 0:
            corners += 1

    return corners


def calcMobility(state):
    """
    Returns the difference between number of our legal moves and the opponent's
    legal moves for the given state

    Calculating "Potential Mobility" (number of legal moves for current and future 
    states) is a complex task. Here, we only calculate the "Actual Mobility".
    """
    numOfAgents = state.getNumAgents()
    if numOfAgents == 2:
        return len(state.getLegalActions(0)) - len(state.getLegalActions(1))

    # 4 players
    return len(state.getLegalActions(0)) - (len(state.getLegalActions(1)) + len(state.getLegalActions(2)) + len(state.getLegalActions(3)))


def calcStability(state):
    """
    The stability measure of a coin is a quantitative representation of how 
    vulnerable it is to being flanked.

    In this implementation, I have defined stability as below:
    A coin is stable if its inside a corner or an edge position.
    (Specifically, a corner coin is classed as stable, and an edge coin is semi-stable)
    
    Returns the number of stable pieces that the agent has.
    """
    stablePositions = [(0, 0), (0, 7), (7, 0), (7, 7)]
    # stablePositions = state.getCorners()
    for i in range(1, 6):
        stablePositions.append((0, i))  # top edge
        stablePositions.append((7, i))  # bottom edge
        stablePositions.append((i, 0))  # left edge
        stablePositions.append((i, 7))  # right edge

    stablePieces = 0
    agentPiecesPositions = state.getPieces()
    for pos in agentPiecesPositions:
        if (pos in stablePositions):
            stablePieces += 1
    
    return stablePieces



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
    # Parity is a little bit greedy-ish... so we assign less weight to it 
    # In this game, the score may change a lot in just 1 move.
    # So, we can't rely too much on parity.
    parityWeight = 1.0
    cornersWeight = 5.0
    mobilityWeight = 2.0
    stabilityWeight = 3.0

    # parity
    parity = calcParity(currentGameState)

    # corners
    corners = calcCorners(currentGameState)

    # mobility: Calculating "Potential Mobility" (number of legal moves for current and future states) is a complex task. Here, we calculate the "Actual Mobility" (number of legal moves for just the current state) instead.
    mobility = calcMobility(currentGameState)

    # stability: Number of stable disks
    stability = calcStability(currentGameState)

    # print(currentGameState.getPieces())
    # print("parity: ", parity)
    # print("mobility: ", mobility)
    # print("corners: ", corners)
    # print("stability: ", stability)

    # Weighted Sum of all the heuristics
    evaluation = (
        parity * parityWeight
        + mobility * mobilityWeight
        + corners * cornersWeight
        + stability * stabilityWeight
    )

    return evaluation


# Abbreviation
better = betterEvaluationFunction