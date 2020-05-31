from game2048.game import Game
from game2048.displays import Display
from game2048.agents import ExpectiMaxAgent
from game2048.agents import LearningAgent


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=False)
    return game.score


if __name__ == '__main__':
    game = Game(4, 2048)

    agent = LearningAgent(game=game, display=None,
                          trained=False, model_path="./model/model_default.h5")

    agent.genTrainData(itrTime=)

    agent.learn(itr_time=1000, goal = 128)



    # GAME_SIZE = 4
    # SCORE_TO_WIN = 2048
    # N_TESTS = 10
    #
    # '''====================
    # Use your own agent here.'''
    # from game2048.agents import LearningAgent as TestAgent
    # '''===================='''
    #
    # scores = []
    # for _ in range(N_TESTS):
    #     score = single_run(GAME_SIZE, SCORE_TO_WIN,
    #                        AgentClass=TestAgent)
    #     scores.append(score)
    #
    # print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))