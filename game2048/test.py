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

    # default: 128
    batch_size = 32

    agent = LearningAgent(game=game, display=None,
                          new_model=False, model_path="./model/model_default_batch" + str(batch_size) + ".h5")


    agent.learn(itr_time=1000000, goal = 2048, batch_size=batch_size,)

    # agent = LearningAgent(game=game, display=None,
    #                       new_model=False, model_path="./model/model_default.h5")
    #
    #
    # agent.learn(itr_time=1000000, goal = 2048)