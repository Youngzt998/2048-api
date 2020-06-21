from game2048.game import Game
from game2048.agents import LearningAgent


if __name__ == '__main__':
    game = Game(4, 2048)


    agent = LearningAgent(game=game, display=None,
                          new_model=False,
                          model_path="./model/test1.h5")

    agent.self_test()





