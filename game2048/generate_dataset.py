# Not used any more

import numpy as npy
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

    # 900 training datas, 100 testing datas, k = 900 / 100 = 9
    agent.genTrainData(itrTime=10, goal=2048, k=9)
    agent.genTrainData(itrTime=100, goal=2048, k=9)

    # k = 4
    # agent.genTrainData(itrTime=256, goal=8, k=k)
    # agent.genTrainData(itrTime=128, goal=16, k=k)
    # agent.genTrainData(itrTime=64, goal=32, k=k)
    # agent.genTrainData(itrTime=32, goal=64, k=k)
    # agent.genTrainData(itrTime=16, goal=128, k=k)
    # agent.genTrainData(itrTime=8, goal=256, k=k)
    # agent.genTrainData(itrTime=4, goal=512, k=k)
    # agent.genTrainData(itrTime=2, goal=1024, k=k)
    # agent.genTrainData(itrTime=1, goal=2048, k=k)
    # x = input("itr = 1 generated, continue?")
    # if x is not "Y" and x is not "y":
    #     exit()
    #
    #
    # k = 4
    # agent.genTrainData(itrTime=2560, goal=8, k=k)
    # agent.genTrainData(itrTime=1280, goal=16, k=k)
    # agent.genTrainData(itrTime=640, goal=32, k=k)
    # agent.genTrainData(itrTime=320, goal=64, k=k)
    # agent.genTrainData(itrTime=160, goal=128, k=k)
    # agent.genTrainData(itrTime=80, goal=256, k=k)
    # agent.genTrainData(itrTime=40, goal=512, k=k)
    # agent.genTrainData(itrTime=20, goal=1024, k=k)
    # agent.genTrainData(itrTime=10, goal=2048, k=k)
    # x = input("itr = 10 generated, continue?")
    # if x is not "Y" and x is not "y":
    #     exit()



    #
    # k = 9
    # agent.genTrainData(itrTime=25600, goal=8, k=k)
    # agent.genTrainData(itrTime=12800, goal=16, k=k)
    # agent.genTrainData(itrTime=6400, goal=32, k=k)
    # agent.genTrainData(itrTime=3200, goal=64, k=k)
    # agent.genTrainData(itrTime=1600, goal=128, k=k)
    # agent.genTrainData(itrTime=800, goal=256, k=k)
    # agent.genTrainData(itrTime=400, goal=512, k=k)
    # agent.genTrainData(itrTime=200, goal=1024, k=k)
    # agent.genTrainData(itrTime=100, goal=2048, k=k)
    # x = input("itr = 100 generated, continue?")
    # if x is not "Y" and x is not "y":
    #     exit()
    #
    #
    # k = 9
    # agent.genTrainData(itrTime=256000, goal=8, k=k)
    # agent.genTrainData(itrTime=128000, goal=16, k=k)
    # agent.genTrainData(itrTime=64000, goal=32, k=k)
    # agent.genTrainData(itrTime=32000, goal=64, k=k)
    # agent.genTrainData(itrTime=16000, goal=128, k=k)
    # agent.genTrainData(itrTime=8000, goal=256, k=k)
    # agent.genTrainData(itrTime=4000, goal=512, k=k)
    # agent.genTrainData(itrTime=2000, goal=1024, k=k)
    # agent.genTrainData(itrTime=1000, goal=2048, k=k)
    # x = input("itr = 1000 generated.")



    # X_train = npy.load("./dataset/X_train_" + str(goal) + "_" + str(itrTime * 5) + ".npy")
    # y_train = npy.load("./dataset/y_train_" + str(goal) + "_" + str(itrTime * 5) + ".npy")
    # X_test = npy.load("./dataset/X_test_" + str(goal) + "_" + str(itrTime * 5) + ".npy")
    # y_test = npy.load("./dataset/y_test_" + str(goal) + "_" + str(itrTime * 5) + ".npy")

    # print(X_train)
    # print(y_train)
