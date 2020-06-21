from game2048.game import Game
from game2048.agents import LearningAgent


if __name__ == '__main__':
    game = Game(4, 2048)

    # default: 32
    batch_size = 128

    # set to true if need to create a new model to train
    # warning: this would overwrite existing previous model if game time reaches 500 in training
    new_model = False

    # increase batch size every time a higher score occurs if dynamic_batch = True

    agent = LearningAgent(game=game, display=None,
                          new_model=new_model,
                          model_path="./model/model_multi_level" + str(batch_size) + ".h5")

    agent.multi_level_learn(goal=2048, batch_size=batch_size)




