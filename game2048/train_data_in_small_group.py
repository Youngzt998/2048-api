from game2048.game import Game
from game2048.agents import LearningAgent


if __name__ == '__main__':
    game = Game(4, 2048)

    # default: 32
    batch_size = 128

    # set to true if need to create a new model to train
    # warning: this would overwrite existing previous model if game time reaches 500 in training
    new_model = False

    agent = LearningAgent(game=game, display=None, tch_search_fun=None,
                          new_model=new_model,
                          model_path="./model/model_data_in_small_group.h5")
    agent.improve_from_dataset(group=2000)



