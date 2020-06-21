from game2048.game import Game
from game2048.agents import LearningAgent

if __name__ == '__main__':
    game = Game(4, 2048)

    # default: 32
    batch_size = 128

    # set to true if need to create a new model to train
    # warning: this would overwrite existing previous model if game time reaches 500 in training
    new_model = False

    agent = LearningAgent(game=game, display=None,
                          new_model=new_model,
                          model_path="./model/model_multi_level_from_master.h5")
    # agent.learn_from_dataset_from_master(L=0, R=16, group=100000)
    # agent.learn_from_dataset_from_master(L=16, R=32, group=100000)
    agent.learn_from_dataset_from_master(L=32, R=64, group=100000)
    agent.learn_from_dataset_from_master(L=64, R=128, group=50000)
    agent.learn_from_dataset_from_master(L=128, R=256, group=50000)
    agent.learn_from_dataset_from_master(L=256, R=512, group=30000)
    agent.learn_from_dataset_from_master(L=512, R=1024, group=30000)
    agent.learn_from_dataset_from_master(L=1024, R=2048, group=20000)
