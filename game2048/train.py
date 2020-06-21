from game2048.game import Game
from game2048.agents import LearningAgent


if __name__ == '__main__':
    game = Game(4, 2048)

    # default: 32
    batch_size = 32

    # set to true if need to create a new model to train
    # warning: this would overwrite existing previous model if game time reaches 500 in training
    new_model = False

    # increase batch size every time a higher score occurs if dynamic_batch = True
    dynamic_batch = False

    if(not dynamic_batch):
        agent = LearningAgent(game=game, display=None,
                              new_model=new_model, model_path="./model/model_default_batch" + str(batch_size) + ".h5")
    else:
        agent = LearningAgent(game=game, display=None,
                              new_model=new_model, model_path="./model/model_default_batch_dynamic.h5")

    agent.learn(itr_time=100000, goal=2048, batch_size=batch_size, dynamic_batch=dynamic_batch)




