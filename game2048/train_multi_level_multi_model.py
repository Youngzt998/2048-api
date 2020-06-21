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

    agent = LearningAgent(game=game,
                          display=None,
                          tch_search_fun=None,
                          new_model=new_model,
                          model_path=None,
                          multi_model=True)

    for i in range(20):
        agent.multi_level_multi_model_learn(itr_time=1000, seq=i)

    # agent.multi_level_multi_model_learn(itr_time=1000, seq=0)
    # agent.multi_level_multi_model_learn(itr_time=1000, seq=1)
    # agent.multi_level_multi_model_learn(itr_time=1000, seq=2)
    # agent.multi_level_multi_model_learn(itr_time=1000, seq=3)
    # agent.multi_level_multi_model_learn(itr_time=1000, seq=4)
    # agent.multi_level_multi_model_learn(itr_time=1000, seq=5)
    # agent.multi_level_multi_model_learn(itr_time=1000, seq=6)





