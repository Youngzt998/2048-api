import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from game2048.game import Game
from game2048.displays import Display
from keras.models import load_model

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

# A repeater
class LearningAgent(Agent):

    def __init__(self, game, display=None, tch_search_fun=None,
                 model_path = "./model/model_default.h5", trained = True):

        super().__init__(game, display)

        if tch_search_fun == None:
            from .expectimax import board_to_move
            self.tch_search_fun = board_to_move
        self.trained = trained
        self.model_path = model_path
        self.model = None

        if trained:
            # try to load a trained model
            try:
                self.model = tf.keras.models.load_model(model_path)
            except:
                print("load model failed, do you want to train it now?")
                ans = input()
                if ans is "Y" or ans is "y":
                    self.trained=False
                else:
                    exit()
                self.trained = True


    # def reshapeBoard(self, X):
    #     shape = X.shape
    #     for

    def genTrainData(self, itrTime = 5, goal = 2048, k = 5):
        X_train, y_train = [], []
        X_test, y_test = [], []


        # training data
        for i  in range(itrTime * k):
            # get a random initial game board
            print("Get training data: i = ", i)
            game = Game(4, goal)
            n_itr, max_itr = 0, 10 # np.inf
            while not game.end: #and n_itr < max_itr:
                n_itr += 1
                X_train.append(game.board[:, :, np.newaxis])    # add a dimension
                yi = [0.0, 0.0, 0.0, 0.0]
                direction = self.tch_search_fun(game.board)
                yi[direction] = 1.0
                y_train.append(yi)
                game.move(direction)

        # testing data
        for i in range(itrTime):
            print("Get testing data: i = ", i)
            game = Game(4, goal)
            n_itr, max_itr = 0, np.inf
            while not game.end: #and n_itr < max_itr:
                n_itr += 1
                X_test.append(game.board[:, :, np.newaxis])
                yi = [0.0, 0.0, 0.0, 0.0]
                direction = self.tch_search_fun(game.board)
                yi[direction] = 1.0
                y_test.append(yi)
                game.move(direction)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        np.save("./dataset/X_train_" + str(goal) + "_" + str(itrTime*k) + "_" + str(k), X_train)
        np.save("./dataset/y_train_" + str(goal) + "_" + str(itrTime*k) + "_" + str(k), y_train)
        np.save("./dataset/X_test_" + str(goal) + "_" + str(itrTime*k) + "_" + str(k), X_test)
        np.save("./dataset/y_test_" + str(goal) + "_" + str(itrTime*k) + "_" + str(k), y_test)

        return X_train, y_train, X_test, y_test



    def learn(self, itr_time = 5, goal = 2048,
              batch_size = 128, epochs = 10):

        X_train, y_train, X_test, y_test \
            = self.genTrainData(itrTime = itr_time, goal=goal)
        # print(X_train)

        # X_train = [
        #     [[[[1], [1], [1], [1]],
        #      [[1], [1], [1], [1]],
        #      [[1], [1], [1], [1]],
        #      [[1], [1], [1], [1]]]]
        # ]

        input = keras.Input(shape=(4, 4, 1), name="first_model")
        x = layers.Conv2D(16, 2, activation="relu")(input)
        x = layers.Conv2D(16, 2, activation="relu")(x)
        # x = layers.MaxPooling2D(3)(x)
        # x = layers.Conv2D(32, 2, activation="relu")(x)
        x = layers.Conv2D(4, 2, activation="relu")(x)
        x = layers.Activation(activation="softmax")(x)
        output = layers.GlobalMaxPooling2D()(x)

        model = keras.Model(inputs=input,
                            outputs=output,
                            name="first_model")
        model.summary()

        batch_size = batch_size
        epochs = epochs

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.1)

        score = model.evaluate(X_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        model.save(filepath=self.model_path)

        self.model = model
        self.trained = True

        return score

    def set_trained(self):
        self.trained = True

    def step(self):
        if self.trained:
            direction = np.random.randint(0, 4)
            return direction
        else:
            y = self.model.predict(self.game.board[:, :, np.newaxis])
            direction = np.random.randint(0, 4)
            return direction

















