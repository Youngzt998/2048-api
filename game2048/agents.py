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
                 model_path = "./model/model_default_batch256.h5", new_model = False):

        super().__init__(game, display)

        if tch_search_fun == None:
            from .expectimax import board_to_move
            self.tch_search_fun = board_to_move
        else:
            self.tch_search_fun = tch_search_fun

        self.trained = new_model
        self.model_path = model_path
        self.model = None
        self.one_hot_map = {2 ** i: i for i in range(1, 16)}
        self.one_hot_map[0] = 0

        if not new_model:
            # try to load a trained model
            try:
                self.model = tf.keras.models.load_model(model_path)
            except:
                print("load model failed, need to create a new one ")
                self.model = self.new_model()
        else:
            self.model = self.new_model()


    # def reshapeBoard(self, X):
    #     shape = X.shape
    #     for

    # Not used anymore
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

    def new_model(self):

        input = keras.Input((4, 4, 16))
        conv = input
        filters = 128
        conv41 = layers.Conv2D(filters=filters, kernel_size=(4, 1), kernel_initializer='he_uniform')(conv)
        conv14 = layers.Conv2D(filters=filters, kernel_size=(1, 4), kernel_initializer='he_uniform')(conv)
        conv22 = layers.Conv2D(filters=filters, kernel_size=(2, 2), kernel_initializer='he_uniform')(conv)
        conv33 = layers.Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer='he_uniform')(conv)
        conv44 = layers.Conv2D(filters=filters, kernel_size=(4, 4), kernel_initializer='he_uniform')(conv)

        hidden = layers.concatenate([layers.Flatten()(conv41), layers.Flatten()(conv14),
                                     layers.Flatten()(conv22), layers.Flatten()(conv33),  layers.Flatten()(conv44)])

        x = layers.BatchNormalization()(hidden)
        x = layers.Activation('relu')(hidden)

        for width in [512, 128]:
            x = layers.Dense(width, kernel_initializer='he_uniform')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

        output = layers.Dense(4, activation='softmax')(x)

        model = keras.Model(inputs=input,
                            outputs=output,
                            name="youngster38324's model")
        model.summary()

        # batch_size = batch_size
        # epochs = epochs

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        return model




    def one_hot(self, arr):
        shape = (4, 4)
        ans = np.zeros(shape=shape + (16,), dtype=bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                ans[i, j, self.one_hot_map[arr[i, j]]] = 1

        # print(ans.shape)
        return ans


    def learn(self, itr_time = 5, batch_size = 128, goal = 2048):

        # train over and over again
        cnt = 0
        cnt1 = 0
        for i in range(itr_time):

            X_train = []
            y_train = []
            loss = acc = 0
            game = Game(4, goal)
            while not game.end:

                # print(game.board.shape, '\n')
                # print(np.expand_dims(game.board, axis=0).shape)
                oht = self.one_hot(game.board)
                direction = self.model.predict(oht[np.newaxis, :, :, :])
                good = self.tch_search_fun(game.board)

                X_train.append(oht[:, :, :])
                yi = [0.0, 0.0, 0.0, 0.0]
                yi[good] = 1.0
                y_train.append(yi)
                # y_train.append(yi)

                cnt += 1
                cnt1 += 1
                if cnt == batch_size:
                    # print(set_of_lengths(X_train))
                    loss, acc = self.model.train_on_batch(np.array(X_train), np.array(y_train))
                    if cnt1 % 200 == 0:
                        print("Loss\tAcc")
                        print(loss, acc)

                    X_train = []
                    y_train = []
                    cnt = 0

                if cnt1 % 1000 == 0:
                    print("Agent: ", direction)
                    print("Good: ", yi)
                    cnt1 = 0

                game.move(direction.argmax())

            if(i % 20 == 0):
                print("Training Number: ", i)
                print("Score: ", game.score)


            if i % 1000 ==0:
                self.model.save(filepath=self.model_path)

        self.model.save(filepath=self.model_path)

    def set_trained(self):
        self.trained = True

    def step(self):
        if self.trained:
            direction = np.random.randint(0, 4)
            return direction
        else:
            oht = self.one_hot(self.game.board)
            direction = self.model.predict(oht[np.newaxis, :, :, :])

            # return max point direction
            return direction.argmax()

















