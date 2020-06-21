# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# from keras import backend as K
#
# K.tensorflow_backend._get_available_gpus()
# from tensorflow.python.client import device_lib
# import tensorflow as tf
#
# print(device_lib.list_local_devices())
# print(tf.test.is_built_with_cuda())

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from game2048.game import Game
from game2048.displays import Display
from keras.models import load_model


# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


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
                 model_path="./game2048/model/test1.h5",
                 new_model=False, multi_model = False):

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

        if not new_model and not multi_model:
            # try to load a trained model
            try:
                self.model = tf.keras.models.load_model(model_path)
            except:
                print("Model Path:" + model_path)
                print("load model failed, need to create a new one ")
                self.model = self.new_model()
                print("Do you need to train it now?(Y/N)")
                i = input()
                if i == "Y":
                    self.improve_from_dataset()
        elif not multi_model:
            self.model = self.new_model()

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
                                     layers.Flatten()(conv22), layers.Flatten()(conv33), layers.Flatten()(conv44)])

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

    def learn(self, itr_time=5, batch_size=128, goal=2048, dynamic_batch=False):

        if dynamic_batch:
            batch_size = 8

        max_score = 0
        stable = 8
        satisfied = 0

        stat = {2048: 0, 1024: 0, 512: 0, 256: 0, 128: 0, 64: 0, 32: 0, 16: 0}

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

            if (i % 20 == 0):
                print("Training Number: ", i)
                print("Score: ", game.score)
                print("Stable: ", stable)
                # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                # print(sess)

            if i % 500 == 0:
                self.model.save(filepath=self.model_path)

            # increase batch size
            if dynamic_batch and game.score > max_score:
                max_score = game.score
                batch_size = max_score / 4
                print("Higher score occurred, increase batch_size to", batch_size)
                print("Current Max Score is", max_score)

            for s in [2048, 1024, 512, 256, 128, 64, 32, 16]:
                if game.score >= s:
                    stat[s] += 1

            if game.score >= stable * 2:
                satisfied += 1

            # check if go to next stage
            if i % 1000 == 0:
                print("stat: ", stat)
                print("stable", stable)
                if float(satisfied) / 1000 > 0.96:
                    stable *= 2
                satisfied = 0
                for s in [2048, 1024, 512, 256, 128, 64, 32, 16]:
                    stat[s] = 0


        self.model.save(filepath=self.model_path)

    def multi_level_learn(self, batch_size=128, goal=2048):

        stable = 128
        satisfied = 0

        max_score = 0

        # train over and over again
        i = 0
        cnt = 0
        cnt1 = 0
        while stable != goal:
            i += 1

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

                # only learn useful things
                if game.score >= stable or stable <= 64:
                    X_train.append(oht[:, :, :])
                    yi = [0.0, 0.0, 0.0, 0.0]
                    yi[good] = 1.0
                    y_train.append(yi)
                    cnt += 1

                cnt1 += 1
                if cnt == batch_size and X_train!=[] and y_train != []:
                    # print(set_of_lengths(X_train))
                    loss, acc = self.model.train_on_batch(np.array(X_train), np.array(y_train))
                    if cnt1 % 200 == 0:
                        print("Loss\tAcc")
                        print(loss, acc)

                    X_train = []
                    y_train = []
                    cnt = 0

                game.move(direction.argmax())

            if game.score >= stable * 2:
                satisfied += 1

            if (i % 20 == 0):
                print("Training Number: ", i)
                print("Score: ", game.score)
                print("Stable: ", stable)
                # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                # print(sess)

            if i % 200 == 0:
                self.model.save(filepath=self.model_path)


            # check if go to next stage
            if i % 1000 == 0 and stable <= 32:
                print("Stable proportion: ", float(satisfied) / 1000)
                if float(satisfied) / 1000 > 0.96:
                    stable *= 2
                satisfied = 0


            if i % 1000 == 0 and stable == 64:
                print("Stable proportion: ", float(satisfied) / 1000)
                if float(satisfied) / 1000 > 0.94:
                    stable *= 2
                satisfied = 0


            if i % 1000 == 0 and stable == 128:
                print("Stable proportion: ", float(satisfied) / 1000)
                if float(satisfied) / 1000 > 0.9:
                    stable *= 2
                satisfied = 0


            if i % 1000 == 0 and stable == 256:
                print("Stable proportion: ", float(satisfied) / 1000)
                if float(satisfied) / 1000 > 0.85:
                    stable *= 2
                satisfied = 0

            if i % 1000 == 0 and stable == 512:
                print("Stable proportion: ", float(satisfied) / 1000)
                if float(satisfied) / 1000 > 0.8:
                    stable *= 2
                satisfied = 0

            if i % 1000 == 0 and stable == 1024:
                print("Stable proportion: ", float(satisfied) / 1000)
                if float(satisfied) / 1000 > 0.5:
                    stable *= 2
                satisfied = 0



        self.model.save(filepath=self.model_path)

    def multi_level_multi_model_learn(self, itr_time, seq = 0):

        path128 = "./model_multi/multi128.h5"
        path256 = "./model_multi/multi256.h5"
        path512 = "./model_multi/multi512.h5"
        path1024 = "./model_multi/multi1024.h5"

        batch_size_128 = 32
        batch_size_256 = 64
        batch_size_512 = 128
        batch_size_1024 = 256

        try:
            self.model128 = tf.keras.models.load_model(path128)
            self.model256 = tf.keras.models.load_model(path256)
            self.model512 = tf.keras.models.load_model(path512)
            # self.model1024 = tf.keras.models.load_model(path1024)
        except:
            print("Loar error, new models created")
            self.model128 = self.new_model()
            self.model256 = self.new_model()
            self.model512 = self.new_model()
            # self.model1024 = self.new_model()


        max_score = 0


        X_train_128 = []
        y_train_128 = []
        X_train_256 = []
        y_train_256 = []
        X_train_512 = []
        y_train_512 = []
        # X_train_1024 = []
        # y_train_1024 = []
        for i in range(itr_time):
            if (i % 20 == 0):
                print("Generating Training Data: ", i)

            game = Game(4, 2048)
            while game.score < 1024:

                # print(game.board.shape, '\n')
                # print(np.expand_dims(game.board, axis=0).shape)
                oht = self.one_hot(game.board)
                good = self.tch_search_fun(game.board)

                if game.score <= 128:
                    X_train_128.append(oht[:, :, :])
                    yi = [0.0, 0.0, 0.0, 0.0]
                    yi[good] = 1.0
                    y_train_128.append(yi)

                if game.score == 256:
                    X_train_256.append(oht[:, :, :])
                    yi = [0.0, 0.0, 0.0, 0.0]
                    yi[good] = 1.0
                    y_train_256.append(yi)

                if game.score == 512:
                    X_train_512.append(oht[:, :, :])
                    yi = [0.0, 0.0, 0.0, 0.0]
                    yi[good] = 1.0
                    y_train_512.append(yi)

                # if game.score == 1024:
                #     X_train_512.append(oht[:, :, :])
                #     yi = [0.0, 0.0, 0.0, 0.0]
                #     yi[good] = 1.0
                #     y_train_512.append(yi)

                game.move(good)




        X_train_128 = np.array(X_train_128)
        y_train_128 = np.array(y_train_128)

        X_train_256 = np.array(X_train_256)
        y_train_256 = np.array(y_train_256)

        X_train_512 = np.array(X_train_512)
        y_train_512 = np.array(y_train_512)

        # X_train_1024 = np.array(X_train_1024)
        # y_train_1024 = np.array(y_train_1024)

        np.save("X_train_multi_model_128_" + str(seq), X_train_128)
        np.save("X_train_multi_model_256_" + str(seq), X_train_256)
        np.save("X_train_multi_model_512_" + str(seq), X_train_512)
        # np.save("X_train_multi_model_1024_" + str(seq), X_train_1024)
        np.save("y_train_multi_model_128_" + str(seq), y_train_128)
        np.save("y_train_multi_model_256_" + str(seq), y_train_256)
        np.save("y_train_multi_model_512_" + str(seq), y_train_512)
        # np.save("y_train_multi_model_1024", y_train_1024)

        self.model128.fit(X_train_128, y_train_128,
                       epochs=10, batch_size=128,
                       validation_split=0.05)

        self.model256.fit(X_train_256, y_train_256,
                       epochs=10, batch_size=128,
                       validation_split=0.05)

        self.model512.fit(X_train_512, y_train_512,
                       epochs=10, batch_size=128,
                       validation_split=0.05)

        # self.model.fit(X_train, y_train,
        #                epochs=10, batch_size=128,
        #                validation_split=0.05)

        self.model128.save(filepath=path128)
        self.model256.save(filepath=path256)
        self.model512.save(filepath=path512)
        # self.model1024.save(filepath=path1024)

        stat = {2048: 0, 1024: 0, 512: 0, 256: 0, 128: 0, 64: 0, 32: 0, 16: 0}
        total = 0
        for i in range(1000):
            game = Game(4, 2048)
            while not game.end:
                oht = self.one_hot(game.board)
                direction = None

                if game.score <= 128:
                    direction = self.model128.predict(oht[np.newaxis, :, :, :])

                if game.score == 256:
                    direction = self.model256.predict(oht[np.newaxis, :, :, :])

                if game.score == 512:
                    direction = self.model512.predict(oht[np.newaxis, :, :, :])

                game.move(direction.argmax())
            total += game.score

            for s in [2048, 1024, 512, 256, 128, 64, 32, 16]:
                if game.score >= s:
                    stat[s] += 1

        print("Average Score in 1000 iteration currently is: ", float(total) / 1000.0)
        print("stat: ", stat)

    def improve_from_dataset(self, goal = 2048, group = 10000, go_by_self = True):
        stable = 128
        satisfied = 0

        max_score = 0

        cnt = 0
        while True:
            cnt += 1
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for i in range(group):
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
                    if go_by_self:
                        game.move(direction.argmax())
                    else:
                        game.move(good)

                if i % 100 == 0:
                    print("Generating training data...", i)

            for i in range(int(group / 10)):
                game = Game(4, goal)
                while not game.end:

                    # print(game.board.shape, '\n')
                    # print(np.expand_dims(game.board, axis=0).shape)
                    oht = self.one_hot(game.board)
                    direction = self.model.predict(oht[np.newaxis, :, :, :])
                    good = self.tch_search_fun(game.board)

                    X_test.append(oht[:, :, :])
                    yi = [0.0, 0.0, 0.0, 0.0]
                    yi[good] = 1.0
                    y_test.append(yi)
                    if go_by_self:
                        game.move(direction.argmax())
                    else:
                        game.move(good)

                if i % 100 == 0:
                    print("Generating testing data...", i)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            # np.save("./dataset/old_X_train_" + str(cnt), X_train)
            # np.save("./dataset/old_y_train_" + str(cnt), y_train)
            # np.save("./dataset/old_X_test_"  + str(cnt), X_test)
            # np.save("./dataset/old_y_test_"  + str(cnt), y_test)

            self.model.fit(X_train, y_train,
                           epochs=10, batch_size=128,
                           validation_split=0.05)
            self.model.save(filepath=self.model_path)

            score = self.model.evaluate(X_test, y_test, batch_size=128)
            print("Iteration time:", cnt)
            print("Score: ", score)

            total = 0
            for i in range(1000):
                game = Game(4, goal)
                while not game.end:
                    oht = self.one_hot(game.board)
                    direction = self.model.predict(oht[np.newaxis, :, :, :])
                    game.move(direction.argmax())
                total += game.score

            print("Average Score currently is: ", float(total)/1000.0)
            if float(total)/1000.0 > 700:
                break

    def learn_from_dataset_from_master(self, L, R, group = 100000):

        print("Training: [L, R] = ", L, R)
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # try:
        #     X_train = np.load("./dataset/X_train_between" + str(L) + "_" + str(R))
        #     t_train = np.load("./dataset/y_train_between" + str(L) + "_" + str(R))
        #     np.load("./dataset/X_test_between" + str(L) + "_" + str(R))
        #     np.load("./dataset/y_test_between" + str(L) + "_" + str(R))
        #
        for i in range(group):
            game = Game(4, 2048)
            while not game.end:
                oht = self.one_hot(game.board)
                good = self.tch_search_fun(game.board)

                if game.score > R:
                    break

                if game.score > L:
                    X_train.append(oht[:, :, :])
                    yi = [0.0, 0.0, 0.0, 0.0]
                    yi[good] = 1.0
                    y_train.append(yi)

                game.move(good)

            if i % 200 == 0:
                print("Generating training data... ", i, "/", group)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        np.save("./dataset/X_train_between" + str(L) + "_" + str(R), X_train)
        np.save("./dataset/y_train_between" + str(L) + "_" + str(R), y_train)

        for i in range(int(group / 10)):
            game = Game(4, 2048)
            while not game.end:

                # print(game.board.shape, '\n')
                # print(np.expand_dims(game.board, axis=0).shape)
                oht = self.one_hot(game.board)
                good = self.tch_search_fun(game.board)

                if game.score > R:
                    break

                if game.score > L:
                    X_test.append(oht[:, :, :])
                    yi = [0.0, 0.0, 0.0, 0.0]
                    yi[good] = 1.0
                    y_test.append(yi)

                game.move(good)

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        np.save("./dataset/X_test_between"  + str(L) + "_" + str(R), X_test)
        np.save("./dataset/y_test_between"  + str(L) + "_" + str(R), y_test)

        self.model.fit(X_train, y_train,
                       epochs=10, batch_size=128,
                       validation_split=0.05)
        self.model.save(filepath=self.model_path)

        score = self.model.evaluate(X_test, y_test, batch_size=128)
        print("Score: ", score)

        total = 0
        for i in range(1000):
            game = Game(4, 2048)
            while not game.end:
                oht = self.one_hot(game.board)
                direction = self.model.predict(oht[np.newaxis, :, :, :])
                game.move(direction.argmax())
            total += game.score

        print("Average Score currently is: ", float(total) / 1000.0)



    def set_trained(self):
        self.trained = True

    def self_test(self):

        import time

        totoal_time = 0
        cnt = 0


        stat = {2048: 0, 1024: 0, 512: 0, 256: 0, 128: 0, 64: 0, 32: 0, 16: 0}
        total = 0
        for i in range(1000):

            if i % 10 == 0:
                print("Test: ", i)

            game = Game(4, 2048)
            while not game.end:
                start = time.clock()
                oht = self.one_hot(game.board)
                direction = self.model.predict(oht[np.newaxis, :, :, :])
                dir = direction.argmax()
                end = time.clock()

                totoal_time += end - start
                cnt += 1

                game.move(dir)
            total += game.score

            for s in [2048, 1024, 512, 256, 128, 64, 32, 16]:
                if game.score >= s:
                    stat[s] += 1

        print("Average Score currently is: ", float(total) / 1000.0)
        print("stat: ", stat)
        print("Time for one step (x second/step): ", float(totoal_time) / float(cnt))

    def step(self):
        oht = self.one_hot(self.game.board)
        direction = self.model.predict(oht[np.newaxis, :, :, :])
        return direction.argmax()


