from utils import *
from metrics import *

import numpy as np
np.random.seed(0)
from datetime import datetime
from os import path
import sys


class MFModel:
    """
    This class encapsulates all the necessary data and methods for the recommendation system.
    """
    def __init__(self, n_users, n_items, k, reg, scale):
        """
        Constructor of the MFModel class.

        :param n_users: total number of users
        :param n_items: total number of items
        :param k: the dimension of the latent vectors
        :param reg: 4-tuple of the regularization parameters. the order should be the following: 1. user vectors,
                        2. item vectors, 3. user bias, 4. item bias.
        :param scale: std of the normal distribution for the vectors initialization
        """
        self.k = k

        self.n_users = n_users
        self.n_items = n_items

        self.users_vecs = np.random.normal(scale=scale, size=(n_users, k))
        self.items_vecs = np.random.normal(scale=scale, size=(n_items, k))

        self.users_biases = np.zeros(n_users)
        self.items_biases = np.zeros(n_items)

        self.users_vec_reg, self.items_vec_reg, self.users_bias_reg, self.items_bias_reg = reg

        self.r_mean = None

    def loss_mse(self, ratings_data):
        """
        returns the rss loss with red term and the mse using the data 'ratings_data'.
        :param ratings_data: np array, where each row is [<user_id>, <item_id>, <rating>]
        :return: (rss+l2_loss, mse) tuple
        """
        users_ids = ratings_data[:, 0]
        items_ids = ratings_data[:, 1]
        r = ratings_data[:, 2]

        # slice the latent vecs and biases according to the data's indexes
        u = self.users_vecs[users_ids]
        v = self.items_vecs[items_ids]
        u_bias = self.users_biases[users_ids]
        v_bias = self.items_biases[items_ids]

        # compute loss
        # rss loss
        rss = np.sum((r - (self.r_mean + np.sum(u*v, axis=1) + u_bias + v_bias)) ** 2)
        loss = 0.5 * rss

        # reg loss
        loss += 0.5 * self.users_vec_reg * np.sum(self.users_vecs * self.users_vecs)
        loss += 0.5 * self.items_vec_reg * np.sum(self.items_vecs * self.items_vecs)
        loss += 0.5 * self.users_bias_reg * np.sum(self.users_biases ** 2)
        loss += 0.5 * self.items_bias_reg * np.sum(self.items_biases ** 2)

        # mse
        mse = rss / len(ratings_data)
        return loss, mse

    def LearnModelFromDataUsingSGD(self, lr, n_epochs, train_data, test_data=None):
        """
        Trains the model using sgd algorithm.
        During training it saves the rss and mse of the train and test to log files

        :param lr: learning rate for the sgd
        :param n_epochs: number of epochs.
        :param train_data: data to use for trainig the model.
                           np array - each row is [<user_id>, <item_id>, <rating>]
        :param test_data: data to be used for a out of box validation.
                          np array - each row is [<user_id>, <item_id>, <rating>]
        """
        # log files paths
        log_path_train = path.join('logs', 'sgd_train_lr{}_epo{}_k{}_reg{}.txt'.format(lr, n_epochs, self.k, self.users_vec_reg))
        log_path_test = path.join('logs', 'sgd_test_lr{}_epo{}_k{}_reg{}.txt'.format(lr, n_epochs, self.k, self.users_vec_reg))

        # update the mean raring using the train data
        self.r_mean = np.mean(train_data[:, 2])

        # initialize lists to store and visualize losses and mse during training
        epoch_lst = []
        train_loss_lst = []
        train_mse_lst = []

        test_loss_lst = []
        test_mse_lst = []

        # train
        for epoch in range(n_epochs):
            for rating in train_data:
                user_idx = rating[0]
                item_idx = rating[1]
                r = rating[2]

                # user and item vectors and biases
                u = self.users_vecs[user_idx]
                v = self.items_vecs[item_idx]
                u_bias = self.users_biases[user_idx]
                v_bias = self.items_biases[item_idx]

                # grads
                e = (r - self.r_mean - np.sum(u * v) - u_bias - v_bias)
                du = -e * v + self.users_vec_reg * u
                dv = -e * u + self.items_vec_reg * v
                du_bias = -e + self.users_bias_reg * u_bias
                dv_bias = -e + self.items_bias_reg * v_bias

                # sgd update
                self.users_vecs[user_idx] -= lr * du
                self.items_vecs[item_idx] -= lr * dv
                self.users_biases[user_idx] -= lr * du_bias
                self.items_biases[item_idx] -= lr * dv_bias

            # save losses and mse to log files
            epoch_lst.append(epoch)
            train_loss, train_mse = self.loss_mse(train_data)
            train_loss_lst.append(train_loss)
            train_mse_lst.append(train_mse)
            print('{}, {}, {}'.format(epoch, train_loss, train_mse), file=open(log_path_train, "a"))
            print('{}, {}, {}'.format(epoch, train_loss, train_mse))

            if test_data is not None:
                test_loss, test_mse = self.loss_mse(test_data)
                test_loss_lst.append(test_loss)
                test_mse_lst.append(test_mse)
                print('{}, {}, {}'.format(epoch, test_loss, test_mse), file=open(log_path_test, "a"))
                print('{}, {}, {}'.format(epoch, test_loss, test_mse))

        return {'train_loss': train_loss_lst, 'train_mse': train_mse_lst,
                'test_loss': test_loss_lst, 'test_mse': test_mse_lst}

    def LearnModelFromDataUsingALS(self, n_epochs, train_data, test_data=None):
        """
        Trains the model using ALS algorithm.
        During training it saves the rss and mse of the train and test to log files
        :param n_epochs:
        :param train_data:
        :param test_data:
        :return:
        """
        # log files paths
        log_path_train = path.join('logs', 'als_train_epo{}_k{}_reg{}.txt'.format(n_epochs, self.k, self.users_vec_reg))
        log_path_test = path.join('logs', 'als_test_epo{}_k{}_reg{}.txt'.format(n_epochs, self.k, self.users_vec_reg))

        # update the mean raring using only the train data
        self.r_mean = np.mean(train_data[:, 2])

        # initialize lists to store and visualize losses and mse during training
        epoch_lst = []
        train_loss_lst = []
        train_mse_lst = []

        test_loss_lst = []
        test_mse_lst = []

        # lists of unique user_ids and item_ids in the train data
        users_idx = list(set(train_data[:, 0]))
        items_idx = list(set(train_data[:, 1]))

        # train
        for epoch in range(n_epochs):
            for item_idx in items_idx:
                # filter the data to this item_idx only
                item_msk = train_data[:, 1] == item_idx
                item_data = train_data[item_msk]

                # slice data according to the item_idx
                item_users_idx = item_data[:, 0]
                item_items_idx = item_data[:, 1]
                item_users_vecs = self.users_vecs[item_users_idx]
                item_items_vecs = self.items_vecs[item_items_idx]
                item_users_bias = self.users_biases[item_users_idx]
                item_items_bias = self.items_biases[item_items_idx]
                item_ratings = item_data[:, 2]
                item_n = len(item_data)

                # calculate the new item vector
                v_n = item_users_vecs.T.dot(item_users_vecs)
                v_n += item_n * self.users_vec_reg * np.eye(self.k)
                v_n = np.linalg.inv(v_n)
                e = (item_ratings - self.r_mean - item_items_bias - item_users_bias)
                v_n *= np.sum(e.reshape(-1, 1) * item_users_vecs, axis=0)
                v_n = np.sum(v_n, axis=1)
                # update the item vector in the model
                self.items_vecs[item_idx] = v_n

                # calculate the new item bias
                b_n = np.sum(item_ratings - self.r_mean - item_users_bias) - np.sum(item_items_vecs*item_users_vecs)
                b_n /= (item_n + self.items_bias_reg)
                # update the item bias in the model
                self.items_biases[item_idx] = b_n

            for user_idx in users_idx:
                # filter the data to this user_idx only
                user_msk = train_data[:, 0] == user_idx
                user_data = train_data[user_msk]

                # slice data according to the user_idx
                user_users_idx = user_data[:, 0]
                user_items_idx = user_data[:, 1]
                user_items_vecs = self.items_vecs[user_items_idx]
                user_users_vecs = self.users_vecs[user_users_idx]
                user_users_bias = self.users_biases[user_users_idx]
                user_items_bias = self.items_biases[user_items_idx]
                user_ratings = user_data[:, 2]
                user_n = len(user_data)

                # calculate the new user vector
                u_m = user_items_vecs.T.dot(user_items_vecs)
                u_m += user_n * self.items_vec_reg * np.eye(self.k)
                u_m = np.linalg.inv(u_m)
                e = (user_ratings - self.r_mean - user_items_bias - user_users_bias)
                u_m *= np.sum(e.reshape(-1, 1) * user_items_vecs, axis=0)
                u_m = np.sum(u_m, axis=1)
                # update the item vector in the model
                self.users_vecs[user_idx] = u_m

                # calculate the new user bias
                b_m = np.sum(user_ratings - self.r_mean - user_items_bias) - np.sum(user_items_vecs*user_users_vecs)
                b_m /= (user_n + self.users_bias_reg)
                # update the user bias in the model
                self.users_biases[user_idx] = b_m

            # save losses and mse to log files
            epoch_lst.append(epoch)
            train_loss, train_mse = self.loss_mse(train_data)
            train_loss_lst.append(train_loss)
            train_mse_lst.append(train_mse)
            print('{}, {}, {}'.format(epoch, train_loss, train_mse), file=open(log_path_train, "a"))
            print('{}, {}, {}'.format(epoch, train_loss, train_mse))

            if test_data is not None:
                test_loss, test_mse = self.loss_mse(test_data)
                test_loss_lst.append(test_loss)
                test_mse_lst.append(test_mse)
                print('{}, {}, {}'.format(epoch, test_loss, test_mse), file=open(log_path_test, "a"))
                print('{}, {}, {}'.format(epoch, test_loss, test_mse))

        return {'train_loss': train_loss_lst, 'train_mse': train_mse_lst,
                'test_loss': test_loss_lst, 'test_mse': test_mse_lst}

    def predict(self):
        """
        Uses to model and all its users indexes and items indexes to create the following matrix:
            1. prediction_matrix - matrix where the ij cell is the predicted ranking for user i and item j.

        """
        n_users = self.users_vecs.shape[0]
        n_items = self.items_vecs.shape[0]

        prediction_matrix = np.zeros((n_users, n_items))

        for i in np.arange(n_users):
            for j in np.arange(n_items):
                prediction_matrix[i, j] = self.r_mean + \
                                          np.sum(self.users_vecs[i] * self.items_vecs[j]) + \
                                          self.users_biases[i] + \
                                          self.items_biases[j]

        return prediction_matrix

if __name__ == '__main__':
    # Data Paths
    ratings_data_path = r'data/ratings.dat'
    items_data_path = r'data/movies.dat'

    # read and parse config file
    config_path = r'config_file.txt'

    with open(config_path) as f:
        args = f.readlines()[0].strip().split(',')
    try:
        learning_algorithm = args[0]
        assert learning_algorithm in ['als', 'sgd']
        k = int(args[1])
        n_epochs = int(args[2])
        scale = float(args[3])
        reg = [float(x) for x in args[4:8]]
        if learning_algorithm == 'sgd':
            lr = float(args[8])
    except (IndexError, ValueError, AssertionError) as err:
        print('config file is not legal')
        print(type(err))
        sys.exit(1)

    # Training Description file
    description_path = path.join('logs', 'Description_{}_train_epo{}_k{}_reg{}.txt'.
                                 format(learning_algorithm, n_epochs, k, reg[0]))

    # read and split data
    ratings, idx2user, idx2item = read_ratings_data(ratings_data_path)
    items = read_items_data(items_data_path)
    train, test = train_test_split(ratings)

    # initialize the model
    model = MFModel(n_users=len(idx2user), n_items=len(idx2item), k=k, reg=reg, scale=scale)

    # Train
    initial_training_time = datetime.now()
    if learning_algorithm == 'sgd':
        model.LearnModelFromDataUsingSGD(lr=0.001, n_epochs=n_epochs, train_data=train, test_data=test)
    else:
        model.LearnModelFromDataUsingALS(n_epochs=n_epochs, train_data=train, test_data=test)
    training_time = datetime.now() - initial_training_time

    # predict ratings for all users and items
    prediction_matrix = model.predict()

    # Save Description of learning to file
    print('Learning Algorithm: {}'.format(learning_algorithm), file=open(description_path, "a"))
    print('Hyperparameter k: {}'.format(k), file=open(description_path, "a"))
    print('Number of epochs in training: {}'.format(n_epochs), file=open(description_path, "a"))
    print('Scale for the latent vectors initialization: {}'.format(scale), file=open(description_path, "a"))
    print('Regularization term: {}'.format(reg[0]), file=open(description_path, "a"))
    if learning_algorithm == 'sgd':
        print('Learning rate: {}'.format(lr), file=open(description_path, "a"))
    print('', file=open(description_path, "a"))
    print('Training time: {}'.format(training_time), file=open(description_path, "a"))
    print('', file=open(description_path, "a"))

    # compute metrics on the test set
    metrics_evaluations = {}
    metrics_evaluations['rmse'] = rmse(prediction_matrix, test)
    metrics_evaluations['p@2'] = precision_at_k(prediction_matrix, test, 2)
    metrics_evaluations['p@10'] = precision_at_k(prediction_matrix, test, 10)
    metrics_evaluations['r@2'] = recall_at_k(prediction_matrix, test, 2)
    metrics_evaluations['r@10'] = recall_at_k(prediction_matrix, test, 10)
    metrics_evaluations['mean_average_precision'] = mean_average_precision(prediction_matrix, test)
    metrics_evaluations['mpr'] = mpr(prediction_matrix, test)

    print('Metrics evaluations:', file=open(description_path, "a"))
    for key in metrics_evaluations:
        print('\t{}: {}'.format(key, metrics_evaluations[key]), file=open(description_path, "a"))





