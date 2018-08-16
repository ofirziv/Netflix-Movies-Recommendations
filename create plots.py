from recommendation_system import MFModel
from utils import *
import numpy as np
import time

import matplotlib.pyplot as plt


ratings_data_path = r'data/ratings.dat'
items_data_path = r'data/movies.dat'

# read and split data
ratings, idx2user, idx2item = read_ratings_data(ratings_data_path)
all_items_data = read_items_data(items_data_path)
train, test = train_test_split(ratings)


def deliverable_1():
    # als model
    model = MFModel(n_users=len(idx2user), n_items=len(idx2item), k=20, reg=(0.1, 0.1, 0.1, 0.1), scale=0.01)
    metrics_during_learning = model.LearnModelFromDataUsingALS(n_epochs=15, train_data=train, test_data=test)

    test_mse = metrics_during_learning['test_mse']
    train_mse = metrics_during_learning['train_mse']
    test_loss = metrics_during_learning['test_loss']
    test_loss = [x / len(test) for x in test_loss]
    train_loss = metrics_during_learning['train_loss']
    train_loss = [x / len(train) for x in train_loss]
    epochs_lst = range(len(train_loss))

    plt.figure()
    plt.plot(epochs_lst, train_mse, label='Train', c='blue')
    plt.plot(epochs_lst, test_mse, label='Test', c='red')
    plt.legend()
    plt.title('als - MSE during training - K=20, reg=0.1, scale=0.01, n_epochs=15')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.savefig('plots\\deliverable_1\\als MSE during training - K=20, reg=0.1, scale=0.01, n_epochs=15.png')

    plt.figure()
    plt.plot(epochs_lst, train_loss, label='Train', c='blue')
    plt.plot(epochs_lst, test_loss, label='Test', c='red')
    plt.legend()
    plt.title('als - mean Loss during training - K=20, reg=0.1, scale=0.01, n_epochs=15')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.savefig('plots\\deliverable_1\\als mean Loss during training - K=20, reg=0.1, scale=0.01, n_epochs=15.png')

    # sgd model
    model = MFModel(n_users=len(idx2user), n_items=len(idx2item), k=30, reg=(0.01, 0.01, 0.01, 0.01), scale=0.01)
    metrics_during_learning = model.LearnModelFromDataUsingSGD(lr=0.001, n_epochs=25, train_data=train, test_data=test)

    test_mse = metrics_during_learning['test_mse']
    train_mse = metrics_during_learning['train_mse']
    test_loss = metrics_during_learning['test_loss']
    test_loss = [x/len(test) for x in test_loss]
    train_loss = metrics_during_learning['train_loss']
    train_loss = [x / len(train) for x in train_loss]
    epochs_lst = range(len(train_loss))

    plt.figure()
    plt.plot(epochs_lst, train_mse, label='Train', c='blue')
    plt.plot(epochs_lst, test_mse, label='Test', c='red')
    plt.legend()
    plt.title('sgd - MSE during training - K=30, reg=0.01, scale=0.01, n_epochs=25, lr=0.001')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.savefig('plots\\deliverable_1\\sgd MSE during training - K=30, reg=0.01, scale=0.01, n_epochs=25, lr=0.001.png')

    plt.figure()
    plt.plot(epochs_lst, train_loss, label='Train', c='blue')
    plt.plot(epochs_lst, test_loss, label='Test', c='red')
    plt.legend()
    plt.title('sgd - mean Loss during training - K=30, reg=0.01, scale=0.01, n_epochs=25, lr=0.001')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.savefig('plots\\deliverable_1\\sgd mean Loss during training - K=30, reg=0.01, scale=0.01, n_epochs=25, lr=0.001.png')


def deliverable_2():
    reg_lst = [0.1, 1, 10, 100, 1000]
    reg_mse_lst = []
    reg_loss_lst = []
    for reg in reg_lst:
        model = MFModel(n_users=len(idx2user), n_items=len(idx2item), k=15, reg=(reg, reg, reg, reg), scale=0.01)
        # metrics_during_learning = model.LearnModelFromDataUsingSGD(lr=0.001, n_epochs=15, train_data=train,
        #                                                            test_data=test)
        metrics_during_learning = model.LearnModelFromDataUsingALS(n_epochs=10, train_data=train, test_data=test)

        reg_mse_lst.append(min(metrics_during_learning['test_mse']))
        reg_loss_lst.append(min(metrics_during_learning['test_loss']))

    log_reg_lst = [np.log10(x) for x in reg_lst]
    plt.figure()
    plt.plot(log_reg_lst, reg_mse_lst, label='mse', c='blue')
    plt.legend()
    plt.title('als - best test mse as function of log(reg) - K=20, scale=0.01, n_epochs=15, lr=0.001')
    plt.xlabel('log10(reg)')
    plt.ylabel('mse')
    plt.savefig('plots\\deliverable_2\\als - best test mse as function of log(reg) - K=20, scale=0.01, n_epochs=15, lr=0.001.png')

    plt.figure()
    plt.plot(log_reg_lst, reg_loss_lst, label='loss', c='blue')
    plt.legend()
    plt.title('als - best test loss as function of log(reg) - K=20, scale=0.01, n_epochs=15, lr=0.001')
    plt.xlabel('log10(reg)')
    plt.ylabel('loss')
    plt.savefig('plots\\deliverable_2\\als - best test loss as function of log(reg) - K=20, scale=0.01, n_epochs=15, lr=0.001.png')


def deliverable_3and4():
    reg = 0.1
    d_lst = [4, 10, 20, 40, 50, 70, 100, 200]
    mse_lst = []
    loss_lst = []
    run_time = []

    for d in d_lst:
        model = MFModel(n_users=len(idx2user), n_items=len(idx2item), k=d, reg=(reg, reg, reg, reg), scale=0.01)
        start_time = time.time()
        metrics_during_learning = model.LearnModelFromDataUsingALS(n_epochs=8, train_data=train, test_data=test)

        run_time.append(time.time() - start_time)
        mse_lst.append(min(metrics_during_learning['test_mse']))
        loss_lst.append(min(metrics_during_learning['test_loss']))

    plt.figure()
    plt.plot(d_lst, mse_lst, label='mse', c='blue')
    plt.legend()
    plt.title('als - best test mse as function of vectors dim.  scale=0.01, n_epochs=8, reg=0.1')
    plt.xlabel('vectors dimension')
    plt.ylabel('mse')
    plt.savefig('plots\\deliverable_3\\als - best test mse as function of vectors dim.  scale=0.01, n_epochs=8, reg=0.1.png')

    plt.figure()
    plt.plot(d_lst, loss_lst, label='loss', c='blue')
    plt.legend()
    plt.title('als - best test loss as function of vectors dim.  scale=0.01, n_epochs=8, reg=0.1')
    plt.xlabel('vectors dimension')
    plt.ylabel('loss')
    plt.savefig('plots\\deliverable_3\\als - best test loss as function of vectors dim.  scale=0.01, n_epochs=8, reg=0.1.png')

    plt.figure()
    plt.plot(d_lst, run_time, label='run_time', c='blue')
    plt.legend()
    plt.title('als - run time as function of vectors dim.  scale=0.01, n_epochs=8, reg=0.1')
    plt.xlabel('vectors dimension')
    plt.ylabel('seconds')
    plt.savefig('plots\\deliverable_4\\als -run time as function of vectors dim.  scale=0.01, n_epochs=8, reg=0.1.png')


def deliverable_5():
    model = MFModel(n_users=len(idx2user), n_items=len(idx2item), k=50, reg=(0.1, 0.1, 0.1, 0.1), scale=0.01)
    _ = model.LearnModelFromDataUsingALS(n_epochs=8, train_data=train, test_data=test)
    prediction_matrix = model.predict()

    train_users_lst = train[:, 0]
    users_more_than_3_occ = []
    unique, counts = np.unique(train_users_lst, return_counts=True)
    for user, user_counts in zip(unique, counts):
        if user_counts > 10:
            users_more_than_3_occ.append(user)
        if len(users_more_than_3_occ) == 5:
            break

    for u in users_more_than_3_occ:
        user_path = 'plots\\deliverable_5\\description of predictions for user_id - {}.txt'.format(idx2user[u])

        user_rankings_in_train = train[train_users_lst == u]
        user_rankings_in_test = test[test[:, 0] == u]

        print('Movies that were in the train set:', file=open(user_path, "a"))
        for item in user_rankings_in_train:
            try:
                item_name = all_items_data[idx2item[item[1]]]['item_name']
                truth_rank = item[2]
                predicted_rank = prediction_matrix[u, item[1]]
                print('\tMovie name: {}, Truth rank value: {}, Predicted rank by the model: {}'.
                      format(item_name, truth_rank, predicted_rank), file=open(user_path, "a"))
            except:
                pass

        print('_______________________________________________________________', file=open(user_path, "a"))

        print('Movies that were in the test set:', file=open(user_path, "a"))
        for item in user_rankings_in_test:
            try:
                item_name = all_items_data[idx2item[item[1]]]['item_name']
                truth_rank = item[2]
                predicted_rank = prediction_matrix[u, item[1]]
                print('\tMovie name: {}, Truth rank value: {}, Predicted rank by the model: {}'.
                      format(item_name, truth_rank, predicted_rank), file=open(user_path, "a"))
            except:
                pass



# deliverable_1()
# deliverable_2()
# deliverable_3and4()
deliverable_5()

