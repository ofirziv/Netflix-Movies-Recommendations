import numpy as np


def rmse(prediction_matrix, test_set):
    n = len(test_set)
    users = test_set[:, 0]
    items = test_set[:, 1]
    ranks = test_set[:, 2]

    residuals = ranks - prediction_matrix[users, items]
    mse = np.sum(residuals * residuals) / n

    return np.square(mse)


def mpr(prediction_matrix, test_set):
    # list of unique users in the test set
    users = list(set(test_set[:, 0]))
    total = 0

    n_items_total = prediction_matrix.shape[1]

    for user in users:
        best_k_ranked_items = np.argsort(-prediction_matrix[user])
        items_in_test_for_user = test_set[test_set[:, 0] == user][:, 1]

        total_rank_for_user = 0
        for item in items_in_test_for_user:
            rank, = np.where(best_k_ranked_items == item)
            total_rank_for_user += rank / n_items_total
        average_rank_for_user = total_rank_for_user / len(items_in_test_for_user)
        total += average_rank_for_user

    return total / len(users)


def precision_at_k(prediction_matrix, test_set, k):
    # list of unique users in the test set
    users = list(set(test_set[:, 0]))

    total = 0   # variable to store the sum of all users
    for user in users:
        best_k_ranked_items = np.argsort(-prediction_matrix[user])[:k]
        items_in_test = test_set[test_set[:, 0] == user][:, 1]
        tp = np.sum([1 for x in best_k_ranked_items if x in items_in_test])

        total += tp/k

    return total / len(users)


def recall_at_k(prediction_matrix, test_set, k):
    # list of unique users in the test set
    users = list(set(test_set[:, 0]))

    total = 0   # variable to store the sum of all users
    for user in users:
        best_k_ranked_items = np.argsort(-prediction_matrix[user])[:k]
        items_in_test_for_user = test_set[test_set[:, 0] == user][:, 1]
        n_items_in_test_for_user = len(items_in_test_for_user)
        tp = np.sum([1 for x in best_k_ranked_items if x in items_in_test_for_user])

        total += tp/n_items_in_test_for_user

    return total / len(users)


def mean_average_precision(prediction_matrix, test_set):
    # list of unique users in the test set
    users = list(set(test_set[:, 0]))

    n_items_total = prediction_matrix.shape[1]

    total = 0  # variable to store the sum of all users
    for user in users:
        sorted_items = np.argsort(-prediction_matrix[user])  # the minus is in order to sort in descending order
        items_in_test_for_user = test_set[test_set[:, 0] == user][:, 1]
        n_items_in_test_for_user = len(items_in_test_for_user)

        # iterate over all values between 1 and the number of items in the data
        recall_at_i_minus_1 = 0  # initialize for the first iteration
        for i in range(1, n_items_total):
            tp_at_i = np.sum([1 for x in sorted_items[:i] if x in items_in_test_for_user])

            precision_at_i = tp_at_i / i
            recall_at_i = tp_at_i / n_items_in_test_for_user

            total += precision_at_i * (recall_at_i - recall_at_i_minus_1)

            recall_at_i_minus_1 = recall_at_i  # update for the next iteration

    return total / len(users)
