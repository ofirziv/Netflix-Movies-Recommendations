import numpy as np
np.random.seed(0)


def read_ratings_data(rating_path):
    """
    Reads the rating data.
    :param rating_path: path to the ratings.dat file
    :return: 1. rating array where each row is [<user_id>, <item_id>, <ranking>]
             2. a mapping from indexes to the relevant users.
             3. a mapping from indexes to the relevant items.
    """
    with open(rating_path) as f:
        ratings = np.array([[int(val) for val in x.strip().split('::')[:3]] for x in f.readlines()])

    users_ids = list(set(ratings[:, 0]))
    userid2idx = dict(zip(users_ids, range(len(users_ids))))

    items_ids = list(set(ratings[:, 1]))
    itemis2idx = dict(zip(items_ids, range(len(items_ids))))

    ratings[:, 0] = [userid2idx[id] for id in ratings[:, 0]]
    ratings[:, 1] = [itemis2idx[id] for id in ratings[:, 1]]

    # reverse the mappings. will be useful to determine the real id from the idexes
    idx2user = {v: k for k, v in userid2idx.items()}
    idx2item = {v: k for k, v in itemis2idx.items()}

    return ratings, idx2user, idx2item


def read_items_data(items_path):
    """

    :param items_path: path to the movies.dat file
    :return:
    """
    with open(items_path) as f:
        content = f.readlines()
        items_data = {}
        for item in content:
            item_data = {}
            item = item.strip().split('::')
            item_data['item_name'] = item[1]
            item_data['item_genres'] = item[2].split('|')
            items_data[int(item[0])-1] = item_data
    return items_data


def train_test_split(rating_data):
    n_ratings = rating_data.shape[0]

    msk = np.random.random(n_ratings) < 0.8

    train = rating_data[msk]
    test = rating_data[~msk]

    return train, test


def print_user_details_by_ind(user_id, train_data, prediction_matrix, idx2item, h, path):
    user_rankings = [ranking for ranking in train_data if ranking[0] == user_id]
    user_ranks = user_rankings[:, 2]
    user_items_names = user_rankings[:, 1]
    user_items_names = [idx2item[idx]['item_name'] for idx in user_items_names]

    print('The user with index {}, ranked the following movies:'.format(user_id), file=open(path, "a"))
    for i, (r, name) in enumerate(zip(user_ranks, user_items_names)):
        print('\t{}. {} - ranking value: {}'.format(i, name, r), file=open(path, "a"))
    print('_____________________________________________', file=open(path, "a"))
    print('', file=open(path, "a"))

    top_h_items_idx = np.argsort(-prediction_matrix[user_id])[:h]
    top_h_names = [idx2item[idx]['item_name'] for idx in top_h_items_idx]
    print('The top {} movies for user {} are:'.format(h, user_id), file=open(path, "a"))
    for i, item in enumerate(top_h_names):
        print('\t{}. {}'.format(i, item), file=open(path, "a"))
    print('', file=open(path, "a"))
    print('', file=open(path, "a"))
    print('', file=open(path, "a"))
    print('', file=open(path, "a"))



