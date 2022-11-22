import numpy as np
import pandas as pd

'''function that splits authors into prolific_author_list (y) and coauthor_list (one variable from X)'''
def generate_coauthors(authors): 
    prolific_author_list = []
    coauthor_list = []
    for i in range(len(authors)):
        if authors[i] < 100: 
            prolific_author_list.append(authors[i])
        else: 
            coauthor_list.append(authors[i])

    return prolific_author_list, coauthor_list

'''Function for converting the index from each feature into a one-hot encoding vector style'''
def onehot_func(bag_len, data, feature):
    instance_amount = data.shape[0]
    for i in range(instance_amount):
        instance_one_hot = np.zeros(bag_len).astype(int)
        try:
            for j in range (len(data[feature][i])):
                instance_one_hot[data[feature][i][j] - 1] = 1
        except:
            if data[feature][i] != "":
                instance_one_hot[data[feature][i]] = 1
        data[feature][i] = instance_one_hot
    return data

'''Function for selecting the coauthors' one-hot encoding vector'''
def coauthors_onehot(prolific_num, data, feature):
    instance_amount = data.shape[0]
    for i in range(instance_amount):
        data[feature][i] = data[feature][i][prolific_num: ]
    return data

'''Function for converting the index from one-hot encoding vector style to actual author IDs as output'''
def decode_func(y_pred): 
    output = []
    for y_hat in y_pred.toarray():
        if sum(y_hat) == 0:
            output.append(-1)
        else: 
            non_zeros = []
            for i in range(len(y_hat)): 
                if y_hat[i] == 1:
                    non_zeros.append(i+1)
            output.append(' '.join(map(str, non_zeros)))
    return output

def convert_to_csv(y_pred_ids): 
    '''Convert output to csv'''
    output = pd.DataFrame(columns=["Id", "Predict"])
    output['Id'] = range(len(y_pred_ids))
    output["Predict"] = y_pred_ids
    output.to_csv('output.csv', index=False)

'''Function for splitting the data into different sub-dataframe by venue'''
def train_grouped_df(train_y, train_X, num):
    train_grouped_dflist = []
    train_y_dflist = []
    for i in range(num):
        train_grouped_dflist.append(train_X.loc[train_X['venue'] == i])
        train_y_dflist.append(train_y.iloc[list(train_grouped_dflist[i].index)])
    train_grouped_dflist.append(train_X.loc[train_X['venue'] == ""])
    train_y_dflist.append(train_y.iloc[list(train_grouped_dflist[-1].index)])
    return train_grouped_dflist, train_y_dflist

'''Function for splitting profilic authors and coauthors in each sub-dataframe'''
def sub_split_authors(trainX_dflist, trainy_dflist):
    for i in range(len(trainX_dflist)):
        prolific_authors_list = []
        coauthors_list = []
        for authors in trainy_dflist[i]["authors"]:
            prolific_authors, coauthors = generate_coauthors(authors)
            prolific_authors_list.append(prolific_authors)
            coauthors_list.append(coauthors)

        trainy_dflist[i]["authors"] = prolific_authors_list
        trainX_dflist[i]["coauthors"] = coauthors_list
    return trainX_dflist, trainy_dflist

'''Function for converting the index from each feature into a one-hot encoding vector style of the sub-dataframe'''
def sub_onehot_func(bag_len, data, feature):
    instance_amount = list(data[feature].index)
    for i in instance_amount:
        instance_one_hot = np.zeros(bag_len).astype(int)
        try:
            for j in range (len(data[feature][i])):
                instance_one_hot[data[feature][i][j] - 1] = 1
        except:
            if data[feature][i] != "":
                instance_one_hot[data[feature][i]] = 1
        data[feature][i] = instance_one_hot
    return data

'''Function for selecting the coauthors' one-hot encoding vector of the sub-dataframe'''
def sub_coauthors_onehot(prolific_num, data, feature):
    instance_amount = list(data[feature].index)
    for i in instance_amount:
        data[feature][i] = data[feature][i][prolific_num: ]
    return data
    
def sub_onehot(trainX_dflist, train_y_dflist, wordbag_len, authors_num, prolific_num, venue_num):
    for i in range(len(trainX_dflist)):
        # trainX_dflist[i] = sub_onehot_func(wordbag_len, trainX_dflist[i], "title")
        # trainX_dflist[i] = sub_onehot_func(wordbag_len, trainX_dflist[i], "abstract")
        trainX_dflist[i] = sub_onehot_func(authors_num, trainX_dflist[i], "coauthors")
        trainX_dflist[i] = sub_coauthors_onehot(prolific_num, trainX_dflist[i], "coauthors")
        train_y_dflist[i] = sub_onehot_func(prolific_num, train_y_dflist[i], "authors")
    return trainX_dflist, train_y_dflist