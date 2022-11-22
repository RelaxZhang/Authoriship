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

'''Function for converting the predicted result into a output csv format for submission'''
def convert_to_csv(y_pred_ids): 
    '''Convert output to csv'''
    output = pd.DataFrame(columns=["Id", "Predict"])
    output['Id'] = range(len(y_pred_ids))
    output["Predict"] = y_pred_ids
    output.to_csv('output.csv', index=False)