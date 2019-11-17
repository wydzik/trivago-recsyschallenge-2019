import numpy as np
import pandas as pd

def split_data(data_set, divider):
    unique_session_ids = data_set.session_id.unique()
    unique_sessions = len(unique_session_ids)
    print(unique_sessions)
    number_of_train_sessions = int(unique_sessions * divider)
    print(number_of_train_sessions)
    print("Before dropping duplicates: " + str(data_set.shape[0]) + " records.")
    data_set.drop_duplicates(subset = ["session_id", "step"], keep = 'first', inplace = True)
    print("After dropping duplicates: "+ str(data_set.shape[0]) + " records.")
    last_train_session_id = unique_session_ids[number_of_train_sessions]
    print(last_train_session_id)
    last_training_index = data_set[data_set.session_id == last_train_session_id][-1:].index[0]
    print(last_training_index)
    first_test_index = last_training_index + 1
    train_set = data_set[:last_training_index + 1]
    test_set = data_set[first_test_index:]
    last_step_indexes = test_set[test_set.step == 1].index - 1
    indexes_to_blur = test_set[(test_set.index.isin(last_step_indexes))
                               & (test_set.action_type == "clickout item")].index
    test_set_blurred = test_set.copy()
    test_set_blurred.loc[(test_set_blurred.index.isin(indexes_to_blur)), 'reference'] = np.nan
    last_index = test_set_blurred[-1:].index[0]
    test_set_blurred.loc[(test_set_blurred.index == last_index)
                         & (test_set_blurred.action_type == "clickout item"), "reference"] = np.nan
    train_set.to_csv('C:/Users/Wydzik/PycharmProjects/trivagoRecSysChallenge/datasets/train.csv')
    test_set.to_csv('C:/Users/Wydzik/PycharmProjects/trivagoRecSysChallenge/datasets/groundTruth.csv')
    test_set_blurred.to_csv('C:/Users/Wydzik/PycharmProjects/trivagoRecSysChallenge/datasets/test.csv')
    return train_set, test_set_blurred, test_set


if __name__ == '__main__':
    data_set = pd.read_csv("C:/Users/Wydzik/Downloads/trivagoRecSysChallengeData2019_v2/train.csv")
    train_set, test_set, ground_truth = split_data(data_set, 0.8)
