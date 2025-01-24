from typing import Union

from package.data.dataset import DataSet
from sklearn.model_selection import train_test_split


class DataManager(object):
    def __init__(self):
        self.train_dataset = DataSet()
        self.val_dataset = DataSet()
        self.test_dataset = DataSet()

    def split_data_train_test(self, dataset, test_size=0.25, random_state=42, shuffle=True):
        train_features, test_features, train_labels, test_labels = train_test_split(dataset.features, 
                                                                                    dataset.targets, 
                                                                                    test_size=test_size, 
                                                                                    random_state=random_state, 
                                                                                    shuffle=shuffle)
        
        self.train_dataset.set_features(train_features)
        self.train_dataset.set_targets(train_labels)
        self.test_dataset.set_features(test_features)
        self.test_dataset.set_targets(test_labels)

    def split_data_train_val_test(self, dataset, train_size=0.7, val_test_prop=0.5, random_state=42, shuffle=True):
        """This function should give the validation set in addition to training and test sets

        Parameters
        ----------
        dataset : _type_
            _description_
        val_size : _type_
            _description_
        test_size : _type_
            _description_
        random_state : int, optional
            _description_, by default 42
        shuffle : bool, optional
            _description_, by default True
        """
        train_features, tmp_features, train_labels, tmp_labels = train_test_split(dataset.features, 
                                                                                  dataset.targets, 
                                                                                  test_size=1-train_size, 
                                                                                  random_state=random_state, 
                                                                                  shuffle=shuffle)
        
        val_features, test_features, val_labels, test_labels = train_test_split(tmp_features, 
                                                                                tmp_labels, 
                                                                                test_size=val_test_prop, 
                                                                                random_state=random_state, 
                                                                                shuffle=shuffle)
        

        self.train_dataset.set_features(train_features)
        self.train_dataset.set_targets(train_labels)
        self.val_dataset.set_features(val_features)
        self.val_dataset.set_targets(val_labels)
        self.test_dataset.set_features(test_features)
        self.test_dataset.set_targets(test_labels)

    def split_by_index(self, dataset, train_indices: list, test_indices: list, val_indices: Union[None, list]=None):
        self.train_dataset.set_features(dataset.features[train_indices])
        self.train_dataset.set_targets(dataset.targets[train_indices])
        self.test_dataset.set_features(dataset.features[test_indices])
        self.test_dataset.set_targets(dataset.targets[test_indices])
        if val_indices is not None:
            self.val_dataset.set_features(dataset.features[val_indices])
            self.val_dataset.set_targets(dataset.targets[val_indices])

    def get_train_data(self):
        return self.train_dataset
    
    def get_val_data(self):
        return self.val_dataset

    def get_test_data(self):
        return self.test_dataset
