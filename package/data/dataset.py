
class DataSet(object):
    def __init__(self):
        self.data_size = None 
        self.features = None 
        self.targets = None 

        self.data = None
        
    def set_features(self, features):
        self.features = features

    def set_targets(self, targets):
        self.targets = targets

    def get_features(self):
        return self.features
    
    def get_targets(self):
        return self.targets
