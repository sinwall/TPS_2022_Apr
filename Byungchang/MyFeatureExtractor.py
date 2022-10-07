from sklearn.base import TransformerMixin, BaseEstimator
# Import more libraries if needed

class MyFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, arg1, arg2):
        # Initialize your extractor here
        # __init__ may be omitted if your extractor does not require parameters
        super(MyFeatureExtractor, self).__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        
    def fit(self, X):
        # Fit your extractor
        # Assume X is a dataframe of shape (n_sequences * n_steps(=60), n_channels(=13))
        # i.e. the form obtained just after reading train.csv
        
        
        # and return self in the end
        return self
    
    def transform(self, X, y=None):
        # Transform X
        # Assume X is a dataframe of shape (n_sequences * n_steps(=60), n_channels(=13))
        # i.e. the form obtained just after reading train.csv
        
        # In the end, return a dataframe such that:
        # - the shape is (n_sequences, n_features)
        # - the index values are equal to the sequence id of corresponding sequence
        raise NotImplementedError
    # You don't need to implement fit_transform, as it is inherited from TransformerMixin.


