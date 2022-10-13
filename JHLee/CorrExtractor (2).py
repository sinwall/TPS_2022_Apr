from sklearn.base import TransformerMixin, BaseEstimator

class CorrExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X, y=None):
        def autocorr(x, t=1):
            return np.corrcoef(np.array([x[:-t], x[t:]]))[0,1]
        seq_no = X['sequence'].iloc[::60]
        x = X.loc[:, 'sensor_00':'sensor_12'].values.reshape(-1, 60, 13)
        lenn = x.shape[0]
        features = {}
        for k in range(lenn):
            features[k] = {}
            comat = np.corrcoef(np.transpose(x[k, :, :]))
            dcomat = np.corrcoef(np.transpose(np.diff(x[k, :, :], axis=0)))
            for i in range(13):
                for j in range(13):
                    if i>j:
                        features[k][f'cor_{j:0>2}_{i:0>2}'] = comat[i,j]
                    elif j>i:
                        features[k][f'dcor_{i:0>2}_{j:0>2}'] = dcomat[i,j]
        for k in range(lenn):
            for i in range(13):
                for j in range(10):
                    features[k][f'acor_{i:0>2}_lag{j+1:0>2}'] = autocorr(x[k, :, i], j+1)
                    features[k][f'adcor_{i:0>2}_lag{j+1:0>2}'] = autocorr(np.diff(x[k, :, i], axis=0), j+1)
        return pd.DataFrame(features).transpose().reindex(seq_no).replace(np.nan, 0)