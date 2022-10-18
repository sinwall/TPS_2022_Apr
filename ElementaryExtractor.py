import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.stats import kurtosis
from tsfresh.feature_extraction.extraction import extract_features

class ElementaryExtractor(BaseEstimator, TransformerMixin):
    features_to_use = ['med_abs_val_00',
                       'max_abs_val_00',
                       'sum_abs_diff_00',
                       'l2_sum_00',
                       'l2_sum_diff_00',
                       'l2_sum_diff2_00',
                       'kurt_00',
                       'sm_00',
                       'kurt_diff_00',
                       'mean_01',
                       'med_abs_val_01',
                       'l2_sum_diff2_01',
                       'sm_01',
                       'iqr_diff_01',
                       'mean_02',
                       'med_abs_val_02',
                       'max_abs_val_02',
                       'med_abs_diff_02',
                       'max_abs_diff_02',
                       'l2_sum_diff_02',
                       'l2_sum_diff2_02',
                       'std_02',
                       'kurt_02',
                       'std_diff_02',
                       'iqr_diff_02',
                       'kurt_diff_02',
                       'med_abs_val_03',
                       'med_abs_diff_03',
                       'max_abs_diff_03',
                       'sum_abs_diff_03',
                       'sm_03',
                       'iqr_diff_03',
                       'mean_04',
                       'med_abs_val_04',
                       'max_abs_val_04',
                       'med_abs_diff_04',
                       'max_abs_diff_04',
                       'l2_sum_04',
                       'l2_sum_diff2_04',
                       'iqr_04',
                       'kurt_04',
                       'sm_04',
                       'kurt_diff_04',
                       'mean_05',
                       'med_abs_diff_05',
                       'sum_abs_diff_05',
                       'sm_05',
                       'mean_06',
                       'med_abs_val_06',
                       'med_abs_diff_06',
                       'max_abs_diff_06',
                       'l2_sum_diff2_06',
                       'kurt_06',
                       'iqr_diff_06',
                       'kurt_diff_06',
                       'med_abs_val_07',
                       'sum_abs_diff_07',
                       'l2_sum_07',
                       'l2_sum_diff_07',
                       'l2_sum_diff2_07',
                       'iqr_07',
                       'sm_07',
                       'iqr_diff_07',
                       'kurt_diff_07',
                       'max_abs_diff_08',
                       'sum_abs_diff_08',
                       'l2_sum_08',
                       'l2_sum_diff_08',
                       'l2_sum_diff2_08',
                       'iqr_08',
                       'kurt_08',
                       'iqr_diff_08',
                       'kurt_diff_08',
                       'mean_09',
                       'max_abs_diff_09',
                       'sum_abs_diff_09',
                       'l2_sum_09',
                       'l2_sum_diff2_09',
                       'sm_09',
                       'iqr_diff_09',
                       'kurt_diff_09',
                       'mean_10',
                       'med_abs_val_10',
                       'max_abs_diff_10',
                       'l2_sum_diff2_10',
                       'std_10',
                       'kurt_10',
                       'sm_10',
                       'std_diff_10',
                       'kurt_diff_10',
                       'mean_11',
                       'sum_abs_diff_11',
                       'l2_sum_diff_11',
                       'sm_11',
                       'iqr_diff_11',
                       'kurt_diff_11',
                       'max_abs_diff_12',
                       'sum_abs_diff_12',
                       'l2_sum_12',
                       'l2_sum_diff2_12',
                       'iqr_12',
                       'kurt_12',
                       'sm_12',
                       'kurt_diff_12',
                       'up_sum_02',
                       'up_max_02',
                       'up_mean_02',
                       'down_count_02']

    def fit(self, X):
        return self

    def transform(self, X, y=None):
        seq_no = X['sequence'].iloc[::60]
        x = X.loc[:, 'sensor_00':'sensor_12'].values.reshape(-1, 60, 13)
        features = dict()
        for i in range(13):
            channel = x[:, :, i]
            # mean
            features[f'mean_{i:0>2}'] = np.mean(channel, axis=1)
            # median of absolute values
            features[f'med_abs_val_{i:0>2}'] = np.median(np.abs(channel), axis=1)
            # maximum of absolute values
            features[f'max_abs_val_{i:0>2}'] = np.max(np.abs(channel), axis=1)
            # median of absolute diff
            features[f'med_abs_diff_{i:0>2}'] = np.median(np.abs(np.diff(channel, axis=1)), axis=1)
            # maximum of absolute diff
            features[f'max_abs_diff_{i:0>2}'] = np.max(np.abs(np.diff(channel, axis=1)), axis=1)
            # absolute sum of difference
            features[f'sum_abs_diff_{i:0>2}'] = np.sum(np.abs(np.diff(channel, axis=1)), axis=1)
            # square sum
            features[f'l2_sum_{i:0>2}'] = np.linalg.norm(channel, axis=1)
            # square sum of difference
            features[f'l2_sum_diff_{i:0>2}'] = np.linalg.norm(np.diff(channel, axis=1), axis=1)
            # square sum of 2-diff
            features[f'l2_sum_diff2_{i:0>2}'] = np.linalg.norm(np.diff(np.diff(channel, axis=1), axis=1), axis=1)
            # standard deviation
            features[f'std_{i:0>2}'] = np.std(channel, axis=1)
            features[f'iqr_{i:0>2}'] = np.quantile(channel, 0.75, axis=1) - np.quantile(channel, 0.25, axis=1)
            features[f'kurt_{i:0>2}'] = kurtosis(channel, axis=1)
            features[f'sm_{i:0>2}'] = np.nan_to_num(features[f'std_{i:0>2}'] / np.abs(np.mean(channel, axis=1))).clip(
                -1e30, 1e30)

            features[f'std_diff_{i:0>2}'] = np.std(np.diff(channel, axis=1), axis=1)
            features[f'iqr_diff_{i:0>2}'] = np.quantile(np.diff(channel, axis=1), 0.75, axis=1) - np.quantile(
                np.diff(channel, axis=1), 0.25, axis=1)
            features[f'kurt_diff_{i:0>2}'] = kurtosis(np.diff(channel, axis=1), axis=1)

        sensor_02 = x[:, :, 2]
        features[f'up_count_02'] = np.sum(np.diff(sensor_02, axis=1) >= 0, axis=1)
        features[f'up_sum_02'] = np.sum(np.clip(np.diff(sensor_02, axis=1), 0, None), axis=1)
        features[f'up_max_02'] = np.max(np.clip(np.diff(sensor_02, axis=1), 0, None), axis=1)
        features[f'up_mean_02'] = np.nan_to_num(features[f'up_max_02'] / features[f'up_count_02'], posinf=40)

        features[f'down_count_02'] = np.sum(np.diff(sensor_02, axis=1) < 0, axis=1)
        features[f'down_sum_02'] = np.sum(np.clip(np.diff(sensor_02, axis=1), None, 0), axis=1)
        features[f'down_min_02'] = np.sum(np.clip(np.diff(sensor_02, axis=1), None, 0), axis=1)
        features[f'down_mean_02'] = np.nan_to_num(features[f'down_min_02'] / features[f'down_count_02'], neginf=-40)

        return pd.DataFrame(features, index=seq_no)[self.features_to_use]


class TsfreshExtractor(BaseEstimator, TransformerMixin):
    sensorwise_fcs = [{'agg_autocorrelation': [{'f_agg': 'var', 'maxlag': 40}],
                       'agg_linear_trend': [{'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'}],
                       'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 4, 'k': 10},
                                          {'coeff': 6, 'k': 10}],
                       'augmented_dickey_fuller': [{'attr': 'usedlag'}],
                       'fft_coefficient': [{'coeff': 1, 'attr': 'imag'}],
                       'skewness': [{}],
                       'spkt_welch_density': [{'coeff': 2}]},
                      {'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 1, 'k': 10},
                                          {'coeff': 2, 'k': 10},
                                          {'coeff': 3, 'k': 10},
                                          {'coeff': 4, 'k': 10},
                                          {'coeff': 5, 'k': 10},
                                          {'coeff': 6, 'k': 10},
                                          {'coeff': 7, 'k': 10},
                                          {'coeff': 9, 'k': 10}],
                       'fft_aggregated': [{'aggtype': 'kurtosis'}],
                       'fft_coefficient': [{'coeff': 1, 'attr': 'imag'}],
                       'spkt_welch_density': [{'coeff': 2}],
                       'variation_coefficient': [{}]},
                      {'absolute_sum_of_changes': [{}],
                       'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'var'},
                                            {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'var'},
                                            {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'var'},
                                            {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'max'},
                                            {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'var'}],
                       'change_quantiles': [{'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'},
                                            {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                                            {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
                                            {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'},
                                            {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'},
                                            {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
                                            {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'},
                                            {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                            {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
                                            {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                            {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'},
                                            {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
                                            {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'}],
                       'cid_ce': [{'normalize': True}],
                       'cwt_coefficients': [{'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 2}],
                       'fft_coefficient': [{'coeff': 1, 'attr': 'abs'}],
                       'matrix_profile': [{'threshold': 0.98, 'feature': 'min'}],
                       'partial_autocorrelation': [{'lag': 2}],
                       'permutation_entropy': [{'tau': 1, 'dimension': 4}],
                       'quantile': [{'q': 0.1}],
                       'ratio_value_number_to_time_series_length': [{}],
                       'spkt_welch_density': [{'coeff': 2}],
                       'standard_deviation': [{}],
                       'time_reversal_asymmetry_statistic': [{'lag': 1}]},
                      {'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 4, 'k': 10},
                                          {'coeff': 5, 'k': 10},
                                          {'coeff': 6, 'k': 10},
                                          {'coeff': 7, 'k': 10}],
                       'augmented_dickey_fuller': [{'attr': 'usedlag'}],
                       'fft_coefficient': [{'coeff': 1, 'attr': 'imag'}]},
                      {'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'min'},
                                            {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'},
                                            {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'max'},
                                            {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'var'},
                                            {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'},
                                            {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'min'}],
                       'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 10, 'k': 10},
                                          {'coeff': 2, 'k': 10}],
                       'augmented_dickey_fuller': [{'attr': 'usedlag'}],
                       'autocorrelation': [{'lag': 2}, {'lag': 6}],
                       'cid_ce': [{'normalize': True}],
                       'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 1},
                                                  {'num_segments': 10, 'segment_focus': 3},
                                                  {'num_segments': 10, 'segment_focus': 5},
                                                  {'num_segments': 10, 'segment_focus': 6},
                                                  {'num_segments': 10, 'segment_focus': 7},
                                                  {'num_segments': 10, 'segment_focus': 9}],
                       'fft_aggregated': [{'aggtype': 'kurtosis'}, {'aggtype': 'skew'}],
                       'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},
                                           {'coeff': 0, 'attr': 'real'},
                                           {'coeff': 3, 'attr': 'abs'},
                                           {'coeff': 4, 'attr': 'abs'}],
                       'fourier_entropy': [{'bins': 100}],
                       'friedrich_coefficients': [{'coeff': 1, 'm': 3, 'r': 30},
                                                  {'coeff': 3, 'm': 3, 'r': 30}],
                       'index_mass_quantile': [{'q': 0.2}, {'q': 0.3}, {'q': 0.7}],
                       'kurtosis': [{}],
                       'large_standard_deviation': [{'r': 0.25}],
                       'number_peaks': [{'n': 10}, {'n': 5}],
                       'partial_autocorrelation': [{'lag': 4}, {'lag': 9}],
                       'permutation_entropy': [{'tau': 1, 'dimension': 5}],
                       'ratio_beyond_r_sigma': [{'r': 0.5}, {'r': 1}, {'r': 2}],
                       'skewness': [{}],
                       'spkt_welch_density': [{'coeff': 2}],
                       'time_reversal_asymmetry_statistic': [{'lag': 2}]},
                      {'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 2, 'k': 10},
                                          {'coeff': 4, 'k': 10},
                                          {'coeff': 5, 'k': 10},
                                          {'coeff': 6, 'k': 10}],
                       'cwt_coefficients': [{'widths': (2, 5, 10, 20), 'coeff': 10, 'w': 20}],
                       'fft_aggregated': [{'aggtype': 'kurtosis'}],
                       'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},
                                           {'coeff': 4, 'attr': 'abs'}],
                       'fourier_entropy': [{'bins': 100}],
                       'partial_autocorrelation': [{'lag': 9}],
                       'permutation_entropy': [{'tau': 1, 'dimension': 4}]},
                      {'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'max'}],
                       'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 5, 'k': 10},
                                          {'coeff': 6, 'k': 10}],
                       'fft_coefficient': [{'coeff': 1, 'attr': 'imag'}],
                       'spkt_welch_density': [{'coeff': 2}]},
                      {'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'min'}],
                       'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 1, 'k': 10},
                                          {'coeff': 2, 'k': 10},
                                          {'coeff': 4, 'k': 10},
                                          {'coeff': 5, 'k': 10},
                                          {'coeff': 6, 'k': 10}],
                       'augmented_dickey_fuller': [{'attr': 'usedlag'}],
                       'change_quantiles': [{'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'}],
                       'fft_coefficient': [{'coeff': 1, 'attr': 'abs'},
                                           {'coeff': 1, 'attr': 'imag'}],
                       'number_crossing_m': [{'m': 0}],
                       'skewness': [{}],
                       'spkt_welch_density': [{'coeff': 2}]},
                      {'kurtosis': [{}]},
                      {'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'var'}],
                       'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 3, 'k': 10},
                                          {'coeff': 4, 'k': 10},
                                          {'coeff': 5, 'k': 10},
                                          {'coeff': 6, 'k': 10},
                                          {'coeff': 7, 'k': 10},
                                          {'coeff': 8, 'k': 10}],
                       'augmented_dickey_fuller': [{'attr': 'usedlag'}],
                       'autocorrelation': [{'lag': 6}],
                       'fft_coefficient': [{'coeff': 1, 'attr': 'imag'}],
                       'quantile': [{'q': 0.9}],
                       'spkt_welch_density': [{'coeff': 2}]},
                      {'agg_autocorrelation': [{'f_agg': 'var', 'maxlag': 40}],
                       'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'}],
                       'ar_coefficient': [{'coeff': 0, 'k': 10}, {'coeff': 10, 'k': 10}],
                       'augmented_dickey_fuller': [{'attr': 'pvalue'}, {'attr': 'usedlag'}],
                       'autocorrelation': [{'lag': 1}, {'lag': 2}, {'lag': 5}, {'lag': 6}],
                       'change_quantiles': [{'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'},
                                            {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'}],
                       'cid_ce': [{'normalize': True}],
                       'fft_aggregated': [{'aggtype': 'skew'}],
                       'fft_coefficient': [{'coeff': 4, 'attr': 'abs'}],
                       'fourier_entropy': [{'bins': 100}],
                       'friedrich_coefficients': [{'coeff': 3, 'm': 3, 'r': 30}],
                       'kurtosis': [{}],
                       'linear_trend': [{'attr': 'pvalue'}],
                       'partial_autocorrelation': [{'lag': 3}, {'lag': 4}, {'lag': 9}],
                       'permutation_entropy': [{'tau': 1, 'dimension': 4}],
                       'quantile': [{'q': 0.2}],
                       'spkt_welch_density': [{'coeff': 2}]},
                      {'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 2, 'k': 10},
                                          {'coeff': 4, 'k': 10},
                                          {'coeff': 5, 'k': 10},
                                          {'coeff': 6, 'k': 10},
                                          {'coeff': 7, 'k': 10}],
                       'augmented_dickey_fuller': [{'attr': 'usedlag'}],
                       'fft_aggregated': [{'aggtype': 'kurtosis'}, {'aggtype': 'skew'}],
                       'fft_coefficient': [{'coeff': 1, 'attr': 'imag'}],
                       'spkt_welch_density': [{'coeff': 2}]},
                      {'agg_linear_trend': [{'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'},
                                            {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'min'}],
                       'ar_coefficient': [{'coeff': 0, 'k': 10},
                                          {'coeff': 1, 'k': 10},
                                          {'coeff': 10, 'k': 10},
                                          {'coeff': 2, 'k': 10},
                                          {'coeff': 6, 'k': 10}],
                       'augmented_dickey_fuller': [{'attr': 'usedlag'}],
                       'autocorrelation': [{'lag': 1}, {'lag': 2}],
                       'binned_entropy': [{'max_bins': 10}],
                       'change_quantiles': [{'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'},
                                            {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
                                            {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'}],
                       'fft_aggregated': [{'aggtype': 'kurtosis'}, {'aggtype': 'skew'}],
                       'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},
                                           {'coeff': 1, 'attr': 'abs'},
                                           {'coeff': 22, 'attr': 'abs'},
                                           {'coeff': 23, 'attr': 'abs'},
                                           {'coeff': 24, 'attr': 'abs'},
                                           {'coeff': 25, 'attr': 'abs'}],
                       'fourier_entropy': [{'bins': 100}],
                       'kurtosis': [{}],
                       'partial_autocorrelation': [{'lag': 2}, {'lag': 3}],
                       'ratio_beyond_r_sigma': [{'r': 2}],
                       'spkt_welch_density': [{'coeff': 2}]}]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df_tsf = []
        for i in range(13):
            sensor_name = f'sensor_{i:0>2}'
            ts = X[['sequence', 'step', sensor_name]]
            features = extract_features(
                ts,
                self.sensorwise_fcs[i],
                column_id='sequence',
                column_sort='step'
            )
            df_tsf.append(features)
        df_tsf = pd.concat(df_tsf, axis=1)
        return df_tsf