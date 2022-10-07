#ver 4
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from pyts.transformation import BagOfPatterns as BOP

class MBOP(BaseEstimator, TransformerMixin):

    """Multivariate Bag of patterns.
    Given multivariate time series , MBOP splits individual time series, BOP-transform 
    them and gather them in one dataframe. further note is documnet of BOP.
    
    BOP
    This algorithm uses a sliding window to extract subsequences from the
    time series and transforms each subsequence into a word using the
    Piecewise Aggregate Approximation and the Symbolic Aggregate approXimation
    algorithms. Thus it transforms each time series into a bag of words.
    Then it derives the frequencies of each word for each time series.

    Parameters of MBOP
    ----------
    
    n_channels : non negative int (default = 13)
        number of time series
        
    m_occur: positive float strictly under 1 (default = 0.01)
        parameter for reduction of dimension of features.
        ignores feature of pattern of trivial occurrence.
        while normal BOP produces features every pattern,
        MBOP will drop column with mean of occurence less than m_occur 
        i.e. patterns that appear less than (sample * m_occur)
    
    
    
    
    window_size : int or float (default = 4)
        Length of the sliding window. If float, it represents
        a percentage of the size of each time series and must be
        between 0 and 1.

    word_size : int or float (default = 4)
        Length of the words. If float, it represents
        a percentage of the length of the sliding window and must be
        between 0. and 1.

    n_bins : int (default = 10)
        The number of bins to produce. It must be between 2 and
        ``min(window_size, 26)``.

    strategy : 'uniform', 'quantile' or 'normal' (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurence of back to back
        identical occurences of the same words.

    window_step : int or float (default = 1)
        Step of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The step of
        sliding window will be computed as
        ``ceil(window_step * n_timestamps)``.
    
    
    norm_mean 
    is not supported for initiallizing MBOP
        

    norm_std : bool (default = True)
    is not supported for initiallizing MBOP

    sparse : bool (default = True)
        Return a sparse matrix if True, else return an array.

    overlapping : bool (default = True)
        If True, time points may belong to two bins when decreasing the size
        of the subsequence with the Piecewise Aggregate Approximation
        algorithm. If False, each time point belong to one single bin, but
        the size of the bins may vary.

    alphabet :
    is not supported for initiallizing MBOP    
    
    """
    def __init__(self, n_channels=13, m_occur=0.01,
                 window_size=4 , word_size=4 ,  n_bins =10, strategy = "quantile" ,  sparse  = False, 
                 numerosity_reduction=True, window_step=1 ,overlapping=True ,print_progress = False ):      
        self.window_size = window_size
        self.word_size = word_size
        self.n_bins = n_bins
        self.strategy = strategy
        self.numerosity_reduction = numerosity_reduction
        self.window_step = window_step
        self.sparse = sparse
        self.overlapping = overlapping    # BOP parameters til here
        self.m_occur= m_occur           # minimum of mean occurence columns with mean lower than minimum occur will be dropped out for size of feature 
        self.n_channels = n_channels
        self.col_list=[]   #list of columns index with nontrivial occurrence called in self.reducer in self.fitting
        self.MACHINES=[]   #stores n_channel Bop machines in this list
        self.idces=[]
        self.ft_X=None
        self.print_progress = print_progress
        if type(self.print_progress)!=bool:
            raise TypeError("has to be boolean")
    
        
    def reducer (self, X,save_trans_X=False): 
        """part of fitting. function used in self.fit
        create instance variable reducing dimension of features.
        
        X :  3d arrary with (sample, time, channel)"""
        self.red_rate = np.zeros((self.n_channels,2))
        trans_X_list=[]
        for i in range(self.n_channels):
            #acutal transform happens here.
            #thus when fitting take save_trans_X= True then we can use self.recycle to recycle this result 
            temp_col=self.MACHINES[i].transform(X[:,:,i])    
            self.col_list.append(temp_col.mean(axis=0)>self.m_occur)
            self.red_rate[i,0] = self.col_list[i].shape[0]
            self.red_rate[i,1] = self.col_list[i].sum()
            if self.print_progress:
                print("{}-th channel finished".format(i))
                print(f"Amount of reduction for {i} is: {self.red_rate[i,0]} to {red_rate[i,1]}")
            if save_trans_X:
                trans_X_list.append(temp_col[:,self.col_list[i]])                
        if save_trans_X:
            self.X_of_fit_list=trans_X_list
        return self
    
        
    def fit(self, X,save_trans_X=False): 
        """
        Fits BOP machines, given suitable dataframe.
        There are n_channel number of different BOP Machines to fit.
        Note that fit() actualy calculates result of fit_transform(X) 
        during it's process.
        Hence if user is willing to save this calculation
        take save_trans_X=True
        then return of fit_transform(X) will be stored as instance
        variable ft_X
        
        X : DataFrame with first column : index,
                           last n_channel column: data of interest 
            For fit takes first and last n_channel-columns of data
            and transform data with index by first column of X.
             (n_samples*time rows, alpha) alpha: integer larger than n_channel.
             
            MBOP considers first column to be index and
            last n_channel columns to be data of interest
            fit will take first column as index of resultant dataframe
            must make sure that X.iloc[:,0] is series of index and
            last n_channel columns store data of interest
        
        Creates
        -------
        (when save_trans_X=True) self.tf_X : result of fit_transform(X)         
        """
        self.MACHINES=[]  
        self.col_list=[]                #resets col_list
        data_3d=X.iloc[:,-self.n_channels :].to_numpy().reshape(-1,60,13)      #separating data and information array and reshaping by (n_sample , -1)
        seq=(X.iloc[:,0].to_numpy().reshape(-1,60))[:,0]  #seq is 1d array 
        for i in range(self.n_channels):
            self.MACHINES.append(BOP(
                window_size=self.window_size, word_size=self.word_size,
            n_bins=self.n_bins, strategy=self.strategy, sparse=self.sparse,
            numerosity_reduction=self.numerosity_reduction,
            window_step=self.window_step, overlapping=self.overlapping))
            self.MACHINES[i].fit(data_3d[:,:,i])
            if self.print_progress: 
                print("{}-th machine fitted".format(i))
        if self.print_progress:
            print("reducing")
        self.reducer(data_3d,save_trans_X=save_trans_X) #makes object variable for transform (collecting index of nontrivial columns)
        if save_trans_X:
            self.ft_X=self.recycle(seq)
            del self.X_of_fit_list
        if self.print_progress:
            print("all fitted")
        return self
    
    def recycle(self,seq=None): 
        """part of fitting
           activates when parameter of fit is True
        """
        if self.print_progress:
            print("fit_transform result has been saved as instance variable ft_X")
        return pd.DataFrame(np.concatenate(self.X_of_fit_list,axis=1),index=seq)
    
    def gods_sake(self):
        print("help me")
        return self
        
    
    def transform(self, X,y=None,train_transform=False):
        """
        Transforms last n_channels-columns of X to (n_smaple, n_feature) DataFrame,
        with index from first column of X.
        If train_transform=True, method will try to find previously calculated
        result while fitting.  
        X : dataframe with first column holding index of X_new(reurn of transform) and
            last-n_channels-columns holding data to transform.
            Need to make sure first and last n_channel columns are correct
            
        y : ignored

        train_transform : If is True and save_trans_X was True when fitting, 
                          retrieves transform result (Default  = False )
                          deletes ft_X
        Returns
        ------
        X_new : dataframe indexed with first column of X (n_samples, n_features)
        """
        if train_transform:
            if type(self.ft_X)!=type(None):
                transform_X=self.ft_X.copy()
                del self.ft_X
                self.ft_X=None
                print("previous calculation ft_X deleted")
                return transform_X

        temp_col_list=[]
        data_3d=X.iloc[:,-self.n_channels :].to_numpy().reshape(-1,60,13)
        seq=(X.iloc[:,0].to_numpy().reshape(-1,60))[:,0]  #seq is 1d array 
        for i in range(self.n_channels):
            temp_col=self.MACHINES[i].transform(data_3d[:,:,i])
            temp_col_list.append(temp_col[:,self.col_list[i]])  #We reduce number of feautures by choosing non trivial patterns with indexing.
            del temp_col
        transform_X=np.concatenate(temp_col_list,axis=1)
        if self.print_progress:
            print("shape={}".format(transform_X.shape))
        del temp_col_list
        return pd.DataFrame(transform_X,index=seq)
        
    
    
    def fit_transform(self,X,y=None):
        """Faster than fitting and transforming"""
        self.fit(X,save_trans_X=True)
        transform_X=self.ft_X.copy()
        del self.ft_X
        return transform_X
    
    
    
    
    
    def refinement(self,trans_train_X,new_m_occur=0.011): 
        """
        method defined for search of better m_occur (larger than m_occur)
        used to find larger m_occur parameter i.e. larger refinement , smaller dimension of feature.
        mostly 1percent works fine
        
        trans_train_X : The transform of X used to fit BOPs.i.e. fit_transform (X).
                        Need to input transform of exactly same dataframe that 
                        has been used for fitting
        new_m_occur  :  float larger than self.m_occur or list of such floats.
                        
        Creates
        -------
        idces : list of indices(Int64Index) corresponding to inputed list or even single new_m_occur (pandas.core.indexes.numeric.Int64Index)
        
        
        Example
        --------
        Being Int64Index and being a subindex of columns of fit_transform(X), can input directly.
        >>train_X=fit_transform(train_X)
        >>test_X=transform(test_data)
        >>for i in self.idces:
        >>    clf.fit(train_X[i],train_y)
        >>    clf.score(test_X[i],test_y)
        """
        self.idces=[]
        try:
            for i in new_m_occur:
                if i<self.m_occur:
                    print("new minimum occurrence has to be larger than previous one")
                    pass
                else:
                    self.idces.append((trans_train_X.mean()[(trans_train_X.mean()>i)
                                                  ]).index)
            print("instance variable created: idces list of new minimum mean occurence")
        except TypeError:
            if new_m_occur<self.m_occur:
                print("new minimum occurrence has to be larger than previous one")
                return None
            else:
                print("instance variable created: list with single new minimum mean occurence")
                self.idces.append((trans_train_X.mean()[(trans_train_X.mean()>new_m_occur)
                                                                              ]).index) 