'''
Available methods are the followings:
[1] fuzzy_cmeans
[2] _cmeans_
[3] _cmeans_predict

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 06-11-2020
'''

import numpy as np
from warnings import warn
from scipy.spatial.distance import cdist

__all__ = ['fuzzy_cmeans','_cmeans_','_cmeans_predict']

def _cmeans_(X, U, m=2, metric='euclidean'):
    
    '''
    Single step in generic fuzzy c-means clustering 
    algorithm. This method is modified from 
    `fuzz.cluster._cmeans0()`.
    
    .. versionadded:: 05-11-2020
    
    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
    \t Training instances to cluster.
        
    U : 2d-array, shape of (n_sample, n_cluster)
    \t Fuzzy c-partitioned matrix.
        
    m : `float`, optional, default:2
    \t Weighting parameter.

    metric: `str`, optional, default:'euclidean'
    \t The distance metric to use. Passes any option 
    \t accepted by `scipy.spatial.distance.cdist`.

    Returns
    -------
    cntr : 2d-array, shape of (n_cluster, n_feature)
    \t Cluster centers.  
        
    u : 2d-array, shape of (n_sample, n_cluster)
    \t Updated fuzzy c-partitioned matrix.
        
    jm : `float`
    \t Objective function.
    '''
    # Normalize "fuzzy c-partition" then eliminating 
    # any potential zero values.
    U0 = U.T.copy()
    U0 = U0/np.sum(U0, axis=0)
    U0 = np.fmax(U0, np.finfo(np.float64).eps)

    # Compute cluster centers.
    um = U0**m
    cntr = um.dot(X)/um.sum(axis=1).reshape(-1,1)

    # Calculate distance between kth instance and clusters 
    # (default is "Euclidean distance") and make sure that 
    # there is no "0" in distance array. Any "0" will 
    # cause error when new "fuzzy partition" is computed.
    d = cdist(X, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    # Objective function for a fuzzy c-partition.
    jm = (um*d.T**2).sum()

    # Update "fuzzy c-partition".
    c_part = lambda x,c,m: 1/((x[:,[c]]/x)**(2./(m-1))).sum(axis=1)
    u = np.array([c_part(d,c,m) for c in range(U.shape[1])])

    return cntr, u.T, jm

def _cmeans_predict(X, U, cntr, m=2, metric='euclidean'):
    
    '''
    Very similar to initial clustering, except `cntr` is not 
    updated, thus `X` is forced into known (trained) clusters.
    
    .. versionadded:: 05-11-2020
    
    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
    \t Training instances to cluster.
        
    U : 2d-array, shape of (n_sample, n_cluster)
    \t Fuzzy c-partitioned matrix.
    
    cntr : 2d-array, shape of (n_cluster, n_feature)
    \t Cluster centers.
    
    m : `float`, optional, default:2
    \t Weighting parameter.

    metric: `str`, optional, default:'euclidean'
    \t The distance metric to use. Passes any option 
    \t accepted by `scipy.spatial.distance.cdist`.

    Returns
    -------   
    u : 2d-array, shape of (n_sample, n_cluster)
    \t Updated fuzzy c-partitioned matrix.
        
    jm : `float`
    \t Objective function.
    '''
    # Normalize "fuzzy c-partition" then eliminating 
    # any potential zero values.
    U0 = U.T.copy()
    U0 = U0/np.sum(U0, axis=0)
    U0 = np.fmax(U0, np.finfo(np.float64).eps)

    # For prediction, we do not recalculate cluster 
    # centers. X is forced to conform to the prior 
    # clustering.
    d = cdist(X, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)
    
    # Objective function for a fuzzy c-partition.
    um = U0**m
    jm = (um*d.T**2).sum()

    # Update "fuzzy c-partition".
    c_part = lambda x,c,m: 1/((x[:,[c]]/x)**(2./(m-1))).sum(axis=1)
    u = np.array([c_part(d,c,m) for c in range(U.shape[1])])

    return u.T, jm

class fuzzy_cmeans:
    '''
    `Fuzzy c-Means` (also referred to as soft clustering) is 
    a form of clustering in which each instance can belong 
    to more than one cluster.
    
    .. versionadded:: 05-11-2020
    
    Parameters
    ----------
    n_clusters : `int`, optional, default:2
    \t Number of clusters. This is relevant when 
    \t `init` is None.

    init : 2d-array, shape of (n_sample, n_cluster)
    \t Initial fuzzy c-partitioned matrix. If `None` 
    \t provided, algorithm is randomly initialized.

    n_init : `int`, optional, default:10
    \t Number of time the algorithm will be run with 
    \t different initial fuzzy c-partitioned matrix. 
    \t The final results will be the best output of
    \t `n_init` consecutive runs in terms of `jm`.

    m : `float`, optional, default:2
    \t Weighting parameter.

    metric: `str`, optional, default:'euclidean'
    \t The distance metric to use. Passes any option 
    \t accepted by `scipy.spatial.distance.cdist`.

    max_iter : `int`, optional, default:300
    \t Maximum number of iterations of `fuzzy_cmeans` 
    \t algorithm for a single run.

    tol : float, default=1e-4
    \t Tolerance with regards to Frobenius norm of 
    \t the difference in the cluster centers of two 
    \t consecutive iterations to declare convergence.

    random_state : `int`, optional, default:None
    \t Determines random number generation for fuzzy 
    \t c-partitioned matrix initialization.
    
    Attributes
    ----------
    centers_ : 2d-array, shape of (n_cluster, n_feature)
    \t Cluster centers.  

    initial_u : 2d-array, shape of (n_sample, n_cluster)
    \t Initial fuzzy c-partitioned matrix.

    partition : 2d-array, shape of (n_sample, n_cluster)
    \t Lastest fuzzy c-partitioned matrix.

    jm : list of `float`
    \t List of objective functions as per iteration.

    iter_centers_ : `list` of 2D arrays
    \t List of cluster centers as per iteration.

    n_iter : `int`
    \t Number of iterations used in determine cluster
    \t centers.

    labels_ : 1D array of `int`
    \t Index of the centroid the i'th observation is 
    \t closest to (crisp clustering)
    
    inertia_ : `float`
    \t Sum of squared distances of samples to their 
    \t closest cluster center.
    
    References
    ----------
    [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 
        3rd ed. Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, 
        eq 10.28 - 10.35.
        
    Examples
    --------
    >>> from fuzzy import *
    >>> from sklearn.datasets import make_blobs

    >>> X, y = make_blobs(n_samples=500, centers=4, 
    ...                   n_features=2, random_state=9)
    
    >>> model = fuzzy_cmeans(n_clusters=4, m=2)
    >>> model.fit_predict(X)
    
    # Example of attributes.
    >>> model.centers_ # Cluster centers
    >>> model.partition # The latest partition matrix
    >>> model.labels_ # Hard clustering
    >>> self.inertia_ # Within sum of squared distances
    
    # Predict new data.
    >>> new_X, y = make_blobs(n_samples=50, centers=4, 
    ...                       n_features=2, random_state=99)
    
    # Predict `new_X`, This is equivalent to calling 
    # `fit_predict(X)` followed by `predict(new_X)`.
    >>> model.predict(new_X)
    
    # Transform `X` to cluster-distance space. This is 
    # equivalent to calling `fit_predict(X)` followed by
    # `transform(X)`.
    >>> model.transform(X)
    '''
    def __init__(self, n_clusters=2, init=None, n_init=10, m=2, 
                 metric='euclidean', max_iter=300, tol=0.0001, 
                 random_state=None):
       
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.m = m
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _check_params(self, X):
        
        if not hasattr(X,'__array__'):
            raise ValueError(f'X must be 2D array, got {type(X)}.')

        if not isinstance(self.n_clusters,int):
            raise ValueError(f'`n_clusters` must be integer, '
                             f'got {type(n_clusters)} instead.')

        if X.shape[0] < self.n_clusters:
            raise ValueError(f'n_samples={X.shape[0]} must be >= '
                             f'n_clusters={self.n_clusters}.')
        
        if self.n_init <= 0:
            raise ValueError(f'n_init should be > 0, got '
                             f'{self.n_init} instead.')
    
        if hasattr(self.init,'__array__'):
            if self.init.shape[0]!=X.shape[0]:
                raise ValueError('Number of samples of `init` must '
                                 'be {:,}, got {:,}.'
                                 .format(X.shape[0],init.shape[0]))
                
            elif self.n_init != 1:
                warn(f'Explicit initial fuzzy c-partitioned matrix passed: '
                     f'performing only one init in {self.__class__.__name__} '
                     f'instead of n_init={self.n_init}.')
                self.n_init = 1
            self.U = [self.init]
            self.n_clusters = self.init.shape[1]
            
        elif self.init is None:
            k = np.iinfo(np.int32).max
            seeds = np.random.randint(k, size=self.n_init)
            self.U = [self._partition_matrix(X, m) for m in seeds]
        
        else: raise ValueError(f'init must be 2D array, got {type(self.init)}.')

        if self.max_iter <= 0:
            raise ValueError(f'max_iter should be > 0, got '
                             f'{self.max_iter} instead.')

        if self.tol <= 0:
            raise ValueError(f'tol should be > 0, got {self.tol} instead.')
    
    def _partition_matrix(self, X, seed=None):
        
        '''Generate fuzzy c-partitioned matrix given seed.'''
        np.random.seed(seed)
        # Number of samples in `X`.   
        u = np.random.rand(X.shape[0], self.n_clusters)
        u = np.fmax(u.astype(np.float64), np.finfo(np.float64).eps)
        return u/u.sum(axis=1).reshape(-1,1)
    
    def _fuzzy_cMeans(self, X, u):
        
        '''
        Iteration process of computing fuzzy c-partitioned matrix.
        '''
        # Initialize loop parameters.
        Jm, cntr = [], []
        u0 = u.copy()
        n = 0
        
        # Main cmeans loop.
        while n < self.max_iter:
            
            # Compute fuzzy c-means given changing `u0`. The
            # other parameters remain unchanged.
            u1 = u0.copy()
            c, u0, jm = _cmeans_(X, u0, self.m, self.metric)
            
            # Store cluster centers (`cntr`) as well as 
            # objective function value in each iteration.
            cntr.append(c)
            Jm.append(jm)
            n += 1
            
            # The algorithm stops when the partition matrix
            # change is below define tolerance (`tol`),
            # otherwise it keeps adjusting partition matrix
            # until number of iterations is above `max_iter`.
            if np.linalg.norm(u0-u1)<self.tol: break
        
        # Compute sum of squared distances of samples 
        # to their closest cluster center (assuming "Euclidean").
        d = cdist(X, c, metric='euclidean')
        inertia = (np.amin(d, axis=1)**2).sum()

        return c, u1, Jm, cntr, n, inertia
    
    def fit_predict(self, X):
        '''
        Compute fuzzy c-means cluster centers and predict 
        cluster index for each sample (also referred to as 
        hard clustering).

        Parameters
        ----------
        X : 2d-array, shape of (n_sample, n_feature)
        \t Training instances to cluster.
        '''
        self._check_params(X)
        
        # Initialize for-loop parameters.
        best_Jm = None
        
        for u in self.U:
            
            # Determine cluster centers that optimizes
            # the objective function (Jm).
            c, u1, Jm, cntr, n_iter, inertia = \
            self._fuzzy_cMeans(X, u)
            
            if best_Jm is None or Jm[-1] < best_Jm:
                best_Jm = Jm[-1]
                self.centers_ = c.copy()
                self.initial_u = u.copy()
                self.partition = u1.copy()
                self.jm = Jm
                self.n_iter = n_iter
                self.labels_ = np.argmax(u1,axis=1).ravel()
                self.inertia_ = inertia
                centers = cntr.copy()
        
        # Transform `centers`, where each index stores
        # cluster center from every iteration.
        centers = np.array([n.ravel() for n in centers])
        indices = int(centers.shape[1]/X.shape[1])
        self.iter_centers_ = np.hsplit(centers, indices)
        
        return self
   
    def predict(self, X, init=None, return_init=False, 
                return_jm=False, return_inertia=False):
        
        '''
        Predict will used the same settings as trained data
        
        Parameters
        ----------
        X : 2d-array, shape of (n_sample, n_feature)
        \t Training instances to cluster.
        
        init : 2d-array, shape of (n_sample, n_cluster)
        \t Initial fuzzy c-partitioned matrix. If `None` 
        \t provided, algorithm is randomly initialized.
        
        return_init : `bool`, optional, default:False
        \t If `True`, also return Initial fuzzy 
        \t c-partitioned matrix. It is relevant when 
        \t `init` is none or random generation is 
        \t initialized.
        
        return_jm : `bool`, optional, default:False
        \t If `True`, also return list of objective
        \t function values as per iteration.
        
        return_inertia : `bool`, optional, default:False
        \t If `True`, also return sum of squared distances
        \t of all instances.
        
        Returns
        -------
        u1 : 2d-array, shape of (n_sample, n_cluster)
        \t Lastest fuzzy c-partitioned matrix.
        
        labels_ : 1D array of `int`
        \t Index of the centroid the i'th observation is 
        \t closest to.
        
        u0 : 2d-array, shape of (n_sample, n_cluster)
        \t Initial fuzzy c-partitioned matrix. Only provided 
        \t if `return_init` is `True`.
        
        Jm : list of `float`, optional
        \t List of objective functions as per iteration.
        \t Only provided if `return_jm` is `True`.
        
        inertia_ : `float`, optional
        \t Sum of squared distances of samples to their 
        \t closest cluster center. Only provided if
        \t `return_inertia` is `True`.
        '''
        # If init is none, the partition matrix is randomly
        # generated (fixed `random_state`), otherwise it uses
        # `init` that passed.
        if init is None:
            k = np.iinfo(np.int32).max
            seed = np.random.randint(k, size=1)
            u0 = self._partition_matrix(X, seed)
        else: u0 = init.copy()
        
        # Check `u0`.
        if not hasattr(u0,'__array__'):
            raise ValueError(f'init must be 2D array, got {type(u0)}.') 
            
        elif u0.shape[0]!=X.shape[0]:
            raise ValueError('Number of samples of `init` must '
                             'be {:,}, got {:,}.'
                             .format(X.shape[0], u0.shape[0]))
            
        elif u0.shape[1]!=self.centers_.shape[0]:
            raise ValueError(f'Number of clusters of `init` must '
                             f'be {self.centers_.shape[0]}, got '
                             f'{u0.shape[1]}.')
 
        # Initialize loop parameters
        Jm, n = [], 0
        kwgs = dict(cntr=self.centers_, m=self.m, metric=self.metric)
 
        # It would take n=2 for algorithm to stop as `centers_`
        # is fixed and initial partitioned matrix is randomly
        # generated.
        while n < self.max_iter:

            # Compute fuzzy c-means given changing `u0`. The
            # other parameters remain unchanged.
            u1 = u0.copy()    
            u0, jm = _cmeans_predict(X, u0, **kwgs)
            
            # Store objective function value in each iteration.
            Jm.append(jm)
            n += 1

            # The algorithm stops when the partition matrix
            # change is below define tolerance (`tol`),
            # otherwise it keeps adjusting partition matrix
            # until number of iterations is above `max_iter`.
            if np.linalg.norm(u0-u1)<self.tol: break
        
        # Return u1 and Crisp or Hard clustering
        ret = [u1, np.argmax(u1, axis=1)]
        
        # Add other parameters.
        if return_init: ret.append(u0)
        if return_jm: ret.append(Jm)
        if return_inertia:
            # Compute sum of squared distances of samples 
            # to their closest cluster center (assuming "Euclidean").
            d = cdist(X, self.centers_, metric='euclidean')
            inertia = (np.amin(d, axis=1)**2).sum()
            ret.append(inertia)
        
        return tuple(ret)
 
    def transform(self, X):

        '''
        Transform X to a cluster-distance space. In the 
        new space, each dimension is the distance to the 
        cluster centers.  
        
        Parameters
        ----------
        X : 2d-array, shape of (n_sample, n_feature)
        \t Training instances to transform.
            
        Returns
        -------
        X_new : array, shape [n_samples, k]
        \t X transformed in the new space.
        '''
        d = cdist(X, self.centers_, metric='euclidean')
        return np.amin(d, axis=1)