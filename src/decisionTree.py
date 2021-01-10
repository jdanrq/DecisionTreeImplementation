import numpy as np
np.set_printoptions(precision=4, floatmode='fixed', suppress=True)

class Tree:        
    def __init__(self):
        self.is_leaf = False
        
    def __call__(self, X):
        if self.is_leaf:
            # return  majority  class  label (from  training  data)
            return np.full(len(X), self.value)
        else:
            # else: obtain  results  from  child  nodes (recursion !)
            result = np.full(len(X), np.nan)
            split = self.rule(X)
            result[ split] = self.left( X[ split])
            result[~split] = self.right(X[~split])
            return result            
    
    def split(self, X, Y):
        if self._stop_criterion(X, Y):
            # make  the  node a leaf
            self.is_leaf = True
            classes, counts = np.unique(Y, return_counts=True)
            majority_class = classes[np.argmax(counts)]
            self.value = majority_class
            return self          
       # else: determine  the  best  split  and  recurse
        N, M  = X.shape
        scores = np.full((N, M), np.nan)
        # simply test all possible splits
        for k in range(N):  
            for l in range(M):
                split = X[:, l] <  X[k, l]
                scores[k, l] = self._split_criterion(split, X, Y)
                
        # best split
        k, l = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        self.rule = self._make_rule(l, X[k,l])
        split  = self.rule(X)
        self.left = Tree().split(X[split],  Y[split])
        self.right = Tree().split(X[~split], Y[~split])
        self.is_leaf = False
        return self
    
    @staticmethod
    def _split_criterion(split, X, Y):
        # split  should  be a boolean  array  indicating  wether  the  data  satisfiesthe  selected  rule or not
        if sum(split) in {0, len(Y)}:
            return 0  # all datapoints in one leaf...
        
        classes = np.unique(Y)
        C = np.full((2, len(classes)), np.nan)  # contingency table
        for k, cls in enumerate(classes):
            C[0, k] = np.sum(Y[ split]==cls)
            C[1, k] = np.sum(Y[~split]==cls)
        return Tree._gini_index(C)
        
    @staticmethod
    def _gini_index(C):
        # return Gini-Index given contingency table C
        Cj_ = np.einsum('jk->j',C)
        C_k = np.einsum('jk->k',C)
        C__ = np.einsum('jk->', C)
        C_colnorm = np.einsum('jk, j-> jk', C, 1/Cj_)
        H = np.sum(C_colnorm**2, axis=1)
        return -np.sum((C_k/C__)**2) + (Cj_/C__).dot(H)    
        
    @staticmethod
    def _stop_criterion(X, Y):
        # implement  the  stopping  criterion. keep  splitting  until  either  all  databelongs  to the  same  class or  there is only 1 sample  left
        if len(Y) <= 1 or len(np.unique(Y)) <= 1:
            return True
        return False
    
    @staticmethod
    def _make_rule(idx, val):
        # return  the  splitting  rule (univariate  splits  for  numerical  data)
        return lambda X: X[:, idx] < val
    
class decision_tree:
    def fit(self, X, Y):
        self.classes = np.unique(Y)
        self.tree = Tree().split(X, Y)
        return self
    
    def predict(self, X):
        return self.tree(X).astype(self.classes.dtype)
    
    def score(self, X, Y):
        Yhat = self.predict(X)
        return np.mean(Y==Yhat)
    