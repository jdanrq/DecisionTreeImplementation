import numpy as np

class Tree:        
    def __call__(self, X):
        if self.is_leaf:
            # return  majority  class  label (from  training  data)
        # else: obtain  results  from  child  nodes (recursion !)
        return prediction
    
    def split(self, X, Y):
       if self._stop_criterion(X, Y):
          # make  the  node a leaf
       # else: determine  the  best  split  and  recurse
       self.rule = best  splitting  rule
       split = self.rule(X)
       self.left = Tree().split(X[split],   Y[split])
       self.right = Tree().split(X[~ split], Y[~ split])
       return  self       
    
    @staticmethod
    def _split_criterion(split, X, Y):
        # split  should  be a boolean  array  indicating  wether  the  data  satisfiesthe  selected  rule or not
        return  gini  index of the  split
    
    @staticmethod
    def _stop_criterion(X, Y):
        # implement  the  stopping  criterion. keep  splitting  until  either  all  databelongs  to the  same  class or  there is only 1 sample  left
        return True/False
    
    @staticmethod
    def _make_rule(idx, val):
        # return  the  splitting  rule (univariate  splits  for  numerical  data)
        return lambda X: X[:, idx] < val
    
class decision_tree:
    def fit(self, X, Y):
        self.tree = Tree().split(X, Y)
        return self
    
    def predict(self, X):
        return self.tree(X)
    
    def score(self, X, Y):
        Yhat = self.predict(X)
        return accuracy of the prediction
    