from Datasets.Dataset import AbsractDataFrameDataset
import pandas as pd
import numpy as np
from sklearn import datasets


class IrisDataset(AbsractDataFrameDataset):
    def __init__(self):
        self.name = 'iris'
        super(IrisDataset, self).__init__()

    def load(self):
        iris = datasets.load_iris()
        iris_df = pd.DataFrame(data=np.concatenate((iris.data, iris.target.reshape(-1, 1)), axis=1)
                               , columns=['Petal length', 'Petal Width', 'Sepal Length', 'Sepal Width', 'target'])
        self.df = iris_df
