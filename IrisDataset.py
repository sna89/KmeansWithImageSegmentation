import pandas as pd
import numpy as np
from sklearn import datasets


class IrisDataset():
    def __init__(self):
        iris = datasets.load_iris()
        iris_df = pd.DataFrame(data=np.concatenate((iris.data, iris.target.reshape(-1, 1)), axis=1)
                               , columns=['Petal length', 'Petal Width', 'Sepal Length', 'Sepal Width', 'target'])
        iris_df = iris_df.drop(columns="target", axis=1)
        self.iris_df = iris_df

    def get_data(self):
        return self.iris_df

