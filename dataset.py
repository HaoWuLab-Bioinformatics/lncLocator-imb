from torch.utils.data import Dataset
import numpy
import numpy as np
import pandas
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


class LDAMLncAtlasDataset(Dataset):
    def __init__(self, CNRCI_path, feature_path):
        self.data = pd.read_csv(CNRCI_path)
        self.data = self.data[self.data['code'].str.len() < 20000]
        self.data.reset_index(inplace=True, drop=True)

        print('-' * 80)
        print('Loading dataset: ', CNRCI_path)

        self.mu = np.mean(self.data['Value'].values)
        self.sigma = np.std(self.data['Value'].values)

        print('average: ', self.mu)
        print('standard bias', self.sigma)

        print('Processing the filtering...')
        df1 = self.data[self.data['Value'] < -1]
        df2 = self.data[self.data['Value'] > 1]

        self.data = pd.concat([df1, df2])
        self.data.reset_index(inplace=True, drop=True)
        print('dataset size: ', self.data.shape[0])

        self.feature = pandas.read_csv(
            feature_path,
            sep=',',
            header=None,
            index_col=None,
            low_memory=False).values.tolist()
        self.feature = numpy.array(self.feature)
        self.feature = self.feature[1:, 1:]
        print('Load successfully.')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        entry = self.data.loc[idx]
        item['CNRCI'] = float(entry['Value'])
        item['feature'] = self.feature[idx]
        item['code'] = entry['code']
        return item

    def get_labels(self):
        return [int(self.data['Value'][i] > 0)
                for i in range(len(self.data['Value']))]

    def get_cls_num_list(self):
        return np.bincount(self.get_labels())

    def get_feature_len(self):
        return len(self.feature[0])

