

class WrappersDataset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]