from dataset.dataloader.base_dset import BaseDset


class Custom(BaseDset):

    def __init__(self):
        super(Custom, self).__init__()

    def load(self, base_path):
        super(Custom, self).load(base_path)
