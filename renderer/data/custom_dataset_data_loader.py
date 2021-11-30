import torch

def CreateDataset(opt):
    from data.video_dataset import videoDataset
    dataset = videoDataset()
    dataset.initialize(opt)
    return dataset

class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, start_idx):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.sampler = MySequentialSampler(self.dataset, start_idx) if opt.serial_batches else None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            sampler=self.sampler,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

class MySequentialSampler(torch.utils.data.Sampler):
    """Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
        start_idx (int): the point of dataset to start from
    """
    def __init__(self, data_source, start_idx):
        self.data_source = data_source
        self.start_idx = start_idx

    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))

    def __len__(self):
        return len(self.data_source) - self.start_idx

def CreateDataLoader(opt, start_idx=0):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt, start_idx)
    return data_loader
