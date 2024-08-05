#!/usr/bin/env python
# coding:utf-8

from multiprocessing import Pool

import numpy as np


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True, num_processes=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_processes = max(1, num_processes)

        self.indices = list(range(len(self.dataset)))
        self.current_index = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        batch_end = min(self.current_index + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_index : batch_end]
        self.current_index += self.batch_size

        if self.num_processes == 1:
            # Single process operation
            batch_items = [self.load_single_item(i) for i in batch_indices]
        else:
            # Multi-process operation
            with Pool(processes=self.num_processes) as pool:
                batch_items = pool.map(self.load_single_item, batch_indices)

        batch_data, batch_labels = map(list, zip(*batch_items))
        return np.array(batch_data, dtype=np.float32), batch_labels

    def load_single_item(self, index):
        return self.dataset[index]
