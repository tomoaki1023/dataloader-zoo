#!/usr/bin/env python
# coding:utf-8

import numpy as np


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = True

        self.idx_list = list(range(len(self.dataset)))
        self.current_index = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.idx_list)
        self.current_index = 0

        return self

    def __next__(self):
        if self.current_index >= len(self.idx_list):
            raise StopIteration

        batch_indices = self.idx_list[self.current_index : self.current_index + self.batch_size]
        self.current_index += self.batch_size

        batch_data, batch_labels = map(list, zip(*[self.dataset[i] for i in batch_indices]))

        return batch_data, batch_labels
