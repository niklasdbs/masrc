import collections

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from utils.rl.replay.batch import Batch


class EpisodeCollate:
    def __init__(self, n_agents: int, batch_first=True):
        self.n_agents = n_agents
        self.batch_first = batch_first

    def collate(self, batch) -> Batch:
        lengths = np.array(list(map(lambda b: len(b["dones"]), batch)))
        max_sequence_length = np.max(lengths)
        batch_size = len(batch)
        mask = np.ones(shape=(batch_size, max_sequence_length, self.n_agents), dtype=np.long)
        for i in range(batch_size):
            mask[i, (lengths[i]):, :] = 0

        elem = batch[0]
        batch = {key: recursive_pad_sequence([default_collate(d[key]) for d in batch], batch_first=self.batch_first) for key in
                 elem}
        batch["mask"] = default_collate(mask)
        return Batch(batch,
                     _batch_first=self.batch_first,
                     _batch_size=batch_size,
                     _max_sequence_length=max_sequence_length)

    def __call__(self, batch):
        return self.collate(batch)


def recursive_pad_sequence(batch, batch_first=True):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return pad_sequence(batch, batch_first=batch_first)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: recursive_pad_sequence([d[key] for d in batch], batch_first=batch_first) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(recursive_pad_sequence(samples, batch_first=batch_first) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [recursive_pad_sequence(samples, batch_first=batch_first) for samples in transposed]
