import collections

import torch


class Batch(dict):
    def __init__(self, *args, **kwargs):
        self.batch_first = kwargs.pop("_batch_first", True)
        self.batch_size = kwargs.pop("_batch_size", None)
        self.max_sequence_length = kwargs.pop("_max_sequence_length", None)

        super().__init__(*args, **kwargs)

    def to(self, device: torch.device):
        return Batch(recursive_to(self, device=device),
                     _batch_first=self.batch_first,
                     _batch_size=self.batch_size,
                     _max_sequence_length=self.max_sequence_length
                     )

    def flatten(self,start_dim=0, end_dim=1):
        return Batch(recursive_flatten(self, start_dim=start_dim, end_dim=end_dim),
                     _batch_first=self.batch_first,
                     _batch_size=self.batch_size,
                     _max_sequence_length=self.max_sequence_length
                     )


def recursive_flatten(elem, start_dim=0, end_dim=1):
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return elem.flatten(start_dim=start_dim, end_dim=end_dim)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: recursive_flatten(elem[key], start_dim=start_dim, end_dim=end_dim) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(recursive_flatten(entry, start_dim=start_dim, end_dim=end_dim) for entry in zip(*elem)))
    elif isinstance(elem, collections.abc.Sequence):
        return [recursive_flatten(entry, start_dim=start_dim, end_dim=end_dim) for entry in elem]


def recursive_to(elem, device: torch.device):
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return elem.to(device)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: recursive_to(elem[key], device=device) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(recursive_to(entry, device=device) for entry in zip(*elem)))
    elif isinstance(elem, collections.abc.Sequence):
        return [recursive_to(entry, device=device) for entry in elem]

def recursive_unsqueeze(elem, dim=0):
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return elem.unsqueeze(dim=dim)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: recursive_unsqueeze(elem[key], dim=dim) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(recursive_unsqueeze(entry, dim=dim) for entry in zip(*elem)))
    elif isinstance(elem, collections.abc.Sequence):
        return [recursive_unsqueeze(entry, dim=dim) for entry in elem]
