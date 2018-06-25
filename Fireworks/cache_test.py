from Fireworks import cache
from Fireworks.message import Message
import torch
import numpy as np
import pandas as pd
tensors = {
    'a': torch.Tensor([1,2,3]),
    'b': torch.Tensor([4,5,6]),
    }
vectors = {
    'c': np.array([7,8,9]),
    'd': np.array([10,11,12]),
}
tensors2 = {
    'a': torch.Tensor([13,14,15]),
    'b': torch.Tensor([16,17,18]),
    }
vectors2 = {
    'c': np.array([19,20,21]),
    'd': np.array([22,23,24]),
}

def test_slice_to_list():
    a = slice(0,10)
    l = cache.slice_to_list(a)
    assert l == [0,1,2,3,4,5,6,7,8,9]
    a = slice(0,10,2)
    l = cache.slice_to_list(a)
    assert l == [0,2,4,6,8]

def test_init():
    m = cache.DummyCache(10)
    assert m.max_size == 10
    dummy_message = Message(tensors, vectors)
    assert len(m) == 0
    m[0] = dummy_message[0]
    assert len(m) == 1
    m[3:6] = dummy_message
    assert len(m) == 4
    assert False

def test_insert(): pass

def test_free(): pass