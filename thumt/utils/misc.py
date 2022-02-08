# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_GLOBAL_STEP = 0
_COLLECTIONS = {}
_MSG = {}

def get_global_step():
    return _GLOBAL_STEP


def set_global_step(step):
    global _GLOBAL_STEP
    _GLOBAL_STEP = step


def add_to_collection(name, value):
    if name not in _COLLECTIONS:
        _COLLECTIONS[name] = [value]
    else:
        _COLLECTIONS[name].append(value)


def get_collection(name, clear=True):
    if name not in _COLLECTIONS:
        return None

    col = _COLLECTIONS[name]

    if clear:
        _COLLECTIONS[name] = []

    return col


def send(msg, value):
    global _MSG
    if value is None:
        raise ValueError("the value of %s is None." % msg)

    if msg in _MSG:
        raise ValueError("%s already existed." % msg)

    _MSG[msg] = value


def recieve(msg):
    if msg not in _MSG:
        return None

    val = _MSG[msg]
    del _MSG[msg]

    return val


def clear():
    global _MSG, _COLLECTIONS
    _MSG = {}
    _COLLECTIONS = {}
