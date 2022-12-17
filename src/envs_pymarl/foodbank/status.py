from enum import IntEnum


class StockRemaining(IntEnum):
    FULL = 0
    MANY = 1
    FEW = 2
    NONE = 3


class StockChange(IntEnum):
    NONE = 0
    SLIGHTLY = 1
    SOMEWHAT = 2
    GREATLY = 3


class Satisfaction(IntEnum):
    HARDLY = 0
    SOMEWHAT = 1
    COMLETELY = 2
    OVERLY = 3


class Progress(IntEnum):
    ONGOING = 0
    DONE = 1
