import enum


class ChainType(str, enum.Enum):
    SUMMARIZE = "summarize"
    STUFF = "stuff"
    MAP = "map_reduce"


class SummarizeType(str, enum.Enum):
    STUFF = "stuff"
    MAP = "map_reduce"
    REFINE = "refine"
