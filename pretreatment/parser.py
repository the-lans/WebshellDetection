from . import tool


class LineParser:
    def __init__(self, name, max_length, method):
        self._name = name
        self._max_length = max_length
        self._method = method

    @property
    def name(self): return self._name

    def __call__(self, line, vectorization_config):
        # words = vectorization_config.splitter(line)
        # words = tool.cutoff(words, self.max_length)
        return self._method(line, vectorization_config.embedding)

    @property
    def max_length(self): return self._max_length

    def vector_after_padding(self, line, vectorization_config):
        vector = self.__call__(line, vectorization_config)
        # 由于需要在加上起始符号(对于encoder)以及结尾符号(decoder)， 所以实际的 padding 长度 = 最大词长度 + 1
        return vector


def line_parser(name, max_length):
    def wrapper(method):
        return LineParser(name, max_length, method)
    return wrapper
