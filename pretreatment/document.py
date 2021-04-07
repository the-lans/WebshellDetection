from abc import ABCMeta, abstractmethod
from math import ceil
from collections import defaultdict
import numpy as np


class Document(metaclass=ABCMeta):
    def __init__(self, batch_size, vectorization_config, *parsers):
        self._batch_size = batch_size
        self._parsers = parsers
        self._embedding = vectorization_config.embedding
        self._vectorization_config = vectorization_config
     
    @property
    def num_batches(self):
        return ceil(self._num_samples() / self._batch_size)

    @abstractmethod
    def _all_samples(self): return []

    @abstractmethod
    def _num_samples(self): return 0

    def _parse(self, lines):
        results = defaultdict(list)
        for line in lines:
            for parser in self._parsers:
                vector = parser.vector_after_padding(
                    line, self._vectorization_config)
                results[parser.name].append(vector)
        return {name: np.stack(arrays, axis=0) for name, arrays in results.items()}

    def _to_batches(self, lines):
        for batch_index in range(self.num_batches):
            batch_index_start = batch_index * self._batch_size
            batch_index_end = batch_index_start + self._batch_size
            lines_this_batch = lines[batch_index_start:batch_index_end]
            yield lines_this_batch

    def load(self, never_stop=True):
        while True:
            lines = self._all_samples()
            for lines_this_batch in self._to_batches(lines):
                yield self._parse(lines_this_batch)
            if not never_stop:
                break

    def load_all(self):
        lines = self._all_samples()
        return self._parse(lines)


class ItertorDocument(Document):
    def __init__(self, itertor, batch_size, vectorization_config, *parsers):
        self._itertor = itertor
        self._batch_size = batch_size
        self._parsers = parsers
        self._embedding = vectorization_config.embedding
        self._vectorization_config = vectorization_config
        super(ItertorDocument, self).__init__(
            batch_size, vectorization_config, *parsers)

    def _all_samples(self):
        return list(self._itertor)

    def _num_samples(self):
        return len(self._all_samples())


class FileDocument(ItertorDocument):
    def __init__(self, data_file, batch_size, vectorization_config, *parsers):
        itertor = open(data_file).readlines()
        super(FileDocument, self).__init__(
            itertor, batch_size, vectorization_config, *parsers)
