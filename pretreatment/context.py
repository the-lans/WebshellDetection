from pretreatment.word2vec_loader import gen_word2vec_model, load_word2vec_model
from pretreatment.document import FileDocument
from os import path
from pretreatment.embedding import Embeding
from typing import List
from copy import copy
from . import tool
from collections import namedtuple

VectorizationConfig = namedtuple(typename="VectorizationConfig", field_names=[
                                 "embedding", "splitter"])


class Context:

    @staticmethod
    def extend(other):
        return copy(other)

    @staticmethod
    def use(other):
        return copy(other)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __init__(self):
        self._sentences_files = []
        self._word2dev_model_name = None
        self._embedding = None
        self._parsers = []
        self._batch_size = 50
        self._resource = None
        self._splitter = None
        self._resource_type = FileDocument

    def set_sentences_files(self, *file_name):
        self._sentences_files = file_name

    def set_word2dev_model_file(self, file_name):
        self._word2dev_model_name = file_name

    def set_resource(self, resource):
        self._resource = resource

    def set_resource_type(self, resource_type):
        self._resource_type = resource_type

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def set_splitter(self, splitter):
        self._splitter = splitter

    def set_line_parser(self, *parsers):
        self._parsers = parsers

    def __str__(self):
        return str(vars(self))

    @property
    def word2dev_model_name(self): return self._word2dev_model_name

    @property
    def embedding(self):
        if not path.exists(self._word2dev_model_name):
            gen_word2vec_model(
                self._splitter, self._word2dev_model_name, *self._sentences_files)
        if not self._embedding:
            word2vec_model = load_word2vec_model(self._word2dev_model_name)
            self._embedding = Embeding(word2vec_model, False)
        return self._embedding

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def document(self):
        if not self._resource:
            raise Exception("请设置资源!!")
        return self._resource_type(self._resource,
                                   self._batch_size,
                                   VectorizationConfig(
                                       self.embedding, self._splitter),
                                   *self._parsers
                                   )

    @property
    def resource(self): return self._resource


    def copy(self):
        return copy(self)


class ContextManager:
    def __init__(self):
        self._contexts = {}

    def new(self, name) -> Context:
        self._contexts[name] = Context()
        return self._contexts[name]

    def extend_as(self, name, as_name) -> Context:
        context = self._get_context(name).copy()
        self._contexts[as_name] = context
        return self._contexts[as_name]

    def _get_context(self, name) -> Context:
        try:
            return self._contexts[name]
        except KeyError:
            print(f"context: {name} 不存在!")
            exit()

    def __call__(self, name) -> Context:
        return self._get_context(name)
