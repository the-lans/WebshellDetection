import numpy as np
from functools import partial


class Embeding:
    """
        对Word2Vec模型进行增强，添加 启用词，结尾词，不常见词
        其编号为已存在的词最大词编号+1,+2,+3
        支持对 keras embeding 层的 padding 操作，enable_padding = True 则会在 embeding 层开启 padding
        此时 0 将作为 padding value
        model: Word2Dev模型
        enable_padding: 是否开启padding
    """

    def __init__(self, model, enable_padding=False):
        feture_vectors = model.wv.vectors

        # 如果开启了 padding 则 将第一个个词(编号0)设为一个无用的 padding_word, 相当于
        # 正常词的编号变为原编号 + 1
        word_index = ['padding_word'] + \
            model.wv.index2entity if enable_padding else model.wv.index2entity

        self._num_word = feture_vectors.shape[0]
        self._word_index = {word: index for index,
                            word in enumerate(word_index)}

        self._enable_padding = enable_padding
        self._index_word = dict(
            zip(self._word_index.values(), self._word_index.keys()))
        self._weight = self.__get_weight(feture_vectors)

    def __get_weight(self, feture_vectors):
        vector_length = feture_vectors.shape[1]
        start_vector = np.ones(vector_length) / 10
        stop_vector = start_vector * 2
        unknow_vector = start_vector * 3
        space_vector = start_vector * 5
        weight = np.vstack([feture_vectors, start_vector,
                            stop_vector, unknow_vector, space_vector])
        # 由于开启padding之后，词编号 + 1，因此这里需要在矩阵头部插入一个全零的向量
        if self._enable_padding:
            padding_vector = np.zeros(vector_length)
            weight = np.vstack([padding_vector, weight])
        else:
            padding_vector = start_vector * 4
            weight = np.vstack([weight, padding_vector])
        return weight

    # 得到三种特殊词编号
    @property
    def start_word(self):
        return self._num_word

    @property
    def stop_word(self):
        return self.start_word + 1

    @property
    def unknow_word(self):
        return self.stop_word + 1

    @property
    def padding_word(self):
        return self.unknow_word + 1

    @property
    def space_word(self):
        return self.padding_word + 1
    
    # 得到词编号
    def get_index(self, word):
        index = self._word_index.get(word, self.unknow_word)
        return index

    # 得到词
    def get_word(self, index):
        return self._index_word.get(index, "")

    def get_word_feature(self, word):
        return self.embding_weight[self.get_index(word), :]

    # 得到添加了三种特殊词的词向量矩阵
    @property
    def embding_weight(self):
        return self._weight

    @property
    def num_words(self):
        return self._num_word + 5

    @property
    def keras_embedding_layer(self):
        keras = self._import("keras")
        weights = self.embding_weight
        layer = keras.layers.Embedding(
            input_dim=weights.shape[0], output_dim=weights.shape[1],
            weights=[weights], trainable=False, mask_zero=self._enable_padding
        )
        return layer

    def _import(self, name):
        try:
            return __import__(name)
        except:
            raise ValueError(f"can't import {name}!")

    @property
    def tf_embedding_layer(self):
        tf = self._import("tensorflow")
        return partial(tf.nn.embedding_lookup, self.embding_weight)

    @property
    def torch_embedding_layer(self):
        torch = self._import("torch")
        weight_tensor = torch.Tensor(self.embding_weight).clone()
        return torch.nn.Embedding.from_pretrained(weight_tensor)

    def embedding_layer(self, app_type):
        layer_dict = {
            "tf": self.tf_embedding_layer,
            "keras": self.keras_embedding_layer
        }
        layer = layer_dict.get(app_type, None)
        if not layer:
            raise ValueError(
                f"only support app type in [{', '.join(layer_dict.keys())}], but receive: {app_type} ")
        return layer
