from gensim.models.word2vec import Word2Vec
import gensim
import os
from pretreatment import tool

class Sentences:
    def __init__(self, spliter, *file_names):
        self._spliter = spliter
        self._file_names = file_names
    
    def __iter__(self):
        for name in self._file_names:
            with open(name, 'rb') as _file:
                for line in _file:
                    yield self._spliter(str(line).strip("\r\n"))
                    
def gen_word2vec_model(spliter,save_file_name,*file_names):
    sentences = Sentences(spliter,*file_names)
    model = Word2Vec(sentences, size=150, min_count=20)
    tool.ensure_dir_exist(save_file_name)
    model.save(save_file_name)

def load_word2vec_model(file_name):
    return Word2Vec.load(file_name)

if __name__ == "__main__":
    pass
