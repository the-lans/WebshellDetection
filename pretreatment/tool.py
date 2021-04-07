#from sh import Command
import os
from urllib.parse import unquote
import numpy as np
import pickle
from functools import wraps
import sys
import time
import math

def check_path_exists(position=0, name=None):
    """
        检查参数中关于文件路径的参数是否存在
        position: 文件路径参数为第几个
        name: 文件参数名字
        exp:

        @check_path_exists(position=0,name="path")
        load_file(path)

        class File:
            @check_path_exists(position=1,name="file_name") # 这里的第一个参数为 self , 所以 position = 1
            def read(self,file_name)

    """

    def decorate(method):
        def wrapper(*args, **kwargs):
            if not (args or kwargs):
                raise Exception(
                    "the decorator must use on the method has args")
            path = None
            if name and kwargs != {}:
                path = kwargs[name]
            elif args:
                path = args[position]
            if not path:
                raise Exception("can't not find check arg")
            if not os.path.exists(path):
                raise FileExistsError(f"file '{path}' is not exist!")
            return method(*args, **kwargs)

        return wrapper

    return decorate


def ensure_dir_exist(path):
    """
        确保该路径直到父路径存在
    """
    dir_name = os.path.dirname(path)
    if not dir_name:
        return
    os.path.exists(dir_name) or os.makedirs(dir_name)


def pad_array(array, length_after_pad, pad_value):
    array_length = array.shape[0]
    if array_length >= length_after_pad:
        return array
    pad_length = (0, length_after_pad - array_length)
    return np.pad(array, pad_length, 'constant', constant_values=(pad_value,))


# @check_path_exists(name="file_name")
# def get_num_line(file_name):
#     wc = Command("wc")
#     text = wc("-l", file_name)
#     return int(text.split(" ")[0])


def save_as(array, file):
    ensure_dir_exist(file)
    with open(file, 'w') as _file_:
        _file_.writelines([str(i)+'\n' for i in array])


def one_hot(array, length):
    rows = []
    for index in range(array.shape[0]):
        row = array[index, :]
        arrays = []
        for col_index in row:
            zeros = np.zeros(length)
            zeros[col_index] = 1
            arrays.append(zeros)
        rows.append(np.vstack(arrays))
    return np.stack(rows, axis=0)


def url_decode(url, times=1):
    decode = url
    for _ in range(times):
        decode = unquote(decode)
    return decode


def pickle_dump(obj, filename):
    ensure_dir_exist(filename)
    pickle.dump(obj, open(filename, 'wb'))


@check_path_exists(name="file_name")
def pickle_load(file_name):
    return pickle.load(open(file_name, 'rb'))


def cutoff(list_like, max_length):
    """
        丢弃长度大于 max_length 的部分
    """
    if len(list_like) <= max_length:
        return list_like
    else:
        return list_like[0:max_length]


class ProgressBar:
    def __init__(self, max_val, symbol, arrow=None, full=None, max_length=50):
        self._max_val = max_val
        self._symbol = symbol
        self._max_length = max_length
        self._array = arrow or ""
        self._full = full or "."

    def _gen_bar(self, i):
        i = math.ceil(i / self._max_val * self._max_length)
        return self._symbol * i + self._array + (self._max_length * len(self._symbol) - i) * self._full

    def _precent(self, i):
        return math.ceil((i+1) / self._max_val * 100)
    
    def update(self, i, prefix="", postfix=""):
        sys.stdout.write('\r%s [%s] %s %s%s %s ' % (
            prefix, self._gen_bar(i), "%d/%d"%(i+1, self._max_val), self._precent(i), '%' , postfix))
        if(i == self._max_val - 1):
            sys.stdout.write("\n")

def scores_and_file(scores, filename, desc=True):
    lines = open(filename)
    return map(lambda x: str(x[0]) + " " + str(url_decode(x[1], 2)), sorted(zip(scores, lines), key=lambda x: x[0], reverse=desc))

def die(*args, **kwargs):
    print(*args, **kwargs)
    exit()