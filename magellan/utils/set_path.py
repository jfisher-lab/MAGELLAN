import os
from os.path import dirname, isdir

# path_root = dirname(dirname(__file__))
try:
    path_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
except NameError:
    path_root = os.getcwd()

# path_data = path_root + '/data/'
path_data = os.path.join(path_root, "data")
# path_result = path_root + '/result/'

path_super_root = dirname(path_root) + "/"

assert isdir(path_data)
# assert isdir(path_result)
