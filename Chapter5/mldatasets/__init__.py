import sys
import warnings
import os
from pathlib import Path

if (sys.version_info < (3, 0)):
    warnings.warn("mldatasets only supports Python 3 (not 2)!")

__version__ = '0.01.5'

from mldatasets.config import init, load
from mldatasets.common import runcmd, make_dummies_with_limits, make_dummies_from_dict, evaluate_class_mdl
from mldatasets.sources.kaggle import Kaggle
from mldatasets.sources.url import URL

init(os.path.join(Path().parent.absolute(), 'data'))
