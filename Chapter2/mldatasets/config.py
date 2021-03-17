import os
import json
import tempfile
from pathlib import Path
import warnings
from .sources.kaggle import Kaggle
from .sources.url import URL

dssave_path = os.path.join(tempfile.gettempdir(), 'data')
dsconfig_path = os.path.join(Path(__file__).parent.absolute(), 'dsconfig.json')
dsconfig = None
dsconfig_dscount = 0

def init(new_dssave_path = None, new_dsconfig_path = None):
    global dssave_path, dsconfig_path, dsconfig, dsconfig_dscount

    if new_dssave_path is not None: 
        dssave_path = new_dssave_path
    
    if new_dsconfig_path is not None: 
        dsconfig_path = new_dsconfig_path

    if os.path.exists(dsconfig_path):
        with open(dsconfig_path) as dsconfig_file:
            dsconfig = json.load(dsconfig_file) 
        dsconfig_dscount = len(dsconfig['datasets'])
    else:
        dsconfig = False
    
    return dsconfig_path, dsconfig_dscount, dsconfig

def load(name=None, **kwargs):
    kparams = locals()['kwargs']
    params = {}
    dataset = None
    if name is not None:
        for i in range(len(dsconfig['datasets'])):
            if dsconfig['datasets'][i]["name"] == name:
                params = dsconfig['datasets'][i].copy()
                del params['name']
                break
        if not bool(params):
            warnings.warn("Dataset not found in dsconfig JSON")
    params.update(kparams) 
    if 'source' not in params:
        warnings.warn("Source not defined")
    else:
        if params['source'] == 'Kaggle':
            source = Kaggle()
        else:
            source = URL()
            
        params1 = source.fetch(**params)
        params2 = source.extract(**params1)
        params3 = source.parse(**params2)
        
        #TODO add code that differenciates the x, y, train, test
        dataset = source.prepare(**params3)
        
    return dataset