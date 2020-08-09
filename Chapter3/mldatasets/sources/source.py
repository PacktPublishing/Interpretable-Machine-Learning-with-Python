import sys
import warnings
import re
import os
import pandas as pd
import numpy as np
import json

class Source:
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.mlds = sys.modules['.'.join(__name__.split('.')[:-2]) or '__main__']
        
    def extract(self, **kwargs):
        nkwargs = locals()['kwargs']
        nkwargs['files'] = False
        if 'path' in nkwargs and 'filenames' in nkwargs and 'filetypes' in nkwargs:  
            if type(nkwargs['filenames']) != type(nkwargs['filetypes']):
                warnings.warn("In dsconfig JSON file filenames and filetypes must be the same type")
            else:
                if isinstance(nkwargs['filenames'], str):
                    nkwargs['filenames'] = [nkwargs['filenames']]
                    nkwargs['filetypes'] = [nkwargs['filetypes']]
                if not isinstance(nkwargs['filenames'], list):
                    warnings.warn("In dsconfig JSON file filenames and filetypes must be strings or lists")
                elif len(nkwargs['filenames']) != len(nkwargs['filetypes']):
                    warnings.warn("In dsconfig JSON file filenames and filetypes must be the same list length")
                else:
                    if 'filesplits' in nkwargs:
                        if isinstance(nkwargs['filesplits'], str):
                            nkwargs['filesplits'] = [nkwargs['filesplits']]
                        if not isinstance(nkwargs['filenames'], list) or len(nkwargs['filenames']) != len(nkwargs['filesplits']):
                            del nkwargs['filesplits']
                    if 'filesplits' not in nkwargs:
                        nkwargs['filesplits'] = ['general'] * len(nkwargs['filenames'])
                    nkwargs['files'] = []
                    for i in range(len(nkwargs['filenames'])):
                        filename = nkwargs['filenames'][i]
                        filetype = nkwargs['filetypes'][i]
                        filesplit = nkwargs['filesplits'][i]
                        if re.search('\*', filename) is not None:
                            #TODO: Multiple files fetch
                            return nkwargs
                        else:
                            filepath = os.path.join(nkwargs['path'], filename)
                            if os.path.exists(filepath):
                                dirname = os.path.basename(os.path.dirname(filepath))
                                fname = os.path.basename(filepath)
                                nkwargs['files'].append({'filetype':filetype, 'filesplit':filesplit, 'filename':filename, '__dirname__':dirname, '__filename__':fname, '__filepath__':filepath})
                    del nkwargs['filenames']
                    del nkwargs['filetypes']
                    del nkwargs['filesplits']
                    
                    print('%s dataset files found in %s folder' % (len(nkwargs['files']), nkwargs['path']))
                    
        return nkwargs
    
    def parse(self, **kwargs):
        nkwargs = locals()['kwargs']
        if 'files' in nkwargs and len(nkwargs['files']):
            for i in range(len(nkwargs['files'])):
                file = nkwargs['files'][i]
                if file['filetype'] == 'csv':
                    if 'csvopts' not in nkwargs:
                        nkwargs['csvopts'] = {}
                    if 'sep' not in nkwargs['csvopts']:
                        nkwargs['csvopts']['sep'] = ','
                    if 'removecols' in nkwargs:
                        removecols = nkwargs['removecols'].copy()
                        nkwargs['csvopts']['usecols'] = lambda x: x not in removecols
                        del nkwargs['removecols']
                    nkwargs['files'][i]['content'] = self.parse_csv(file['__filepath__'], nkwargs['csvopts'])
                elif file['filetype'] == 'xls':  
                    #TODO: create xls handling function
                    pass
                elif file['filetype'] == 'img': 
                    #TODO: create img handling function
                    pass
                
        return nkwargs
    
    def parse_csv(self, fpath, csvopts):
        #TODO: add some extra exceptions ~ convert to numpy array perhaps
        print('parsing '+fpath)
        if 'usecols' in csvopts:
            return pd.read_csv(fpath, **csvopts)[csvopts['usecols']]
        else:
            return pd.read_csv(fpath, **csvopts)
    
    def prepare(self, **kwargs):
        #TODO use gather and args to split and convert files
        return kwargs['files'][0]['content']
        
    def gather(self, files):
        #TODO sort and join by group (filesplit, filetype)
        #sorted(files, key=lambda d: (d['filesplit'], d['filetype'], d['__filepath__']))
        pass