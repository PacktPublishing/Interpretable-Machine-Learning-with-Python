import sys
import warnings
import re
import os
import glob
import pandas as pd
import numpy as np
import json
import cv2
from mldatasets.common import make_dummies_with_limits, make_dummies_from_dict

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
                            filepath = os.path.join(nkwargs['path'], filename)
                            files = [{'filetype':filetype, 'filesplit':filesplit, 'filename':filename,\
                                '__dirname__':os.path.basename(os.path.dirname(f)), '__filename__':os.path.basename(f),\
                                '__filepath__':f} for f in glob.glob(filepath, recursive=True)]
                            nkwargs['files'].extend(files)
                        else:
                            filepath = os.path.join(nkwargs['path'], filename)
                            if os.path.exists(filepath):
                                dirname = os.path.basename(os.path.dirname(filepath))
                                fname = os.path.basename(filepath)
                                nkwargs['files'].append({'filetype':filetype, 'filesplit':filesplit, 'filename':filename, '__dirname__':dirname, '__filename__':fname, '__filepath__':filepath})
                    del nkwargs['filenames']
                    del nkwargs['filetypes']
                    #del nkwargs['filesplits']
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
                    if 'imgopts' not in nkwargs:
                        nkwargs['imgopts'] = {}
                    nkwargs['files'][i]['content'] = self.parse_img(file['__filepath__'], nkwargs['imgopts'])
                
        return nkwargs
    
    def parse_csv(self, fpath, csvopts):
        #TODO: add some extra exceptions ~ convert to numpy array perhaps
        print('parsing '+fpath)
        if 'usecols' in csvopts and isinstance(csvopts['usecols'], (np.ndarray, list)):
            return pd.read_csv(fpath, **csvopts)[csvopts['usecols']]
        else:
            return pd.read_csv(fpath, **csvopts)
    
    def parse_img(self, fpath, imgopts):
        if 'mode' in imgopts and isinstance(imgopts['mode'], (int, np.int8, np.int16, np.int32, np.int64)):
            mode = imgopts['mode']
        else:
            mode = 1
        if 'space' in imgopts and isinstance(imgopts['space'], (int, np.int8, np.int16, np.int32, np.int64)):
            space = imgopts['space']
        else:
            space = 4
        icontent = cv2.imread(fpath, mode)
        icontent = cv2.cvtColor(icontent, space)
        if 'resize' in imgopts and isinstance(imgopts['resize'], (np.ndarray, list)):
            icontent = cv2.resize(icontent, tuple(imgopts['resize']))
        return icontent
    
    def prepare(self, **kwargs):
        nkwargs = locals()['kwargs']
        if 'prepare' in nkwargs and nkwargs['prepare'] and\
            'prepcmds' in nkwargs and len(nkwargs['prepcmds']) and\
            len(nkwargs['files']) and not isinstance(nkwargs['files'][0]['content'], np.ndarray):
                for i in range(len(nkwargs['files'])):
                    if isinstance(nkwargs['files'][i]['content'], pd.DataFrame):
                        df = nkwargs['files'][i]['content'].copy()
                        cmds = nkwargs['prepcmds']
                        cmds.insert(0, "df = dfo.copy(deep=True)")
                        cmds.insert(0, "def prep(dfo):")
                        cmds.append("return df")
                        exec("\r\n\t".join(cmds))
                        df = eval("prep(df)")
                        nkwargs['files'][i]['content'] = df.copy()
                        del df
        if len(nkwargs['files']) == 1:
            #TODO use gather and args to split and convert files
            return nkwargs['files'][0]['content']
        else:
            #splits = list(set(i['filesplit'] for i in nkwargs['files']))
            splits = nkwargs['filesplits']
            all_contents = []
            all_labels = []
            for split in splits:
                contents, labels = list(zip(*[(i['content'], i[nkwargs['target']])\
                                              for i in nkwargs['files'] if i["filesplit"] == split]))
                if len(labels) and isinstance(labels, (tuple, list)):
                    labels = np.array(list(labels)).reshape(-1, 1)
                if len(contents) and isinstance(contents[0], (np.ndarray)):
                    contents = np.array(contents)
                    if 'prepare' in nkwargs and nkwargs['prepare'] and 'prepcmds' in nkwargs and len(nkwargs['prepcmds']):
                        arr = np.copy(contents)
                        cmds = nkwargs['prepcmds']
                        #cmds.insert(0, "arr = np.copy(arro)")
                        cmds.insert(0, "def prep(arr):")
                        cmds.append("return arr")
                        exec("\n    ".join(cmds))
                        contents = eval("prep(arr)")
                        #contents = np.copy(arr)
                        del arr
                all_contents.append(contents)
                all_labels.append(labels)
            return tuple(all_contents + all_labels)
        
    def gather(self, files):
        #TODO sort and join by group (filesplit, filetype)
        #sorted(files, key=lambda d: (d['filesplit'], d['filetype'], d['__filepath__']))
        pass