import os
import json
import warnings
import requests
from zipfile import ZipFile
from .source import Source

class URL(Source):
    
    def fetch(self, **kwargs):
        nkwargs = locals()['kwargs']
        nkwargs['path'] = False
        if 'location' in nkwargs:
            dssave_path = self.mlds.config.dssave_path
            if not os.path.exists(dssave_path):
                os.makedirs(dssave_path)
            location = nkwargs['location']
            r = requests.get(location, allow_redirects=True)
            if 'saveas' in nkwargs:
                filename = nkwargs['saveas']
            else:
                filename = location.rsplit('/', 1)[1]
                if len(filename) == 0:
                    filename = r.headers.get('content-disposition')
                    if filename is not None:
                        filename = re.findall('filename=(.+)', filename)
            zipfile_path = os.path.join(dssave_path, filename)
            open(zipfile_path, 'wb').write(r.content)
            print(location + ' downloaded to ' + zipfile_path)
            if r.headers.get('Content-Type') == 'application/zip' or filename.rsplit('.', 1)[1] == 'zip':
                dsname = filename.rsplit('.', 1)[0]
                unzipdir_path = os.path.join(dssave_path, dsname)
                if os.path.exists(zipfile_path):
                    with ZipFile(zipfile_path, 'r') as zipObj:
                        zipObj.extractall(unzipdir_path)
                        print(zipfile_path + ' uncompressed to ' + unzipdir_path)
                        nkwargs['path'] = unzipdir_path
                else:
                    warnings.warn("Zip file not found at " + zipfile_path)
            else:
                nkwargs['path'] = os.path.dirname(zipfile_path)
                nkwargs['filenames'] = os.path.basename(zipfile_path)
        return nkwargs