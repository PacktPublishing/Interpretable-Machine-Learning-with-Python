import requests
from zipfile import ZipFile
from .source import Source

class URL(Source):
    
    def fetch(self, **kwargs):
        pass