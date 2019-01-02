import json
import os


class Config(object):
    def __init__(self, path = "modelConf.json"):
        path = os.path.join(os.environ['ASRL_CONFIG_PATH'], "asModel/config/",path)
        with open(path, 'rt') as f:
            confContent = f.read()
            self.modelConf = json.loads(confContent)
            self.__dict__ = self.modelConf
