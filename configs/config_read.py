import os
from pathlib import Path
import yaml
import pprint

cwd = Path(__file__).resolve().parent

class ReadConfig():
    def __init__(self, path="data_urban_8k.yaml"):
        with open(os.path.join(cwd, path)) as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
            pp = pprint.PrettyPrinter(indent=2)
            pp.pprint(config) 

        self.model_name = config['model']
        self.epochs = config['epochs']
        self.shuffle = config['shuffle']
        self.datapath = config['datapath']