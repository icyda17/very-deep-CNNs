from model.deep_cnns import CNNDeep, CNNRes
MODELS = {
    'm3': {
        'channels': [[256], [256]],
        'kernels': [80, 3],
        'strides': [4, 1],
        'pools': [1, 1],
        'num_classes': 10
    },
    'm5': {
        'channels': [[128], [128], [256], [512]],
        'kernels': [80, 3, 3, 3],
        'strides': [4, 1, 1, 1],
        'pools': [1, 1, 1, 1],
        'num_classes': 10
    },
    'm11': {
        'channels': [[64], [64]*2, [128]*2, [256]*3, [512]*2],
        'kernels': [80, 3, 3, 3, 3],
        'strides': [4, 1, 1, 1, 1],
        'pools': [1, 1, 1, 1, 0],
        'num_classes': 10
    },
    'm18': {
        'channels': [[64], [64]*4, [128]*4, [256]*4, [512]*4],
        'kernels': [80, 3, 3, 3, 3],
        'strides': [4, 1, 1, 1, 1],
        'pools': [1, 1, 1, 1, 0],
        'num_classes': 10
    },
    'm34_res': {
        'channels': [[64], [64]*4, [128]*4, [256]*4, [512]*4],
        'kernels': [80, 3, 3, 3, 3],
        'strides': [4, 1, 1, 1, 1],
        'pools': [1, 1, 1, 1, 0],
        'num_classes': 10
    },
}


class ReadModel:
    def __init__(self, model_name):
        """
        Default: model M5
        """
        self.model_config = MODELS.get(model_name, 'm5')
        if model_name == 'm34_res':
            self.model_cls = CNNRes(channels=self.model_config['channels'],
                                    kernels=self.model_config['kernels'],
                                    strides=self.model_config['strides'],
                                    pools=self.model_config['pools'],
                                    num_classes=self.model_config['num_classes'])
        else:
            self.model_cls = CNNDeep(channels=self.model_config['channels'],
                                     kernels=self.model_config['kernels'],
                                     strides=self.model_config['strides'],
                                     pools=self.model_config['pools'],
                                     num_classes=self.model_config['num_classes'])
