from model.models import ReadModel
from ultis.ultis_model import count_parameters

model_read = ReadModel('m5')
model_cls = model_read.model_cls
n = count_parameters(model_cls)
print("Number of parameters: %s" % n) # 559114
