from .model import regist_model, get_model
from .tsm import TSM

# regist models, sort by alphabet
regist_model("TSM", TSM)