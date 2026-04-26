import logging
from torch.utils.tensorboard import SummaryWriter

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        
    def close(self):
        self.writer.close()
