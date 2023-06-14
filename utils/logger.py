import sys
import logging
import os


def setup_logger(name: str, level = logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def update_summary(epoch, train_acc, train_loss, val_acc, val_loss, filename, dir, header=True):
    if filename:
        filename = os.path.join(dir, filename)
        with open(filename, 'a') as f:
            if header and f.tell() == 0:
                f.write('Epoch,train_acc,train_loss,val_acc,val_loss\n')
            f.write(f'{epoch},{train_acc},{train_loss},{val_acc},{val_loss}\n')
