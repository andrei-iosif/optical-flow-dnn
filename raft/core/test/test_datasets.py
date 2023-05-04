import logging

import core.datasets as datasets

def test_viper_dataset_loader():
    dataset = datasets.VIPER(use_semseg=True)

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    test_viper_dataset_loader()
