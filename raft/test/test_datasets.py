import core.datasets as datasets

def test_viper_dataset_loader():
    dataset = datasets.VIPER(use_semseg=True)

if __name__ == "__main__":
    test_viper_dataset_loader()
