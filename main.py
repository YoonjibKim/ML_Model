import Constant
from Dataset import Dataset


if __name__ == '__main__':
    dataset = Dataset()
    dataset.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_ON)
    