import Constant
from Dataset import Dataset


if __name__ == '__main__':
    dataset = Dataset()
    dataset.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_OFF)
    dataset.access_dataset(Constant.WRONG_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_ON)
    dataset.access_dataset(Constant.WRONG_EV_TS, Constant.RANDOM_CS_OFF, Constant.GAUSSIAN_OFF)
    dataset.access_dataset(Constant.WRONG_CS_TS, Constant.RANDOM_CS_OFF, Constant.GAUSSIAN_ON)