import Constant
from DataSave import DataSave
from Dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset()

    dataset.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_OFF)
    DataSave.save_gs_features_to_storage(dataset)
    DataSave.save_cs_features_to_storage(dataset)

    dataset.access_dataset(Constant.WRONG_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_ON)
    DataSave.save_gs_features_to_storage(dataset)
    DataSave.save_cs_features_to_storage(dataset)