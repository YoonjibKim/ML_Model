import Constant
from DataSave import DataSave
from Dataset import Dataset

if __name__ == '__main__':
    dataset_1 = Dataset()
    dataset_1.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_OFF)
    DataSave.save_gs_features_to_storage(dataset_1)
    DataSave.save_cs_features_to_storage(dataset_1)