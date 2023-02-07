import Constant
from Consensus import Consensus
from DataSave import DataSave
from Dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset()
    dataset.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_OFF)
    DataSave.save_gs_features_to_storage(dataset)
    DataSave.save_cs_features_to_storage(dataset)

    consensus = Consensus()
    consensus.parsing_dataset(dataset.get_base_dir_path())
    consensus.load_data()


