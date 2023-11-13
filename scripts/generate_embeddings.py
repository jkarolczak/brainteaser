import os
from glob import glob

from structs import DataSet

for full_path in glob("./data/*.npy"):
    dataset = DataSet.from_file(full_path)
    for idx, _ in enumerate(dataset):
        dataset[idx].embed()
    dataset_name = os.path.splitext(os.path.split(full_path)[-1])[0]
    DataSet.to_file(dataset=dataset, path=os.path.join("./data", f"{dataset_name}.pkl"))
