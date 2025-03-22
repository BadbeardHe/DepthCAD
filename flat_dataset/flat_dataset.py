import os
import cv2
import numpy as np
import pandas as pd

import datasets

_VERSION = datasets.Version("0.0.1")

_FEATURES = datasets.Features(
    {
        "ideal_IQ_path": datasets.Value("string"),
        "noise_IQ_path": datasets.Value("string"),
        "prompt": datasets.Value("string"),
        "conf_path": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


def bin_loader(path):
    if not os.path.exists(path):
        raise FileNotFoundError

    shape = (424, 512)
    target_size = (512, 512)
    data = np.fromfile(path, dtype=np.float32).reshape(shape)
    np_data = np.nan_to_num(data, 0)

    resized_data = cv2.resize(np_data, target_size, interpolation=cv2.INTER_LINEAR)
    
    return resized_data.astype(np.float32)


class FLATDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=_FEATURES,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,

                gen_kwargs={
                    "metadata_path": "flat_dataset/train.jsonl",
                    "ideal_IQ_dir": "data/ideal_IQ",
                    "noise_IQ_dir": "data/noise_IQ",
                    "conf_dir": "data/confidence"
                },
            ),
        ]

    def _generate_examples(self, metadata_path, ideal_IQ_dir, noise_IQ_dir, conf_dir):
        metadata = pd.read_json(metadata_path, lines=True)
 
        for _, row in metadata.iterrows():
            prompt = row["text"]
            ideal_IQ_path = os.path.join(ideal_IQ_dir, f"{row['idx']}.npy")
            noise_IQ_path = os.path.join(noise_IQ_dir, f"{row['idx']}.npy")
            conf_path = os.path.join(conf_dir, f"{row['idx'].split('_')[0]}.npy")

            yield row["idx"], {
                "prompt": prompt,
                "ideal_IQ_path": ideal_IQ_path,
                "noise_IQ_path": noise_IQ_path,
                "conf_path": conf_path
            }
