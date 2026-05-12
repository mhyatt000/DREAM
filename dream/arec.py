from pathlib import Path
from typing import Any

from array_record.python.array_record_data_source import ArrayRecordDataSource
import albumentations as albu
import msgpack
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms

import dream
from dream.datasets import ManipulatorNDDSDatasetDebugLevels


DEFAULT_XARM_DREAM_AREC_ROOT = Path("~/.cache/arrayrecords").expanduser()
DEFAULT_XARM_DREAM_AREC_NAME = "xarm_dream_100k"
DEFAULT_XARM_DREAM_AREC_VERSION = "0.1.0"
DEFAULT_XARM_DREAM_AREC_BRANCH = "main"
XARM_DREAM_AREC_RAW_KEYPOINT_NAMES = (
    "base",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "eef",
    "tcp",
)
DEFAULT_XARM_DREAM_KEYPOINT_INDICES = tuple(range(8))
DEFAULT_XARM_DREAM_KEYPOINT_NAMES = tuple(
    XARM_DREAM_AREC_RAW_KEYPOINT_NAMES[i] for i in DEFAULT_XARM_DREAM_KEYPOINT_INDICES
)


def _ndarray_from_serializable(d: dict[str, Any]) -> np.ndarray:
    arr = np.frombuffer(d["data"], dtype=np.dtype(d["dtype"]))
    return arr.reshape(d["shape"])


def _default_unpack(obj: dict[str, Any]) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return _ndarray_from_serializable(obj)
    return obj


def unpack_record(buf: bytes) -> Any:
    return msgpack.unpackb(buf, object_hook=_default_unpack, raw=False)


class XArmDREAMArecDataset(TorchDataset):
    """Torch dataset for the cached xarm_dream_100k Arec dataset.

    The returned sample mirrors ``ManipulatorNDDSDataset`` so it can be used by
    the existing DREAM training code.
    """

    def __init__(
        self,
        network_input_resolution,
        network_output_resolution,
        image_normalization,
        image_preprocessing,
        keypoint_names=None,
        keypoint_indices=DEFAULT_XARM_DREAM_KEYPOINT_INDICES,
        arec_root=DEFAULT_XARM_DREAM_AREC_ROOT,
        arec_name=DEFAULT_XARM_DREAM_AREC_NAME,
        arec_version=DEFAULT_XARM_DREAM_AREC_VERSION,
        arec_branch=DEFAULT_XARM_DREAM_AREC_BRANCH,
        augment_data=False,
        include_ground_truth=True,
        include_belief_maps=False,
        debug_mode=ManipulatorNDDSDatasetDebugLevels["NONE"],
    ):
        if include_belief_maps:
            assert include_ground_truth, (
                'If "include_belief_maps" is True, "include_ground_truth" must also be True.'
            )

        assert isinstance(image_normalization, dict) or not image_normalization, (
            'Expected image_normalization to be either a dict specifying "mean" and '
            '"stdev", or None or False to specify no normalization.'
        )
        assert image_preprocessing in dream.image_proc.KNOWN_IMAGE_PREPROC_TYPES, (
            'Image preprocessing type "{}" is not recognized.'.format(
                image_preprocessing
            )
        )

        self.keypoint_indices = tuple(keypoint_indices)
        self.keypoint_names = list(keypoint_names or DEFAULT_XARM_DREAM_KEYPOINT_NAMES)
        assert len(self.keypoint_names) == len(self.keypoint_indices), (
            f"Expected {len(self.keypoint_indices)} keypoint names for selected "
            f"Arec keypoint indices, but got {len(self.keypoint_names)}."
        )
        self.network_input_resolution = network_input_resolution
        self.network_output_resolution = network_output_resolution
        self.image_preprocessing = image_preprocessing
        self.augment_data = augment_data
        self.include_ground_truth = include_ground_truth
        self.include_belief_maps = include_belief_maps
        self.debug_mode = debug_mode

        self.arec_name = arec_name
        self.arec_version = arec_version
        self.arec_branch = arec_branch
        self.arec_path = (
            Path(arec_root).expanduser()
            / arec_name
            / arec_version
            / arec_branch
            / "data"
        )
        self.shards = sorted(self.arec_path.glob("data-*.arrayrecord"))
        if not self.shards:
            raise FileNotFoundError(
                f"No ArrayRecord shards found under {self.arec_path}."
            )

        self._source = None
        self._length = len(self.source)

        self.tensor_from_image_no_norm_tform = TVTransforms.Compose(
            [TVTransforms.ToTensor()]
        )
        if image_normalization:
            assert (
                "mean" in image_normalization and len(image_normalization["mean"]) == 3
            ), (
                'When image_normalization is a dict, expected key "mean" specifying a '
                "3-tuple to exist, but it does not."
            )
            assert (
                "stdev" in image_normalization
                and len(image_normalization["stdev"]) == 3
            ), (
                'When image_normalization is a dict, expected key "stdev" specifying a '
                "3-tuple to exist, but it does not."
            )
            self.tensor_from_image_tform = TVTransforms.Compose(
                [
                    TVTransforms.ToTensor(),
                    TVTransforms.Normalize(
                        image_normalization["mean"],
                        image_normalization["stdev"],
                    ),
                ]
            )
        else:
            self.tensor_from_image_tform = self.tensor_from_image_no_norm_tform

    @property
    def source(self):
        if self._source is None:
            self._source = ArrayRecordDataSource([str(path) for path in self.shards])
        return self._source

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_source"] = None
        return state

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        record = unpack_record(self.source[index])

        image_rgb_raw = PILImage.fromarray(record["image"]).convert("RGB")
        image_raw_resolution = image_rgb_raw.size

        n_raw_keypoints = len(record["state"]["kp2d"])
        assert max(self.keypoint_indices, default=-1) < n_raw_keypoints, (
            f"Selected keypoint indices {self.keypoint_indices} exceed "
            f"{self.arec_name}'s {n_raw_keypoints} raw keypoints."
        )
        idx = np.asarray(self.keypoint_indices, dtype=np.int64)
        n_keypoints = len(self.keypoint_indices)

        if self.include_ground_truth:
            kp_projs_raw = np.array(
                record["state"]["kp2d"], dtype=np.float32, copy=True
            )[idx]
            keypoint_positions = np.array(
                record["state"]["kp3d_camera"],
                dtype=np.float32,
                copy=True,
            )[idx]
            keypoint_visibility = np.array(
                record.get("info", {}).get(
                    "kp_visible", np.ones(n_raw_keypoints, dtype=bool)
                ),
                dtype=bool,
                copy=True,
            )[idx]
        else:
            kp_projs_raw = np.zeros((n_keypoints, 2), dtype=np.float32)
            keypoint_positions = np.zeros((n_keypoints, 3), dtype=np.float32)
            keypoint_visibility = np.zeros(n_keypoints, dtype=bool)

        image_rgb_before_aug = dream.image_proc.preprocess_image(
            image_rgb_raw,
            self.network_input_resolution,
            self.image_preprocessing,
        )
        kp_projs_before_aug = dream.image_proc.convert_keypoints_to_netin_from_raw(
            kp_projs_raw,
            image_raw_resolution,
            self.network_input_resolution,
            self.image_preprocessing,
        )

        if self.augment_data:
            augmentation = albu.Compose(
                [
                    albu.GaussNoise(),
                    albu.RandomBrightnessContrast(brightness_by_max=False),
                    albu.ShiftScaleRotate(rotate_limit=15),
                ],
                p=1.0,
                keypoint_params={"format": "xy", "remove_invisible": False},
            )
            augmented_data = augmentation(
                image=np.array(image_rgb_before_aug),
                keypoints=kp_projs_before_aug,
            )
            image_rgb_net_input = PILImage.fromarray(augmented_data["image"])
            kp_projs_net_input = np.asarray(
                augmented_data["keypoints"], dtype=np.float32
            )
        else:
            image_rgb_net_input = image_rgb_before_aug
            kp_projs_net_input = kp_projs_before_aug

        assert image_rgb_net_input.size == self.network_input_resolution, (
            "Expected resolution for image_rgb_net_input to be equal to specified "
            "network input resolution, but they are different."
        )

        kp_projs_for_labels = np.asarray(kp_projs_net_input, dtype=np.float32).copy()
        kp_projs_for_labels[~keypoint_visibility] = -1.0
        kp_projs_net_output = dream.image_proc.convert_keypoints_to_netout_from_netin(
            kp_projs_for_labels,
            self.network_input_resolution,
            self.network_output_resolution,
        )

        image_rgb_net_input_as_tensor = self.tensor_from_image_tform(
            image_rgb_net_input
        )
        image_rgb_net_input_viz_as_tensor = self.tensor_from_image_no_norm_tform(
            image_rgb_net_input
        )
        keypoint_positions_as_tensor = torch.from_numpy(keypoint_positions).float()
        kp_projs_net_output_as_tensor = torch.from_numpy(
            np.asarray(kp_projs_net_output, dtype=np.float32)
        ).float()

        episode_id = int(
            np.asarray(record.get("info", {}).get("id", {}).get("episode", 0))
        )
        step_id = int(
            np.asarray(record.get("info", {}).get("id", {}).get("step", index))
        )
        sample = {
            "image_rgb_input": image_rgb_net_input_as_tensor,
            "keypoint_projections_output": kp_projs_net_output_as_tensor,
            "keypoint_positions": keypoint_positions_as_tensor,
            "keypoint_visibility": torch.from_numpy(keypoint_visibility),
            "config": {
                "name": f"{self.arec_name}/{self.arec_version}/{episode_id:06d}/{step_id:06d}",
                "arec_name": self.arec_name,
                "arec_version": self.arec_version,
                "arec_branch": self.arec_branch,
                "index": index,
                "episode_id": episode_id,
                "step_id": step_id,
                "keypoint_names": self.keypoint_names,
                "keypoint_indices": self.keypoint_indices,
            },
        }

        if self.include_belief_maps:
            belief_maps = dream.image_proc.create_belief_map(
                self.network_output_resolution,
                kp_projs_net_output_as_tensor,
            )
            sample["belief_maps"] = torch.tensor(belief_maps).float()

        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["LIGHT"]:
            sample["keypoint_projections_raw"] = torch.from_numpy(kp_projs_raw).float()
            sample["keypoint_projections_input"] = torch.from_numpy(
                np.asarray(kp_projs_net_input, dtype=np.float32)
            ).float()
            sample["image_resolution_raw"] = torch.tensor(image_raw_resolution).float()
            sample["image_rgb_input_viz"] = image_rgb_net_input_viz_as_tensor
            sample["camera_K"] = torch.from_numpy(
                np.array(record["camera"]["intr"]["K"], dtype=np.float32, copy=True)
            ).float()

        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["HEAVY"]:
            pass

        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["INTERACTIVE"]:
            debug_image_raw = dream.image_proc.overlay_points_on_image(
                image_rgb_raw,
                kp_projs_raw,
                self.keypoint_names,
            )
            debug_image_raw.show()

            debug_image = dream.image_proc.overlay_points_on_image(
                image_rgb_net_input,
                kp_projs_net_input,
                self.keypoint_names,
            )
            debug_image.show()

            input("Press Enter to continue...")

        return sample


ManipulatorArecDataset = XArmDREAMArecDataset


if __name__ == "__main__":
    dataset = XArmDREAMArecDataset(
        network_input_resolution=(400, 400),
        network_output_resolution=(100, 100),
        image_normalization=None,
        image_preprocessing="resize",
        include_belief_maps=True,
    )
    sample = dataset[0]

    print(f"records: {len(dataset)}")
    for key, value in sample.items():
        if key == "config":
            print(f"{key}: {value}")
        else:
            print(
                f"{key}: shape={getattr(value, 'shape', None)} dtype={getattr(value, 'dtype', None)}"
            )
