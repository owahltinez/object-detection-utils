import os
import tempfile
import urllib.request

import tensorflow as tf

from object_detection.builders import model_builder
from object_detection.utils import config_util


_BASE_GH_URL = "https://raw.githubusercontent.com/tensorflow/models/master"
_BASE_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2"
_BASE_CONFIG_URL = f"{_BASE_GH_URL}/research/object_detection/configs/tf2"


def _build_ssd_model(configs, num_classes=None):
    model_config = configs["model"]
    if num_classes is not None:
        model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    return model_builder.build(model_config=model_config, is_training=True)


def _load_ssd_checkpoint_weights(detection_model, checkpoint_path, input_shape):

    placeholder_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
    placeholder_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=placeholder_box_predictor,
    )
    ckpt = tf.compat.v2.train.Checkpoint(model=placeholder_model)
    ckpt.restore(checkpoint_path).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros(input_shape))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)


_MODEL_METADATA = {
    "ssd_resnet50_v1_fpn_640x640": {
        "default_image_size": (640, 640),
        "config_path": "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config",
        "assets_path": "20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "ssd_resnet50_v1_fpn_1024x1024": {
        "default_image_size": (1024, 1024),
        "config_path": "ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.config",
        "assets_path": "20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "ssd_resnet101_v1_fpn_640x640": {
        "default_image_size": (640, 640),
        "config_path": "ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.config",
        "assets_path": "20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "ssd_resnet101_v1_fpn_1024x1024": {
        "default_image_size": (1024, 1024),
        "config_path": "ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.config",
        "assets_path": "20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "ssd_resnet152_v1_fpn_640x640": {
        "default_image_size": (640, 640),
        "config_path": "ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.config",
        "assets_path": "20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "ssd_resnet152_v1_fpn_1024x1024": {
        "default_image_size": (1024, 1024),
        "config_path": "ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.config",
        "assets_path": "20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "efficientdet_d0": {
        "default_image_size": (512, 512),
        "config_path": "ssd_efficientdet_d0_512x512_coco17_tpu-8.config",
        "assets_path": "20200711/efficientdet_d0_coco17_tpu-32.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "efficientdet_d1": {
        "default_image_size": (640, 640),
        "config_path": "ssd_efficientdet_d1_640x640_coco17_tpu-8.config",
        "assets_path": "20200711/efficientdet_d1_coco17_tpu-32.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "efficientdet_d2": {
        "default_image_size": (768, 768),
        "config_path": "ssd_efficientdet_d2_768x768_coco17_tpu-8.config",
        "assets_path": "20200711/efficientdet_d2_coco17_tpu-32.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "efficientdet_d3": {
        "default_image_size": (896, 896),
        "config_path": "ssd_efficientdet_d3_896x896_coco17_tpu-32.config",
        "assets_path": "20200711/efficientdet_d3_coco17_tpu-32.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "efficientdet_d4": {
        "default_image_size": (1024, 1024),
        "config_path": "ssd_efficientdet_d4_1024x1024_coco17_tpu-32.config",
        "assets_path": "20200711/efficientdet_d4_coco17_tpu-32.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "efficientdet_d5": {
        "default_image_size": (1280, 1280),
        "config_path": "ssd_efficientdet_d5_1280x1280_coco17_tpu-32.config",
        "assets_path": "20200711/efficientdet_d5_coco17_tpu-32.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "efficientdet_d6": {
        "default_image_size": (1408, 1408),
        "config_path": "ssd_efficientdet_d6_1408x1408_coco17_tpu-32.config",
        "assets_path": "20200711/efficientdet_d6_coco17_tpu-32.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
    "efficientdet_d7": {
        "default_image_size": (1536, 1526),
        "config_path": "ssd_efficientdet_d7_1536x1536_coco17_tpu-32.config",
        "assets_path": "20200711/efficientdet_d7_coco17_tpu-32.tar.gz",
        "build_model_function": _build_ssd_model,
        "load_checkpoint_weights_function": _load_ssd_checkpoint_weights,
    },
}


def _model_url(metadata):
    return f'{_BASE_MODEL_URL}/{metadata["assets_path"]}'


def _model_download(metadata, path=None):
    url = _model_url(metadata)
    path = path or url.split("/")[-1]
    urllib.request.urlretrieve(url, path)
    return path


def _config_url(metadata):
    return f'{_BASE_CONFIG_URL}/{metadata["config_path"]}'


def get_configs_from_url(url):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = f"{tmpdir}/model.config"
        urllib.request.urlretrieve(url, file_path)
        return config_util.get_configs_from_pipeline_file(file_path)


def _load_model_configs(metadata):
    url = _config_url(metadata)
    return get_configs_from_url(url)


def load_detection_model(model_name, image_size=None, **kwargs):
    metadata = _MODEL_METADATA[model_name]
    configs = _load_model_configs(metadata)
    image_size = list(image_size or metadata["default_image_size"])
    detection_model = metadata["build_model_function"](configs, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = f"{tmpdir}/model.tar.gz"
        _model_download(metadata, file_path)
        # NOTE: Only works on bash-like shells with tar installed.
        os.system(f"tar -xf {file_path} --strip-components=1 -C {tmpdir}")
        checkpoint_path = f"{tmpdir}/checkpoint/ckpt-0"
        load_checkpoint_func = metadata["load_checkpoint_weights_function"]
        load_checkpoint_func(detection_model, checkpoint_path, [1, *image_size, 3])

    return detection_model
