import numpy as np
import tensorflow as tf


def read_image_tensor(image_path):
    data = tf.io.read_file(image_path)
    return tf.io.decode_image(data)


def image_dataset_from_paths(
    image_paths,
    bboxes,
    labels,
    num_classes=None,
    batch_size=32,
    image_size=(640, 640),
):
    bboxes = np.array(bboxes, dtype=np.float32)
    labels = np.array(labels, dtype=np.uint8)
    assert len(bboxes.shape) == 3, "bboxes must be a 3D array of [samples, num_boxes, coords]"
    assert len(labels.shape) == 2, "labels must be a 2D array of [samples, labels]"
    assert len(image_paths) == len(bboxes) and len(bboxes) == len(
        labels
    ), f"List lengths must match, found: {len(image_paths)} {len(bboxes)} {len(labels)}"
    assert 0 not in labels, "labels must be 1-indexed because 0 is the background label"

    # Use one-hot encoding for labels.
    num_classes = num_classes or len(np.unique(labels))
    # Convert from one-indexed to zero-indexed for one-hot encoding
    one_hot_labels = np.array([tf.one_hot(sample - 1, num_classes) for sample in labels])

    X = tf.data.Dataset.from_generator(
        lambda: map(read_image_tensor, image_paths),
        output_signature=(tf.TensorSpec(shape=(None, None, 3))),
    )

    def make_ground_truth_dict(detection_boxes, detection_classes):
        return dict(
            detection_boxes=detection_boxes,
            detection_classes=detection_classes,
            detection_scores=[1.0] * len(detection_boxes),
        )

    # Create a dictionary of bbox and label for each of the ground truth samples.
    y1 = tf.data.Dataset.from_tensor_slices(bboxes)
    y2 = tf.data.Dataset.from_tensor_slices(one_hot_labels)
    y = tf.data.Dataset.zip((y1, y2))
    y = y.map(make_ground_truth_dict)

    # Bundle into a single dataset that can be fed into a standard keras training loop.
    dataset = tf.data.Dataset.zip((X, y)).batch(batch_size)

    # Add a few attributes that can be useful later.
    setattr(dataset, "labels", np.unique(labels).tolist())
    setattr(dataset, "image_size", image_size)
    setattr(dataset, "image_paths", image_paths)

    return dataset
