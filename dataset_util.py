import csv
import numpy as np
import pandas as pd
import tensorflow as tf

_FIELDNAMES = ["set", "path", "label", "x_min", "y_min", "_", "_", "x_max", "y_max", "_", "_"]


def read_image_tensor(image_path, image_size=None):
    data = tf.io.read_file(image_path)
    image = tf.io.decode_image(data)
    if image_size:
        image = tf.image.resize(image, image_size)
    return image


def _pad_row(row, pad_size):
    row["detection_count"] = len(row["detection_boxes"])
    if row["detection_count"] < pad_size:
        pad_row_count = pad_size - row["detection_count"]
        row["detection_boxes"] += [[np.nan] * 4] * pad_row_count
        row["detection_scores"] += [np.nan] * pad_row_count
        row["detection_classes"] += [np.nan] * pad_row_count
    return row


def _one_hot_encode(label: int, num_classes: int):
    if np.isnan(label):
        return [np.nan] * num_classes
    else:
        return [1 if i == label else 0 for i in range(num_classes)]


def image_dataset_from_paths(
    image_paths,
    detection_boxes,
    detection_classes,
    image_size=None,
    detection_count_padding=None,
    shuffle=True,
    batch_size=32,
):
    detection_boxes = [list(boxes) for boxes in detection_boxes]
    detection_classes = [list(labels) for labels in detection_classes]
    assert len(image_paths) == len(detection_boxes) and len(detection_boxes) == len(
        detection_classes
    ), (
        f"List lengths must match, found: "
        f"{len(image_paths)} {len(detection_boxes)} {len(detection_classes)}"
    )
    assert 0 not in detection_classes, "labels must be 1-indexed because 0 is the background label"

    # Put everything into a dataframe.
    df = pd.DataFrame({"file_path": image_paths})
    df["detection_boxes"] = detection_boxes
    df["detection_classes"] = detection_classes
    distinct_labels = [x for x in np.unique(sum(detection_classes, [])) if not np.isnan(x)]

    # The annotated examples are all 100% certain.
    df["detection_scores"] = df["detection_boxes"].apply(lambda boxes: [1] * len(boxes))

    # Since each image can have multiple boxes, we must pad all inputs.
    if detection_count_padding is None:
        detection_count_padding = max(len(boxes) for boxes in detection_boxes)
    df = df.apply(lambda row: _pad_row(row, detection_count_padding), axis=1)

    # Convert the labels from one-indexed to zero-indexed one-hot encoding.
    num_classes = len(distinct_labels)
    df["detection_classes"] = df["detection_classes"].apply(
        lambda labels: [_one_hot_encode(x - 1, num_classes) for x in labels]
    )

    if shuffle:
        df = df.sample(frac=1)

    X = tf.data.Dataset.from_generator(
        lambda: (read_image_tensor(path, image_size) for path in df["file_path"]),
        output_signature=(tf.TensorSpec(shape=(None, None, 3))),
    )

    y_boxes = tf.data.Dataset.from_generator(
        lambda: df["detection_boxes"],
        output_signature=(tf.TensorSpec(shape=(None, 4))),
    )
    y_classes = tf.data.Dataset.from_generator(
        lambda: df["detection_classes"],
        output_signature=(tf.TensorSpec(shape=(None, num_classes))),
    )
    y_scores = tf.data.Dataset.from_generator(
        lambda: df["detection_scores"],
        output_signature=(tf.TensorSpec(shape=(None,))),
    )
    y_box_count = tf.data.Dataset.from_generator(
        lambda: df["detection_count"],
        output_signature=(tf.TensorSpec(shape=(), dtype=tf.int32)),
    )
    y = tf.data.Dataset.zip((y_boxes, y_classes, y_scores, y_box_count))
    input_names = ["detection_boxes", "detection_classes", "detection_scores", "detection_count"]
    y = y.map(lambda *columns: {name: column for name, column in zip(input_names, columns)})

    # Bundle into a single dataset that can be fed into a standard keras training loop.
    dataset = tf.data.Dataset.zip((X, y))
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(df)))

    if batch_size:
        batch_opts = dict(deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, **batch_opts)

    # Add a few attributes that can be useful later.
    setattr(dataset, "image_size", image_size)
    setattr(dataset, "class_names", distinct_labels)
    setattr(dataset, "image_paths", df["file_path"].values.tolist())

    return dataset


def image_dataset_from_csv(file_path, **kwargs):

    with tf.io.gfile.GFile(file_path) as f:
        reader = csv.DictReader(f, fieldnames=_FIELDNAMES)
        df = pd.DataFrame.from_records(iter(reader))

    labels = df.label.unique().tolist()

    # NOTE: These operations can probably more efficiently done using
    # vectorized operations from pandas, but this is fast enough.
    grouped_by_path = {path: [] for path in df.path}
    for _, row in df.iterrows():
        label = labels.index(row.label)
        bbox = [row.x_min, row.y_min, row.x_max, row.y_max]
        record = dict(subset=row.set, file_path=row.path, detection_class=label, detection_box=bbox)
        grouped_by_path[row.path].append(record)

    records = []
    for group in grouped_by_path.values():
        subset = file_path = group[0]["subset"]
        file_path = file_path = group[0]["file_path"]
        detection_boxes = [x["detection_box"] for x in group]
        detection_classes = [x["detection_class"] for x in group]
        record = dict(
            subset=subset,
            file_path=file_path,
            detection_boxes=detection_boxes,
            detection_classes=detection_classes,
        )
        records.append(record)

    # Create a dataframe containing all the records grouped by image.
    data = pd.DataFrame.from_records(records)

    # Build a dataset for each individual subset group.
    datasets = {}
    for subset_name in data.subset.unique():
        data_subset = data.loc[data.subset == subset_name]
        datasets[subset_name] = image_dataset_from_paths(
            data_subset.file_path,
            data_subset.detection_boxes,
            data_subset.detection_classes,
            **kwargs,
        )

    return datasets
