import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter


class KerasModelWrapper(tf.keras.Model):
    def __init__(self, detector):
        super().__init__(self)
        self._predictor = self._build_predictor(detector)
        self._loss_function = self._build_loss_function(detector)
        self._preprocessing_function = self._build_preprocessing_function(detector)
        self._postprocessing_function = detector.postprocess
        self._custom_trainable_variables = None

    def _build_preprocessing_function(self, detector):
        def preprocess(image):
            image = tf.cast(image, tf.float32)
            return detector.preprocess(image)

        return preprocess

    def set_trainable_variables(self, trainable_variables):
        self._custom_trainable_variables = trainable_variables

    def call(self, inputs, **kwargs):
        # Preprocess incoming images.
        images, shapes = self._preprocessing_function(inputs)

        # Pass-through to the actual predictor.
        y_pred = self._predictor(images, **kwargs)
        y_pred["preprocessed_inputs"] = images
        y_pred["preprocessed_shapes"] = shapes
        return y_pred

    def postprocess(self, y_pred):
        y_pred["preprocessed_inputs"] = tf.convert_to_tensor(y_pred["preprocessed_inputs"])
        output = self._postprocessing_function(y_pred, y_pred["preprocessed_shapes"])
        output["detection_classes"] = tf.cast(output["detection_classes"], tf.int32)
        output["detection_classes"] = output["detection_classes"] + tf.constant(1)
        return output

    def compile(self, optimizer, loss=None, **kwargs):
        assert loss is None, "Providing a custom loss function is not supported"
        return super().compile(optimizer, self._loss_function, **kwargs)

    def _build_predictor(self, detector):

        # Define input image shape using a variable resolution but 3 channels.
        image = tf.keras.layers.Input(shape=(None, None, 3), dtype=tf.float32)

        # Extract features from the input images.
        feature_maps = detector._feature_extractor(image)

        # Compute boxes from the provided features.
        outputs = {}
        predictor_results_dict = detector._box_predictor(feature_maps)
        for prediction_key, prediction_list in iter(predictor_results_dict.items()):
            prediction = tf.concat(prediction_list, axis=1)
            outputs[prediction_key] = prediction

        # Build model using functional API to wrap predictor.
        return tf.keras.Model(inputs=[image], outputs=outputs)

    def _build_loss_function(self, detector):
        def loss_func(y_true, y_pred, shapes=None):

            # Ignore all the boxes that were added for input padding purposes.
            y_true_unpadded = {}
            counts = [ct[0] for ct in y_true["detection_count"]]
            for key in ["detection_boxes", "detection_classes"]:
                y_true_unpadded[key] = [batch[:ct] for ct, batch in zip(counts, y_true[key])]

            # We have to feed the y_true data separately to the inner detector.
            detector.provide_groundtruth(
                groundtruth_boxes_list=y_true_unpadded["detection_boxes"],
                groundtruth_classes_list=y_true_unpadded["detection_classes"],
            )

            # Finally, compute the losses.
            losses_dict = detector.loss(y_pred, shapes)
            return losses_dict["Loss/localization_loss"] + losses_dict["Loss/classification_loss"]

        return loss_func

    @tf.function
    def _compiled_train_step(self, input_data, y_true):
        # Preprocess incoming images.
        images, shapes = self._preprocessing_function(input_data)

        # Step forward.
        with tf.GradientTape() as tape:
            y_pred = self(images, training=True)
            loss_value = self.loss(y_true, y_pred, shapes=shapes)

        # Compute and apply gradients to our model's weights.
        # NOTE: Losses are computed with the interim values of y_pred, not post-processed.
        trainable_variables = self._custom_trainable_variables or self.trainable_variables
        grads = tape.gradient(loss_value, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        # NOTE: Metrics are not computed to avoid the post-processing calls.
        # self.compiled_metrics.update_state(y_true, y_output)
        # metrics = {m.name: m.result() for m in self.metrics}
        return dict(loss=loss_value)

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        # NOTE: Sample weight is currently ignored by this family of models.
        input_data, y_true, _ = data_adapter.unpack_x_y_sample_weight(data)

        # Iterate over the y_true vector here since we can't do it inside
        # of the graph operations.
        y_true = {k: [v for v in y_true[k]] for k in y_true.keys()}
        return self._compiled_train_step(input_data, y_true)

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        # NOTE: Sample weight is currently ignored by this family of models.
        input_data, y_true, _ = data_adapter.unpack_x_y_sample_weight(data)

        # Preprocess incoming images.
        images, shapes = self._preprocessing_function(input_data)

        # Perform predictions using our model *without* training.
        y_pred = self(images, training=False)
        loss_value = self.loss(y_true, y_pred, shapes=shapes)

        # NOTE: Metrics are not computed to avoid the post-processing calls.
        # self.compiled_metrics.update_state(y_true, y_output)
        # metrics = {m.name: m.result() for m in self.metrics}
        return dict(loss=loss_value)
