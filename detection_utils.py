# @title Load the trained TFLite model and define some visualization functions

# @markdown This code comes from the TFLite Object Detection [Raspberry Pi sample](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi).

import platform
from typing import List, NamedTuple
import json
from tflite_support import metadata
import cv2
import tensorflow as tf
import numpy as np
from constants import CLASSES

Interpreter = tf.lite.Interpreter
load_delegate = tf.lite.experimental.load_delegate
MODEL_INPUT_SIZE = 224
INPUT_DIM = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

# pylint: enable=g-import-not-at-top
def crop_and_resize(img, x, y, w, h):
    cropped_image = img[y:h, x:w].copy()
    # resize the image to fit the model input shape
    resized_cropped_image = cv2.resize(
        cropped_image, INPUT_DIM, interpolation=cv2.INTER_AREA
    )
    resized_cropped_image = np.expand_dims(resized_cropped_image, axis=0)
    resized_cropped_image = resized_cropped_image.astype(np.float32)
    resized_cropped_image = resized_cropped_image / 255
    return resized_cropped_image


def tflite_predict(input_model, data):
    input_details = input_model.get_input_details()
    # print(input_details)
    output_details = input_model.get_output_details()
    input_model.set_tensor(0, data)
    input_model.invoke()
    output_data = input_model.get_tensor(output_details[0]["index"])
    predicted_value = output_data[0][np.argmax(output_data[0])]
    leaf_type = CLASSES[np.argmax(output_data[0])]
    return leaf_type, predicted_value


# load the model
tflite_model = tf.lite.Interpreter(
    model_path="RemoteCropDisease/resources/plant_diseas_model.tflite", num_threads=1
)

# tflite_model.resize_tensor_input(0, [-1, 224, 224, 3])
tflite_model.allocate_tensors()


class ObjectDetectorOptions(NamedTuple):
    """A config to initialize an object detector."""

    enable_edgetpu: bool = False
    """Enable the model to run on EdgeTPU."""

    label_allow_list: List[str] = None
    """The optional allow list of labels."""

    label_deny_list: List[str] = None
    """The optional deny list of labels."""

    max_results: int = -1
    """The maximum number of top-scored detection results to return."""

    num_threads: int = 1
    """The number of CPU threads to be used."""

    score_threshold: float = 0.0
    """The score threshold of detection results to return."""


class Rect(NamedTuple):
    """A rectangle in 2D space."""

    left: float
    top: float
    right: float
    bottom: float


class Category(NamedTuple):
    """A result of a classification task."""

    label: str
    score: float
    index: int


class Detection(NamedTuple):
    """A detected object as the result of an ObjectDetector."""

    bounding_box: Rect
    categories: List[Category]


def edgetpu_lib_name():
    """Returns the library name of EdgeTPU in the current platform."""
    return {
        "Darwin": "libedgetpu.1.dylib",
        "Linux": "libedgetpu.so.1",
        "Windows": "edgetpu.dll",
    }.get(platform.system(), None)


class ObjectDetector:
    """A wrapper class for a TFLite object detection model."""

    _OUTPUT_LOCATION_NAME = "location"
    _OUTPUT_CATEGORY_NAME = "category"
    _OUTPUT_SCORE_NAME = "score"
    _OUTPUT_NUMBER_NAME = "number of detections"

    def __init__(
        self, model_path: str, options: ObjectDetectorOptions = ObjectDetectorOptions()
    ) -> None:
        """Initialize a TFLite object detection model.
        Args:
            model_path: Path to the TFLite model.
            options: The config to initialize an object detector. (Optional)
        Raises:
            ValueError: If the TFLite model is invalid.
            OSError: If the current OS isn't supported by EdgeTPU.
        """

        # Load metadata from model.
        displayer = metadata.MetadataDisplayer.with_model_file(model_path)

        # Save model metadata for preprocessing later.
        model_metadata = json.loads(displayer.get_metadata_json())
        process_units = model_metadata["subgraph_metadata"][0]["input_tensor_metadata"][
            0
        ]["process_units"]
        mean = 0.0
        std = 1.0
        for option in process_units:
            if option["options_type"] == "NormalizationOptions":
                mean = option["options"]["mean"][0]
                std = option["options"]["std"][0]
        self._mean = mean
        self._std = std

        # Load label list from metadata.
        file_name = displayer.get_packed_associated_file_list()[0]
        label_map_file = displayer.get_associated_file_buffer(file_name).decode()
        label_list = list(filter(lambda x: len(x) > 0, label_map_file.splitlines()))
        self._label_list = label_list

        # Initialize TFLite model.
        if options.enable_edgetpu:
            if edgetpu_lib_name() is None:
                raise OSError("The current OS isn't supported by Coral EdgeTPU.")
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate(edgetpu_lib_name())],
                num_threads=options.num_threads,
            )
        else:
            interpreter = Interpreter(
                model_path=model_path, num_threads=options.num_threads
            )

        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]

        # From TensorFlow 2.6, the order of the outputs become undefined.
        # Therefore we need to sort the tensor indices of TFLite outputs and to know
        # exactly the meaning of each output tensor. For example, if
        # output indices are [601, 599, 598, 600], tensor names and indices aligned
        # are:
        #   - location: 598
        #   - category: 599
        #   - score: 600
        #   - detection_count: 601
        # because of the op's ports of TFLITE_DETECTION_POST_PROCESS
        # (https://github.com/tensorflow/tensorflow/blob/a4fe268ea084e7d323133ed7b986e0ae259a2bc7/tensorflow/lite/kernels/detection_postprocess.cc#L47-L50).
        sorted_output_indices = sorted(
            [output["index"] for output in interpreter.get_output_details()]
        )
        self._output_indices = {
            self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
            self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
            self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
            self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
        }

        self._input_size = input_detail["shape"][2], input_detail["shape"][1]
        self._is_quantized_input = input_detail["dtype"] == np.uint8
        self._interpreter = interpreter
        self._options = options

    def detect(self, input_image: np.ndarray) -> List[Detection]:
        """Run detection on an input image.
        Args:
            input_image: A [height, width, 3] RGB image. Note that height and width
              can be anything since the image will be immediately resized according
              to the needs of the model within this function.
        Returns:
            A Person instance.
        """
        image_height, image_width, _ = input_image.shape

        input_tensor = self._preprocess(input_image)

        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()

        # Get all output details
        boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
        classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
        scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
        count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

        return self._postprocess(
            boxes, classes, scores, count, image_width, image_height
        )

    def _preprocess(self, input_image: np.ndarray) -> np.ndarray:
        """Preprocess the input image as required by the TFLite model."""

        # Resize the input
        input_tensor = cv2.resize(input_image, self._input_size)

        # Normalize the input if it's a float model (aka. not quantized)
        if not self._is_quantized_input:
            input_tensor = (np.float32(input_tensor) - self._mean) / self._std

        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor

    def _set_input_tensor(self, image):
        """Sets the input tensor."""
        tensor_index = self._interpreter.get_input_details()[0]["index"]
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, name):
        """Returns the output tensor at the given index."""
        output_index = self._output_indices[name]
        tensor = np.squeeze(self._interpreter.get_tensor(output_index))
        return tensor

    def _postprocess(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
        count: int,
        image_width: int,
        image_height: int,
    ) -> List[Detection]:
        """Post-process the output of TFLite model into a list of Detection objects.
        Args:
            boxes: Bounding boxes of detected objects from the TFLite model.
            classes: Class index of the detected objects from the TFLite model.
            scores: Confidence scores of the detected objects from the TFLite model.
            count: Number of detected objects from the TFLite model.
            image_width: Width of the input image.
            image_height: Height of the input image.
        Returns:
            A list of Detection objects detected by the TFLite model.
        """
        results = []

        # Parse the model output into a list of Detection entities.
        for i in range(count):
            if scores[i] >= self._options.score_threshold:
                y_min, x_min, y_max, x_max = boxes[i]
                bounding_box = Rect(
                    top=int(y_min * image_height),
                    left=int(x_min * image_width),
                    bottom=int(y_max * image_height),
                    right=int(x_max * image_width),
                )
                class_id = int(classes[i])
                category = Category(
                    score=scores[i],
                    label=self._label_list[class_id],  # 0 is reserved for background
                    index=class_id,
                )
                result = Detection(bounding_box=bounding_box, categories=[category])
                results.append(result)

        # Sort detection results by score ascending
        sorted_results = sorted(
            results, key=lambda detection: detection.categories[0].score, reverse=True
        )

        # Filter out detections in deny list
        filtered_results = sorted_results
        if self._options.label_deny_list is not None:
            filtered_results = list(
                filter(
                    lambda detection: detection.categories[0].label
                    not in self._options.label_deny_list,
                    filtered_results,
                )
            )

        # Keep only detections in allow list
        if self._options.label_allow_list is not None:
            filtered_results = list(
                filter(
                    lambda detection: detection.categories[0].label
                    in self._options.label_allow_list,
                    filtered_results,
                )
            )

        # Only return maximum of max_results detection.
        if self._options.max_results > 0:
            result_count = min(len(filtered_results), self._options.max_results)
            filtered_results = filtered_results[:result_count]

        return filtered_results


_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red


def visualize(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detections: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    predictions = []
    for detection in detections:
        prediction_dict = {}
        # Draw bounding_box
        start_point = detection.bounding_box.left, detection.bounding_box.top
        end_point = detection.bounding_box.right, detection.bounding_box.bottom
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        class_name = category.label
        probability = round(category.score, 2)
        prediction_dict["leaf"] = class_name
        prediction_dict["probability"] = probability
        predictions.append(prediction_dict)
        result_text = class_name + " (" + str(probability) + ")"
        text_location = (
            _MARGIN + detection.bounding_box.left,
            _MARGIN + _ROW_SIZE + detection.bounding_box.top,
        )
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            _FONT_SIZE,
            _TEXT_COLOR,
            _FONT_THICKNESS,
        )

    return image, predictions


def visualize_classnames_with_mobilenet(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detections: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    predictions = []
    for detection in detections:
        prediction_dict = {}
        x = detection.x
        y = detection.y
        w = detection.w
        h = detection.h
        resized_image = crop_and_resize(image, x, y, w, h)
        label, score = tflite_predict(tflite_model, resized_image)
        # Draw bounding_box
        start_point = detection.bounding_box.left, detection.bounding_box.top
        end_point = detection.bounding_box.right, detection.bounding_box.bottom
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

        print(f"x{x} y{y} w{w} h{h}")

        print(start_point)
        # Draw label and score
        category = detection.categories[0]
        # class_name = category.label
        # probability = round(category.score, 2)
        class_name = label
        probability = score
        prediction_dict["leaf"] = class_name
        prediction_dict["probability"] = probability
        predictions.append(prediction_dict)
        result_text = probability + " (" + str(probability) + ")"
        text_location = (
            _MARGIN + detection.bounding_box.left,
            _MARGIN + _ROW_SIZE + detection.bounding_box.top,
        )
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            _FONT_SIZE,
            _TEXT_COLOR,
            _FONT_THICKNESS,
        )

    return image
