from collections import defaultdict

from schema.schema import (
    ClassificationOutput,
    DetectionClass,
    DetectionObject,
    DetectionOutput,
)
from service.processing.base import BaseProcessor
from service.processing.utils import (
    convert_image_to_bytes,
    filter_and_plot,
    get_pred_labels_from_logits,
    map_predictions,
)
from settings.processing import (
    CustomClassificationPostProcessorSettings,
    YoloClsPostProcessorSettings,
    YoloDet2ClsPostProcessorSettings,
    YoloPostProcessorSettings,
)


@BaseProcessor.register("CustomClassificationPostProcessor")
class CustomClassificationPostProcessor(BaseProcessor):
    """
    Post-processor for custom classification model predictions, mapping logits to labels and probabilities.
    """

    def __init__(self, settings: CustomClassificationPostProcessorSettings):
        """
        Initializes the post-processor with configuration settings.

        Args:
            settings (dict): Settings containing the labels map and optional output configurations.
        """
        self.labels_map = settings.labels_map
        self.include_all_probabilities = settings.include_all_probabilities

    def __call__(self, logits) -> ClassificationOutput:
        """
        Processes the model's logits to extract the most likely label and associated probabilities.

        Args:
            logits (torch.Tensor): The output logits from the model.

        Returns:
            ClassificationOutput: Processed prediction output with label, probability, and optionally full probabilities.
        """
        try:
            # Convert logits to probabilities and extract predicted labels
            probabilities, pred_labels = get_pred_labels_from_logits(logits)

            # Map predictions to labels and probabilities
            mapped_predictions = map_predictions(
                probabilities, pred_labels, self.labels_map
            )

            # Extract the top label and its probability
            label = mapped_predictions["label"]
            probabilities_dict = mapped_predictions["probabilities"]
            probability = probabilities_dict[label]

            # Prepare top-5 probabilities if required
            if self.include_all_probabilities:
                sorted_probs = dict(
                    sorted(
                        probabilities_dict.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                top5_results = dict(list(sorted_probs.items())[:5])
            else:
                top5_results = {}

            # Return as a ClassificationOutput model
            return ClassificationOutput(
                label=label,
                probability=probability,
                top5=top5_results,
            )

        except Exception as e:
            raise ValueError(f"Error in post-processing logits: {e}")


@BaseProcessor.register("YoloClsPostProcessor")
class YoloClsPostProcessor(BaseProcessor):
    """
    Post-processor for YOLO model predictions, extracting top labels and probabilities.
    """

    def __init__(self, settings: YoloClsPostProcessorSettings):
        """
        Initializes the post-processor with configuration settings.

        Args:
            settings (YoloClsPostProcessorSettings): Settings to include top5 or not.
        """
        self.top5: bool = settings.top5

    def __call__(self, result) -> ClassificationOutput:
        """
        Processes the YOLO output to extract labels and probabilities.

        Args:
            result: The output from the YOLO model inference.

        Returns:
            ClassificationOutput: Processed prediction output with label, probability, and optionally top-5 predictions.
        """
        result = result[0]
        labels = [result.names[i] for i in result.probs.top5]
        probs = [round(i.item(), 2) for i in result.probs.top5conf]

        top5_results = (
            {label: prob for label, prob in zip(labels, probs)} if self.top5 else {}
        )

        # Return as a ClassificationOutput model
        return ClassificationOutput(
            label=labels[0],
            probability=probs[0],
            top5=top5_results if self.top5 else {},
            # image=None,
        )


@BaseProcessor.register("YoloPostProcessor")
class YoloPostProcessor(BaseProcessor):
    """
    Post-processor for YOLO detection model predictions, organizing outputs by detected classes.
    """

    def __init__(self, settings: YoloPostProcessorSettings):
        """
        Initializes the post-processor with configuration settings.

        Args:
            settings (YoloPostProcessorSettings): Settings for bounding box type.
        """
        self.boxtype: str = settings.boxtype
        self.return_out: bool = settings.return_out

    def __call__(self, result) -> DetectionOutput:
        """
        Processes YOLO detection output to extract detected classes and their instances.

        Args:
            result: The output from the YOLO detection model inference, containing bounding boxes.

        Returns:
            DetectionOutput: Processed prediction output with unique detected classes, instances, and class list.
        """
        result = result[0]
        # Access class names from the result
        class_names = result.names

        # Initialize the output dictionary and list of detected class names
        output: dict[str, DetectionClass] = {}
        detected_class_names = set()  # Using a set to avoid duplicates

        # Retrieve the bounding box format based on boxtype setting
        boxes = getattr(result.boxes, self.boxtype, None)
        if boxes is None:
            raise ValueError(f"Invalid boxtype '{self.boxtype}' specified.")

        # Iterate through each detected box
        for cls_idx, conf, box in zip(result.boxes.cls, result.boxes.conf, boxes):
            # Convert tensor values to Python scalars for readability in the output
            cls_idx = int(cls_idx.item())
            class_name = class_names[cls_idx]
            detected_class_names.add(class_name)  # Add to detected classes list
            conf = round(conf.item(), 2)
            box = [round(coord.item(), 2) for coord in box]

            # Prepare the detected object entry
            object_data = DetectionObject(conf=conf, box=box)

            # Update the output dictionary
            if class_name not in output:
                output[class_name] = DetectionClass(count=0, objects=[])

            output[class_name].count += 1
            output[class_name].objects.append(object_data)

        if self.return_out:
            image = convert_image_to_bytes(
                image=result.plot()[..., ::-1], format="JPEG", decode=False
            )
            return DetectionOutput(
                detected_classes=list(detected_class_names), root=output, image=image
            )
        else:
            return DetectionOutput(
                detected_classes=list(detected_class_names), root=output
            )


@BaseProcessor.register("YoloDet2ClsPostProcessor")
class YoloDet2ClsPostProcessor(BaseProcessor):
    """
    Post-processor for YOLO detection model predictions, organizing outputs by detected classes
    and showing only boxes of classes that were not chosen as the predicted class.
    """

    def __init__(self, settings: YoloDet2ClsPostProcessorSettings):
        """
        Initializes the post-processor with configuration settings.

        Args:
            settings (YoloPostProcessorSettings): Settings for bounding box type.
        """
        self.boxtype: str = settings.boxtype
        self.det2cls_percentage_thr: dict = settings.det2cls_percentage_thr
        self.visualization_params: dict = settings.visualization_params

    def __call__(self, result) -> ClassificationOutput:
        """
        Processes YOLO detection output and determines the classification label.

        Args:
            result: The output from the YOLO detection model inference.

        Returns:
            ClassificationOutput: Post-processed output with filtered boxes.
        """
        result = result[0]
        class_names = result.names  # Class names mapping

        # Process boxes
        boxes = getattr(result.boxes, self.boxtype, None)
        if boxes is None:
            raise ValueError(f"Invalid boxtype '{self.boxtype}' specified.")

        # Step 1: Organize detections
        output = defaultdict(lambda: DetectionClass(count=0, objects=[], mean_conf=0.0))
        detected_class_names = set()

        for cls_idx, conf, box in zip(result.boxes.cls, result.boxes.conf, boxes):
            cls_idx = int(cls_idx.item())
            class_name = class_names[cls_idx]
            detected_class_names.add(class_name)

            object_data = DetectionObject(
                conf=round(conf.item(), 2),
                box=[round(coord.item(), 2) for coord in box],
            )
            output[class_name].count += 1
            output[class_name].objects.append(object_data)
            output[class_name].mean_conf += conf.item()

        # Calculate mean confidence
        for det_class in output.values():
            det_class.mean_conf = round(det_class.mean_conf / det_class.count, 2)

        # Step 2: Determine the label
        total_detections = sum(det.count for det in output.values())
        class_percentages = {
            class_name: round((det.count / total_detections), 4)
            for class_name, det in output.items()
        }
        top5_classes = dict(
            sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)[:5]
        )

        valid_classes = [
            (class_name, percentage)
            for class_name, percentage in class_percentages.items()
            if percentage >= self.det2cls_percentage_thr.get(class_name, 0.51)
        ]

        if len(valid_classes) == 1:
            label, prob = valid_classes[0]
            probability = output[label].mean_conf
        elif len(valid_classes) > 1:
            label, probability = "unknown", [
                output[cls[0]].mean_conf for cls in valid_classes
            ]
        else:
            label, probability = "unknown", 0.0

        # Step 3: Filter and plot visualization
        if len(detected_class_names) > 1:
            output_image = filter_and_plot(
                orig_image=result.orig_img[..., ::-1],
                boxes_cls=result.boxes.cls,
                boxes_xyxy=boxes,
                excluded_class=label,
                class_names=class_names,
                **self.visualization_params,
            )
            output_image = convert_image_to_bytes(output_image)
        else:
            output_image = None

        return ClassificationOutput(
            label=label,
            probability=probability,
            top5=top5_classes,
            image=output_image,
        )
