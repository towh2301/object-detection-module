import torch
import numpy as np
import cv2
import time
from ultralytics import RTDETR, YOLO
import supervision as sv


class DETRClass:

    def __init__(self, capture_index):

        self.labels = None
        self.capture_index = capture_index

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        print("Using device: ", self.device)

        self.model1 = YOLO('yolov8n.pt')
        self.model = RTDETR("rtdetr-l.pt")

        self.model.model.names = self.model1.model.names

        torch.save(self.model, "rtdetr-l-names.pt")

        self.CLASS_NAMES_DICT = self.model.model.names

        print("Classes: ", self.CLASS_NAMES_DICT)

        self.box_annotator = sv.BoxAnnotator(
            sv.ColorPalette.default(), thickness=1, text_thickness=1, text_scale=1
        )

    def plot_bboxes(self, results, frame):

        # Extract the results
        boxes = results[0].boxes.cpu().numpy()
        class_id = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy

        class_id = class_id.astype(np.int32)

        detections = sv.Detections(xyxy=xyxy, class_id=class_id, confidence=conf)

        self.labels = [
            f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        frame = self.box_annotator.annotate(frame, detections, self.labels)

        return frame

    def __call__(self):

        cap = cv2.VideoCapture('nyc.mp4')
        assert cap.isOpened()
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened():

            start_time = time.perf_counter()

            setx, frame = cap.read()

            results = self.model.predict(frame)

            frame = self.plot_bboxes(results, frame)

            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("DETR", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


transformer_detector = DETRClass(0)
transformer_detector()
