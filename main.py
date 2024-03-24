import torch
import numpy as np
import cv2
import time
from ultralytics import RTDETR, YOLO
import supervision as sv
from PIL import Image


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

        # Load the class names from the YOLO model to the DETR model
        self.model.model.names = self.model1.names
        torch.save(self.model, "rtdetr-l-names.pt")

        self.CLASS_NAMES_DICT = self.model.model.names

        print("Classes: ", self.CLASS_NAMES_DICT)

        self.box_annotator = sv.BoxAnnotator(
            thickness=1, text_thickness=1, text_scale=1
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
        # After typed, wait a second for detection
        # Type 0 if using camera
        path = input("Enter the path of the image or video: \n")

        # self.detect_image(path)
        if path.endswith(".jpg") or path.endswith(".png") or path.endswith(".jpeg"):
            self.detect_image(path)
        else:
            if path == '0':
                self.detect_video(int(path))
            else:
                self.detect_video(path)

    def detect_video(self, video_path):
        # Open the video file or camera stream
        cap = cv2.VideoCapture(video_path)

        # Check if the video file or camera stream is opened successfully
        assert cap.isOpened()

        # Set the resolution of the video stream
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Loop through the video frames
        while cap.isOpened():

            # Read the video frame
            start_time = time.perf_counter()

            setx, frame = cap.read()

            # Predict the bounding boxes
            results = self.model.predict(frame)

            # Plot the bounding boxes on the frame
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
                1,
            )

            # Display the frame with the bounding boxes
            cv2.imshow("DETR", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

    def detect_image(self, image_path):
        image = Image.open(image_path)
        image.thumbnail((1280, 720))
        resize_path = "myimage.png"
        image.save(resize_path, RGBA=True)

        frame = cv2.imread(resize_path)

        assert frame is not None, "Image not found"  # Make sure the image is loaded correctly

        start_time = time.perf_counter()

        results = self.model.predict(frame)

        frame = self.plot_bboxes(results, frame)

        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        cv2.imshow("DETR", frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()


transformer_detector = DETRClass(0)
transformer_detector()
