import cv2
import mediapipe as mp
import time

#arg = which camera to use
camera = cv2.VideoCapture(1)

#OBJECT DETECTION MODEL CONFIGS

#model classes initiation
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#function that lets us do whatever we need to do w/ out detected objects
def on_result(result, output_image: mp.Image, timestamp_ms: int):
    print(f"Detected {len(result.detections)} object(s) at {timestamp_ms} ms")

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="models/efficientdet_lite0_float32.tflite"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    score_threshold=0.5,
    max_results=5,
    result_callback=on_result,
)


# MAIN LOOP
with ObjectDetector.create_from_options(options) as detector:
    # check if camera is opened
    while camera.isOpened():

        #ret: boolean that indicates whether a frame was successfully captured
        #frame: actual captured image (represented as a NumPy array)
        ret, frame = camera.read()

        #if ret not returned/retrieved, break out of loop
        if not ret:
            break

        #display window w/ frame showing live camera feed
        cv2.imshow("frame", frame)


        #model takes in an rgb frame not bgr so we have to convert here before passing into mp_image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = time.monotonic_ns() // 1_000_000 #returns a clock value that can't go backwards || in nanoseconds because thats what the detect_async() takes
        detector.detect_async(mp_image, timestamp_ms)


        #let user break out of the webcam by pressing q
        if cv2.waitKey(1) == ord('q'):
            break






#stop windows and camera after breaking out of loop
camera.release()
cv2.destroyAllWindows()

