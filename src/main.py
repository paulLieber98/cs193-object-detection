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

#callback function that lets us do whatever we need to do w/ out detected objects
def on_result(result, output_image: mp.Image, timestamp_ms: int): # the ": type" is exactly like in java where we do: public exampleMethod(int n) <-- the "int" is the ": int" in this case
    print(f"Detected {len(result.detections)} object(s) at {timestamp_ms} ms")

#model params
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="models/efficientdet_lite0_float32.tflite"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    score_threshold=0.5,
    max_results=5, #will at most return the first n objects it sees (in this case 5)
    result_callback=on_result,
)


# MAIN LOOP
with ObjectDetector.create_from_options(options) as detector: #initialize object detector model | "with" = same thing as try-with-resources in java so it will close the model automatically
    # check if camera is opened
    while camera.isOpened():

        #ret: boolean that indicates whether a frame was successfully captured
        #frame: actual captured image (represented as a NumPy array)
            #what the frame looks like:
                #ex: (500, 500, 3) 
                    # 500 pixels rows/height, 500 width/length/columns, 3 color channels(each 0-255)
        ret, frame = camera.read()

        #if ret not returned/retrieved, break out of loop
        if not ret:
            break

        #display window w/ frame showing live camera feed
        cv2.imshow("frame", frame)


        #mediapipe takes in an a frame with the color order of RGB and NOT BGR so we have to convert here before passing into mp_image w/ cvtColor()
        #however openCV takes in BGR and not RGB so we need to keep that in mind if we convert back
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #mediapipe doesn't take in our numpy array image so we have to create an mp_image object and pass it into the model instead of passing in the numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        #since we're processing a live video camera feed asynchronously, we need a timestamp to keep everything in check/in order
        timestamp_ms = time.monotonic_ns() // 1_000_000 #returns a clock value that can't go backwards || divide by 1 million to convert it 
                                                        #from nanoseconds to milliseconds because thats what the detect_async() takes
                                                        # + using "//" to make it an int since it only takes ints and not floats

        #runs detection model asynchronously WHILE openCV receives new frames + calls the callback function, passing in these args into the callback function
        #reason why asynchronous:
            #model takes more time to operate on one frame than openCV takes to load a frame
            #if everything was synchronous, we would have low frame rate
            #the model skips on a few frames , making it an asynchronous task.
        detector.detect_async(mp_image, timestamp_ms)


        #let user break out of the webcam by pressing q
        if cv2.waitKey(1) == ord('q'):
            break






#stop windows and camera after breaking out of loop
camera.release()
cv2.destroyAllWindows()

