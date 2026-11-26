import cv2
import mediapipe as mp
import time
import numpy as np

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
    # print(f"Detected {len(result.detections)} object(s) at {timestamp_ms} ms")
    # print(f"Result: {result.detections}\n") 
    #WHAT THIS LOOKS LIKE:
        # [Detection(bounding_box=BoundingBox(origin_x=586, origin_y=404, width=739, height=667), categories=[Category(index=None, score=0.5782987475395203, display_name=None, category_name='person')], keypoints=None)]
    # print(f"Result: {result}\n") 
    #WHAT THIS LOOKS LIKE:
        # DetectionResult(detections=
        #                 [Detection(bounding_box=BoundingBox(origin_x=523, origin_y=407, width=816, height=676), categories=[Category(index=None, score=0.6019241213798523, display_name=None, category_name='person')], keypoints=None)]
        #                 )
    

    #convert image BEFORE loop since this upcoming loop loops thru every INDIVIDUAL detection and we want to draw boxes for every detection
    #at the same time, not individually.
    global ndarray_img #make this global to be able to pass into cv2.imshow() later on (lets us use this frame w/ boxes instead of the one w/o boxes drawn)
    #since cv2.rectangle takes in a numpy array image instead of mediapipe image, thankfully, mediapipe gives us a function that lets us
    #convert an image passed in (as mp_image) into an ndarray for this exact reason:
    ndarray_img = output_image.numpy_view().copy()

    #loops through the amount of objects it detects on current frame
    for detection in range(0, len(result.detections)):
        #to avoid "index out of range" error for the categories attribute (which is a list), if the length of it is greater than 1,
        #simply loop through it. if it isn't (look at else block), just take the first index (the only available index) and use that.
        if len(result.detections[detection].categories) > 1:
            for i in range(0, len(result.detections[detection].categories)):
                print(f"object: {result.detections[detection].categories[i].category_name}")
                print(f"confidence: {result.detections[detection].categories[i].score}")
                print(f"x: {result.detections[detection].bounding_box.origin_x}")
                print(f"y: {result.detections[detection].bounding_box.origin_y}")
                print(f"width: {result.detections[detection].bounding_box.width}")
                print(f"height: {result.detections[detection].bounding_box.height}")
        else:
            print(f"object: {result.detections[detection].categories[0].category_name}")
            print(f"confidence: {result.detections[detection].categories[0].score}")
            print(f"x: {result.detections[detection].bounding_box.origin_x}")
            print(f"y: {result.detections[detection].bounding_box.origin_y}")
            print(f"width: {result.detections[detection].bounding_box.width}")
            print(f"height: {result.detections[detection].bounding_box.height}")

        #boxes + labels fields
        bbox = result.detections[detection].bounding_box
        starting_point = bbox.origin_x, bbox.origin_y
        ending_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        #draw bounding boxes
        cv2.rectangle(ndarray_img, starting_point, ending_point, (0, 255, 0), 3) # cv2.rectangle PARAMS: cv2.rectangle(image, start_point, end_point, color, thickness)
                                                                                                                    #image HAS TO BE A NUMPY ARRAY FORM

        #label boxes (object detections)
        cv2.putText(
            ndarray_img, 
            f"{result.detections[detection].categories[0].category_name}: {str(round(result.detections[detection].categories[0].score * 100, 2))}% confident", #*100 for the percentage form
            (result.detections[detection].bounding_box.origin_x, result.detections[detection].bounding_box.origin_y),
            cv2.FONT_HERSHEY_SIMPLEX, #random font
            1,
            (255, 255, 255),
            3
        ) #PARAMS: cv2.putText(image, text, org, fontFace, fontScale, color, thickness)

    
    print() #add a space/line break for eligibility for terminal output

    




    
#model params
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="models/efficientdet_lite0_float32.tflite"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    score_threshold=0.3,
    max_results=5, #will at most return the first n objects it sees (in this case 5)
    result_callback=on_result,
)

#initiating ndarray_img global variable
ndarray_img = None

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
        if ndarray_img is None:
            final_frame = frame
            bgr_final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            # bgr_final_frame = cv2.flip(bgr_final_frame, 1)
            cv2.imshow("frame", bgr_final_frame) #regular frame
        else:
            final_frame = ndarray_img
            bgr_final_ndarray_img = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            # bgr_final_ndarray_img = cv2.flip(bgr_final_ndarray_img, 1)
            cv2.imshow("frame", bgr_final_ndarray_img) #drawn on frame w/ boxes in callback function


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

