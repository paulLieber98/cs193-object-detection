import cv2

#arg = which camera to use
camera = cv2.VideoCapture(1)

while True:
    #check if camera is opened
    if camera.isOpened():

        #ret: boolean that indicates whether a frame was successfully captured
        #frame: actual captured image (represented as a NumPy array)
        ret, frame = camera.read()

        #if ret not returned/retrieved, break out of loop
        if not ret:
            break

        #display window w/ frame showing live camera feed
        cv2.imshow("frame", frame)

        #let user break out of the webcam by pressing q
        if cv2.waitKey(1) == ord('q'):
            break


#stop windows and camera after breaking out of loop
camera.release()
cv2.destroyAllWindows()

