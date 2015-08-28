import numpy as np
import cv2
import time
from tracked_object import TrackedObject
URL = '/home/jancio/Desktop/portals-kinect/data/video.avi'
MIN_HEIGHT = 70
GUI = True
MIN_CONTOUR_AREA = 100
HEIGHT_TOLERANCE = 10


def filter_frame(frame, bg_reference):
        # Function computes threshold of the frame and filters out all noise

        # Create grayscale version of the frame and blur it
        img_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(img_grayscale,(5,5))
        # Substract the bg_reference to find foreground
        fg = cv2.absdiff(blur, bg_reference)
        #Threshold the foreground
        ret, thresh = cv2.threshold(fg, MIN_HEIGHT, 255, cv2.THRESH_BINARY)
        #Execute erode and dilate functions to eliminate noise
        #erode_kernel = np.ones((5,5),np.uint8)
        #dilate_kernel = np.ones((10,10),np.uint8)
        #erosion = cv2.erode(thresh, erode_kernel,iterations = 1)
        #dilatation = cv2.erode(erosion, dilate_kernel,iterations = 1)
        return thresh


def find_contour_centroids(frame, filtered_fg):
        # Function find centroids of all valid contours in the frame
        # Know issues :
        #   1.  nearby contours are not merged - considering nearness clustering

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #Find contour in a given frame
        _, contours, _ = cv2.findContours(filtered_fg.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
        # Filter the conturs to track only the valid ones
        valid_contours = filter(is_valid_contour, contours)
        centroids = []
        # Iterate over valid contours
        for h,cnt in enumerate(valid_contours):

            # Create an empty mask
            mask = np.zeros(filtered_fg.shape,np.uint8)
            # Draw a contour on the mask
            cv2.drawContours(mask,[cnt],0,255,-1)
            #Find the minimal value of the contour (highest point)
            minVal, _, _, _ =  cv2.minMaxLoc(gray, mask=mask)
            # Threshold whole image with the range (minVal, minVal + tolerance)
            _, thresh = cv2.threshold(gray, minVal+HEIGHT_TOLERANCE,
                    255, cv2.THRESH_BINARY_INV)
            # And the threshold with mask to eliminate all results not in the
            # same contour
            result = cv2.bitwise_and(mask, thresh)
            # Find contour on a result to be able to find centroid
            _, contours, _ = cv2.findContours(result,
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_cnt = max(contours, key=cv2.contourArea)

            if (cv2.contourArea(max_cnt) > 0):
                # Find moments of the found contour
                M = cv2.moments(max_cnt)
                # Compute centroids from the moments
                centroid_x = int(M['m10']/M['m00'])
                centroid_y = int(M['m01']/M['m00'])
                # Append the tuple of coordinates to the result vector
                centroids.append((centroid_x, centroid_y))
        return centroids


def is_valid_contour(contour):
    # Function decides if the contour is good enough to be tracked
    contour_area = cv2.contourArea(contour)
    return contour_area > MIN_CONTOUR_AREA


def tracking_start():
    frame_delay = 50
    # Initialise videl capture
    cap = cv2.VideoCapture(URL)
    # Take first frame as bacground reference
    _, initial_frame = cap.read()
    img_grayscale = cv2.cvtColor(initial_frame,cv2.COLOR_BGR2GRAY)
    bg_reference = cv2.blur(img_grayscale,(5,5))
    # Iterate forever
    obj = False
    while(cap.isOpened()):
        #Read frame
        ret, frame = cap.read()
        if not ret:
            print "Read failed"
            break
        # Obtain thresholded and filtered version
        filtered_fg = filter_frame(frame, bg_reference)

        # Find centroids of all contours
        centroids = find_contour_centroids(frame, filtered_fg)

        if (len(centroids) > 0) :
            if obj:
                obj.update(centroids[0][0], centroids[0][1])
            else:
                obj = TrackedObject(centroids[0][0], centroids[0][1])
        else :
            obj = False

        if GUI:
            red = (255, 0 ,0)
            radius = 10
            for i, centroid in enumerate(centroids):


                cv2.circle(frame, centroid, radius, red)
            if obj:

                cv2.circle(frame, (obj.get_kalman()[0], obj.get_kalman()[1]),radius,(0,255,0))
            cv2.imshow('frame',frame)
            #cv2.imshow('fg', filtered_fg)

            key = cv2.waitKey(frame_delay)

            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('p'):
                time.sleep(2)
            if key & 0xFF == ord('s'):
                frame_delay = 1000
            if key & 0xFF == ord('n'):
                frame_delay = 100
            if key & 0xFF == ord('f'):
                frame_delay = 1



    cap.release()
    if GUI:
        cv2.destroyAllWindows()
