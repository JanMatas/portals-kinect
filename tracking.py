import numpy as np
import cv2
import time
import math
from tracked_object import TrackedObject
URL = '/home/jancio/Desktop/video.avi'
MIN_HEIGHT = 50
GUI = True
MIN_CONTOUR_AREA = 4000
HEIGHT_TOLERANCE = 10
MAX_DISTANCE_TO_PARSE =  300
MAX_DISTANCE_TO_MERGE = 10
counter_in = 0
counter_outt = 0

def filter_frame(frame, bg_reference):
        # Function computes threshold of the frame and filters out all noise

        # Create grayscale version of the frame and blur it
        img_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Substract the bg_reference to find foreground
        fg = cv2.absdiff(img_grayscale, bg_reference)
        #Threshold the foreground
        ret, thresh = cv2.threshold(fg, MIN_HEIGHT, 255, cv2.THRESH_BINARY)
        return thresh

def merge_contours(contours):
    contours = filter(lambda cnt : cv2.contourArea(cnt) > 0, contours)
    merged_contours = []
    cnts_with_centroids = []

    for cnt in contours:
        M = cv2.moments(cnt)
        # Compute centroids from the moments
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])

        cnts_with_centroids.append(((centroid_x, centroid_y), cnt))

    while cnts_with_centroids:
        new_blob = []
        root = cnts_with_centroids[0]
        new_blob.append(root)
        cnts_with_centroids.remove(root)


        cnts_with_centroids = merge_contour_rec(root,
                cnts_with_centroids, new_blob)
        unified_contours = [cnt[1] for cnt in new_blob]
        new_cnt = np.vstack(i for i in unified_contours)
        hull = cv2.convexHull(new_cnt)
        merged_contours.append(hull)
    return merged_contours


def merge_contour_rec(root_cnt, other_cnts, new_blob):
    near_cnts = filter(lambda cnt: is_near(root_cnt, cnt), other_cnts)
    other_cnts = filter(lambda cnt: not is_near(root_cnt, cnt), other_cnts)
    new_blob += near_cnts

    for cnt in near_cnts:
        other_cnts = merge_contour_rec(cnt, other_cnts, new_blob)
    return other_cnts

def is_near(cnt1, cnt2):
    return compute_distance(cnt1[0], cnt2[0]) < MAX_DISTANCE_TO_MERGE



def find_contour_centroids(frame, filtered_fg):
        # Function find centroids of all valid contours in the frame
        # Know issues :
        #   1.  nearby contours are not merged - considering nearness clustering

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #Find contour in a given frame
        _, contours, _ = cv2.findContours(filtered_fg.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
        merge_contours(contours)
        cv2.drawContours(frame,merge_contours(contours),-1,255,-1)
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
            # Find biggest contour on a result
            _, contours, _ = cv2.findContours(result,
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_cnt = max(contours, key=cv2.contourArea)
            # Check if the biggest contour has no-zero area
            if (cv2.contourArea(max_cnt) > 0):
                # Find moments of the found contour
                M = cv2.moments(max_cnt)
                # Compute centroids from the moments
                centroid_x = int(M['m10']/M['m00'])
                centroid_y = int(M['m01']/M['m00'])
                # Append the tuple of coordinates to the result vector
                centroids.append((centroid_x, centroid_y))
        return centroids


def compute_distance(a, b):
    # Function returns the distance between two (x, y) points
    return math.hypot(b[0] - a[0], b[1] - a[1])


def assign_centroids(tracked_objects, centroids, t):
    # Function tries to assign every existing tracked object to a corresponding
    # centroid (contour). It returns the pairs of (objID, contourID) and lists
    # of unused objects and unused centroids

    # Compute all distances between objects and centroids
    distances = []
    for centroidID, centroid in enumerate(centroids):
        for objID, obj in enumerate(tracked_objects):
            distance = compute_distance(centroid, obj.get_prediction(t))
            distances.append((objID, centroidID, distance))
    # Filter out those that are greater than maximal distance to parse
    distances = filter(lambda d : d[2] < MAX_DISTANCE_TO_PARSE, distances)
    # Sort them in ascending order, so the small distances get prioritized
    distances = sorted(distances, key=lambda d: d[2])

    # Initialise empty lists
    used_objects = []
    used_centroids = []
    pairs = []

    # Find the pairs
    for distance in distances:
        objID, centroidID, _ = distance
        # Do not reuse centroids or objects
        if objID not in used_objects and centroidID not in used_centroids:
            pairs.append((objID, centroidID))
            used_centroids.append(centroidID)
            used_objects.append(objID)

    # find the list of unused objects
    unused_objects = list(tracked_objects)
    for used_object_index in sorted(used_objects, reverse=True):
        del unused_objects[used_object_index]

    # find the list of unused centroids
    unused_centroids = list(centroids)
    for used_centroid_index in sorted(used_centroids, reverse=True):
        del unused_centroids[used_centroid_index]

    return pairs, unused_objects, unused_centroids


def is_valid_contour(contour):
    # Function decides if the contour is good enough to be tracked
    contour_area = cv2.contourArea(contour)
    return contour_area > MIN_CONTOUR_AREA

def create_objects(unused_centroids, tracked_objects, t):
    # Function creates a new object for each unasigned centroid
    for centroid in unused_centroids:
        new_obj = TrackedObject(centroid[0], centroid[1], t)
        tracked_objects.append(new_obj)


def update_pairs(pairs, tracked_objects, centroids, t):
    # Function writes the current measurment into the object
    for pair in pairs:
        obj = tracked_objects[pair[0]]
        x, y = centroids[pair[1]]
        obj.update(x, y, t)


def update_missing(unused_objects, tracked_objects, pass_callback):
    # Function up
    for unused_object in unused_objects:
        if unused_object.missing() == -1:
            if unused_object.get_direction() == 1:
                pass_callback("in")
            if unused_object.get_direction() == -1:
                pass_callback("out")
            tracked_objects.remove(unused_object)

def pass_callback(dir):
    global counter_in, counter_outt
    if dir == "in":
        counter_in += 1
    elif dir == "out":
        counter_outt += 1

def tracking_start():
    frame_delay = 50

    # Initialise videl capture
    cap = cv2.VideoCapture(URL)
    # Take first frame as bacground reference
    _, initial_frame = cap.read()
    bg_reference = cv2.cvtColor(initial_frame,cv2.COLOR_BGR2GRAY)

    # Iterate forever
    tracked_objects = []
    while(cap.isOpened()):
        t = cv2.getTickCount()
        #Read frame
        ret, frame = cap.read()
        if not ret:
            print "Read failed"
            break
        # Obtain thresholded and filtered version
        filtered_fg = filter_frame(frame, bg_reference)

        # Find centroids of all contours
        centroids = find_contour_centroids(frame, filtered_fg)

        pairs, unused_objects, unused_centroids = assign_centroids(
                tracked_objects, centroids, t)


        if GUI:
            # Draw found objects - big circle is object, small circle is
            # prediction of its position
            for pair in pairs:
                obj = tracked_objects[pair[0]]
                centroid = centroids[pair[1]]
                cv2.circle(frame, centroid, 10, obj.color, -1)
                cv2.circle(frame, obj.get_prediction(t), 3, obj.color, -1)
            # Draw objects that were not found as black circles
            for unused_object in unused_objects:
                cv2.circle(frame, unused_object.get_prediction(t), 10, (0,0,0),
                    -1)
        # Create objects for centroid that were not assigned
        create_objects(unused_centroids, tracked_objects, t)
        # Update assigned centroid with current measurements
        update_pairs(pairs, tracked_objects, centroids, t)
        # Delete missing objects and call callbacks
        update_missing(unused_objects, tracked_objects, pass_callback)

        if GUI:
            # Show counters
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,str(counter_outt),(10,50), font, 1,(0,0,255),2)
            cv2.putText(frame,str(counter_in),(10,400), font, 1,(0,0,255),2)
            cv2.imshow('frame',frame)
            cv2.imshow('fg',filtered_fg)
            key = cv2.waitKey(frame_delay)

            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('p'):
                time.sleep(2)
            # Allow user to control speed
            if key & 0xFF == ord('s'):
                frame_delay = 1000
            if key & 0xFF == ord('n'):
                frame_delay = 100
            if key & 0xFF == ord('f'):
                frame_delay = 1


    # Teardown
    cap.release()
    if GUI:
        cv2.destroyAllWindows()
