import argparse
from path import Path
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np



(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
    parser.add_argument("--output-file", "-o", default=None, help="Output numpy file")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_file = Path(args.output_file)

    # first and last frame to track
    first_index = 691
    last_index = 731

    # Set up tracker.

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[4]
    if int(minor_ver) < 0:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # Read first frame
    frame = cv2.imread(dataset_dir/'{:06}'.format(first_index)+'.png')

    # Select a bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    index = first_index
    arr = np.zeros(shape=(last_index-first_index,4))
    while index < last_index:

        # Read a new frame
        frame = cv2.imread(dataset_dir/'{:06}'.format(index)+'.png')
        index = index+1

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box and save result
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            height = frame.shape[0]
            width = frame.shape[1]
            arr[index-first_index-1][0] = int(p1[0]/(width/416))
            arr[index-first_index-1][1] = int(p1[1]/(height/128))
            arr[index-first_index-1][2] = int(p2[0]/(width/416))
            arr[index-first_index-1][3] = int(p2[1]/(height/128))
            for i,num in enumerate(arr[index-first_index-1]):
                if num < 0:
                    arr[index-first_index-1][i] = 0
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27 : break
    print('finished')
    np.save(output_file,arr)
