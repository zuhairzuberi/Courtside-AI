from __future__ import print_function
from imutils.object_detection import non_max_suppression
import cv2
import numpy as np
from easydict import EasyDict
from random import randint
import sys
from imutils.video import FPS
import torch
import torch.nn as nn
from torchvision import models
from utils.checkpoints import load_weights

args = EasyDict({
    # Change the 'videoPath' to the file name you want to eval.
    'videoPath': "evaluation_videos/test.mp4",
    
    'detector': "tracker",
    'classes': ["person"],
    'tracker': "CSRT",
    'trackerTypes': ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'],
    'singleTracker': False,
    'draw_line': False,
    'weights': "yolov3.weights",
    'config': "yolov3.cfg",
    'COLORS': np.random.uniform(0, 255, size=(1, 3)),
    'base_model_name': 'r2plus1d_multiclass',
    'pretrained': True,
    'lr': 0.0001,
    'start_epoch': 1,
    'num_classes': 13,
    'seq_length': 16,
    'vid_stride': 8,
    'labels': {"0":"block", "1":"pass", "2":"run", "3":"dribble", "4":"shoot", "5":"ball in hand", "6":"defense", "7":"pick", "8":"no_action", "9":"walk", "10":"discard", "11":"dunk", "12":"layup"},
    'model_path': "model_checkpoints/r2plus1d_augmented-2/",
    'history_path': "histories/history_r2plus1d_augmented-2.txt",
    
    # Change the 'trained_model_path' to the path of which ever model checkpoint you want to use.
    'trained_model_path': "model_checkpoints/r2plus1d_augmented-2/r2plus1d_multiclass_15_0.0001.pt",
})

# Creates an object tracker based on the specified trackerType.
# Uses OpenCV's tracker types found in args.trackerTypes.
def createTrackerByName(trackerType):
    if trackerType == args.trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == args.trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == args.trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == args.trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == args.trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == args.trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == args.trackerTypes[6]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif trackerType == args.trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in args.trackerTypes:
            print(t)

    return tracker

# Grabs and returns the names of the YOLO object detection output layers.
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# Draws the bounding box around the human object (class_id == 0) selected at the start.
# Writes down the model prediction on top of the bounding box.
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    if class_id == 0:
        label = str(args.classes[class_id])
        color = args.COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Main process of extracting the video frames from our given input file.
def extractFrame(videoPath):
    videoFrames = []
    playerBoxes = []

    # Extracting the OpenCV version info.
    (major, minor) = cv2.__version__.split(".")[:2]
    
    # For OpenCV 3.2 or before.
    if args.detector == "tracker":
        if int(major) == 3 and int(minor) < 3:
            if args.singleTracker:
                tracker = cv2.Tracker_create(args.tracker.upper())
        else:
            if args.singleTracker:
                OPENCV_OBJECT_TRACKERS = {
                    "csrt": cv2.legacy.TrackerCSRT_create(),
                    "kcf": cv2.legacy.TrackerKCF_create(),
                    "mil": cv2.legacy.TrackerMIL_create()
                }

                tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()

    # Initializing the bounding box coordinates of the selected object as well as the FPS.
    initBB = None
    fps = None

    # Setting up the YOLOv3 Neural Network.
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    cap = cv2.VideoCapture(videoPath)

    # Technically, we can select as many players as we like but it will significantly increase the compute time.
    player_threshold = 99999

    # Tries to read the first frame of the video, if unsuccessful exit the program.
    if not args.singleTracker:
        success, frame = cap.read()
        
        if not success:
            print('Unable to access the video.')
            sys.exit(1)

        bboxes = []
        colors = []

        # Using an infinite loop, we can keep selecting more objects using OpenCV's selectROI function.
        while True:
            bbox = cv2.selectROI('MultiTracker', frame, fromCenter=False, showCrosshair=True)
            bboxes.append(bbox)
            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
            print("Press 'Q' to quit and to start tracking.")
            print("Press ANY KEY to select another object.")
            
            checkIfQ = cv2.waitKey(0) & 0xFF
            if (checkIfQ == 113):
                break

        createTrackerByName(args.tracker)
        trackers = cv2.legacy.MultiTracker_create()

        for bbox in bboxes:
            trackers.add(createTrackerByName(args.tracker), frame, bbox)

    frameCount = 0
    while (cap.isOpened()):
        _, frame = cap.read()

        if not _:
            break

        Width = frame.shape[1]
        Height = frame.shape[0]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        court_color = np.uint8([[[189, 204, 233]]])
        hsv_court_color = cv2.cvtColor(court_color, cv2.COLOR_BGR2HSV)
        hue = hsv_court_color[0][0][0]

        lower_color = np.array([hue - 5, 10, 10])
        upper_color = np.array([hue + 5, 225, 225])

        mask = cv2.inRange(hsv, lower_color, upper_color)

        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        res = cv2.bitwise_and(frame, frame, mask=opening)
        cv2.imshow('res', res)

        if args.draw_line:
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            low_thresh = 0.5 * high_thresh
            edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=3)
            cv2.imshow('Canny Edge Detector', edges)

            minLineLength = 200
            maxLineGap = 500
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

            LINE_COLOR = (255, 0, 0)

            if lines is None:
                continue
            else:
                a, b, c = lines.shape
                for i in range(2):
                    for x1, y1, x2, y2 in lines[i]:
                        if args.draw_line:
                            cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 3)
                        player_threshold = min(player_threshold, y1, y2)

        # Detecting people.
        if args.detector == "HOG":
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            orig = frame.copy()

            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

            for (x, y, w, h) in rects:
                cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.1)

            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        elif args.detector == "yolov3":
            scale = 0.00392
            blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            k = 0
            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                pad = 5

                if (round(y + h) < player_threshold):
                    k += 1
                    continue
                else:
                    draw_prediction(frame, class_ids[i], round(x - pad), round(y - pad), round(x + w + pad), round(y + h + pad))
        elif args.detector == "tracker":
            if args.singleTracker:
                if initBB is not None:
                    (success, box) = tracker.update(frame)

                    if success:
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 255, 0), 2)

                    fps.update()
                    fps.stop()

                    info = [
                        ("Tracker", tracker),
                        ("Success", "Yes" if success else "No"),
                        ("FPS", "{:.2f}".format(fps.fps())),
                    ]
                    
                    for (i, (k, v)) in enumerate(info):
                        text = "{}: {}".format(k, v)
                        cv2.putText(frame, text, (10, Height - ((i * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                videoFrames.append(frame)
                success, boxes = trackers.update(frame)
                playerBoxes.append(boxes)

                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        else:
            continue

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if args.singleTracker:
                initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
                tracker.init(frame, initBB)
                fps = FPS().start()
        elif key == ord("q"):
            break

        frameCount += 1

    cap.release()
    cv2.destroyAllWindows()

    return videoFrames, playerBoxes, Width, Height, colors

# Makes sure that each frame of the given input video has been resized to 128x176 for model compatibility.
def cropVideo(clip, crop_window, player=0):
    video = []

    for i, frame in enumerate(clip):
        x = int(crop_window[i][player][0])
        y = int(crop_window[i][player][1])
        w = int(crop_window[i][player][2])
        h = int(crop_window[i][player][3])

        cropped_frame = frame[y:y+h, x:x+w]
        
        # Resizes to 128x176 here.
        try:
            resized_frame = cv2.resize(
                cropped_frame,
                dsize=(int(128), int(176)),
                interpolation=cv2.INTER_NEAREST)
        except:
            if len(video) == 0:
                resized_frame = np.zeros((int(176), int(128), 3), dtype=np.uint8)
            else:
                resized_frame = video[i-1]
        assert resized_frame.shape == (176, 128, 3)
        video.append(resized_frame)

    return video

# Permutes the batch of frames.
def inference_batch(batch):
    batch = batch.permute(0, 4, 1, 2, 3)
    return batch

# Combines the frames into a fixed length, extracts the cropped frame around each of the players
# which were determined by the bounding box.
def cropWindows(vidFrames, playerBoxes, seq_length=16, vid_stride=8):
    player_count = len(playerBoxes[0])
    player_frames = {}
    for player in range(player_count):
        player_frames[player] = []

    n_clips = len(vidFrames) // vid_stride

    continue_clip = 0
    for clip_n in range(n_clips):
        crop_window = playerBoxes[clip_n*vid_stride: clip_n*vid_stride + seq_length]
        for player in range(player_count):
            if clip_n*vid_stride + seq_length < len(vidFrames):
                clip = vidFrames[clip_n*vid_stride: clip_n*vid_stride + seq_length]
                player_frames[player].append(np.asarray(cropVideo(clip, crop_window, player)))
            else:
                continue_clip = clip_n
                break
        if continue_clip != 0:
            break

    for i in range(continue_clip, n_clips):
        for player in range(player_count):
            crop_window = playerBoxes[vid_stride*i:]
            frames_remaining = len(vidFrames) - vid_stride * i
            clip = vidFrames[vid_stride*i:]
            player_frames[player].append(np.asarray(cropVideo(clip, crop_window, player) + [
            np.zeros((int(176), int(128), 3), dtype=np.uint8) for x in range(seq_length-frames_remaining)
        ]))

    # Checking if frames match the total number of clips.
    assert(len(player_frames[0]) == n_clips)

    return player_frames

# Writes out the output video with bounding boxes and predictions on top of the the original frames.
def writeVideo(videoPath, videoFrames, playerBoxes, predictions, colors, frame_width=1280, frame_height=720, vid_stride=8):
    out = cv2.VideoWriter(videoPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width, frame_height))
    for i, frame in enumerate(videoFrames):

        # Drawing bounding boxes here.
        for player in range(len(playerBoxes[0])):
            box = playerBoxes[i][player]

            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, colors[player], 2, 1)

            # Write the predictions.
            if i // vid_stride < len(predictions[player]):
                cv2.putText(frame, args.labels[str(predictions[player][i // vid_stride])], (p1[0] - 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[player], 2)

        cv2.imshow('frame', frame)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

def main():
    videoFrames, playerBoxes, Width, Height, colors = extractFrame(args.videoPath)
    frames = cropWindows(videoFrames, playerBoxes, seq_length=args.seq_length, vid_stride=args.vid_stride)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initializing the model.
    model = models.video.r2plus1d_18(pretrained=args.pretrained, progress=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10, bias=True)
    model = load_weights(model, args.trained_model_path)

    if torch.cuda.is_available():
        model = model.to(device)

    model.eval()

    predictions = {}
    for player in range(len(playerBoxes[0])):
        input_frames = inference_batch(torch.FloatTensor(frames[player]))
        input_frames = input_frames.to(device=device)

        with torch.no_grad():
            outputs = model(input_frames)
            _, preds = torch.max(outputs, 1)

        predictions[player] = preds.cpu().numpy().tolist()

    file_name = args.videoPath.split('/')[-1].split('.')[0]
    output_path = "evaluation_videos/output/" + file_name + "_output.mp4"
    print("\nVideo has finished processing. View your results here: " + output_path)
    
    writeVideo(output_path, videoFrames, playerBoxes, predictions, colors, frame_width=1280, frame_height=720, vid_stride=16)

if __name__ == "__main__":
    main()