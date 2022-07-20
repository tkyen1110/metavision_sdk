import os
import cv2
import numpy as np

import torch

# Import of Metavision Machine Learning binding
import metavision_sdk_ml
import metavision_sdk_cv
from metavision_sdk_core import EventBbox


from metavision_core.utils import get_sample

file_dir = "../sample_recordings"
SEQUENCE_FILENAME_RAW = "driving_sample.raw"
# if the file doesn't exist, it will be downloaded from Prophesee's public sample server 
get_sample(SEQUENCE_FILENAME_RAW, folder=file_dir)
SEQUENCE_FILENAME_RAW = os.path.join(file_dir, SEQUENCE_FILENAME_RAW)
assert os.path.isfile(SEQUENCE_FILENAME_RAW)


from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm

DELTA_T = 10000  # 10 ms

def init_event_producer():
    return EventsIterator(SEQUENCE_FILENAME_RAW, start_ts=0, delta_t=DELTA_T, relative_timestamps=False)

#initialize an iterator to get the sensor size
mv_it = init_event_producer()
ev_height, ev_width = mv_it.get_size()

print("Dimensions:", ev_width, ev_height)


USE_EVENT_CUBE_MODEL = False

if USE_EVENT_CUBE_MODEL:
    NN_MODEL_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "../pre_trained_models/red_event_cube_05_2020"))
else:
    NN_MODEL_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "../pre_trained_models/red_histogram_05_2020"))

print("NN_MODEL_DIRECTORY: ", NN_MODEL_DIRECTORY)

# check whether we can use the GPU or we should fall back on the CPU
DEVICE = "cpu"  # "cpu", "cuda" (or "cuda:0", "cuda:1", etc.)
if torch.cuda.is_available():
    DEVICE = "cuda"

NN_DOWNSCALE_FACTOR = 2 # divide events input height and width by this factor before applying NN, this gives us a good trade-off between accuracy and performance
DETECTOR_SCORE_THRESHOLD = 0.4 # ignore all detections below this threshold
NMS_IOU_THRESHOLD = 0.4 # apply Non-Maximum Suppression when the intersection over union (IOU) is above this threshold


from metavision_ml.detection_tracking import ObjectDetector

network_input_width  = ev_width  // NN_DOWNSCALE_FACTOR
network_input_height = ev_height // NN_DOWNSCALE_FACTOR

object_detector = ObjectDetector(NN_MODEL_DIRECTORY,
                                 events_input_width=ev_width,
                                 events_input_height=ev_height,
                                 runtime=DEVICE,
                                 network_input_width=network_input_width,
                                 network_input_height=network_input_height)
object_detector.set_detection_threshold(DETECTOR_SCORE_THRESHOLD)
object_detector.set_iou_threshold(NMS_IOU_THRESHOLD)


cdproc = object_detector.get_cd_processor()
frame_buffer = cdproc.init_output_tensor()
print("frame_buffer.shape: ", frame_buffer.shape)
if USE_EVENT_CUBE_MODEL:
    assert frame_buffer.shape == (10, network_input_height, network_input_width)
else:
    assert frame_buffer.shape == (2, network_input_height, network_input_width)
assert (frame_buffer == 0).all()


NN_accumulation_time = object_detector.get_accumulation_time()
def generate_detection(ts, ev):
    current_frame_start_ts = ((ts-1) // NN_accumulation_time) * NN_accumulation_time
    cdproc.process_events(current_frame_start_ts, ev, frame_buffer)

    detections = np.empty(0, dtype=EventBbox)
    
    if ts % NN_accumulation_time == 0:  # call the network only when defined
        # call neural network to detect objects 
        detections = object_detector.process(ts, frame_buffer)
        # reset neural network input frame
        frame_buffer.fill(0)
    return detections
print("NN_accumulation_time: ", NN_accumulation_time)


TRAIL_THRESHOLD=10000
trail = metavision_sdk_cv.TrailFilterAlgorithm(width=ev_width, height=ev_height, threshold=TRAIL_THRESHOLD)

# Filter done after the detection
ev_filtered_buffer = trail.get_empty_output_buffer()
def noise_filter(ev):
    # apply trail filter
    trail.process_events(ev, ev_filtered_buffer)
    return ev_filtered_buffer.numpy()


def init_data_association():
    return metavision_sdk_ml.DataAssociation(width=ev_width, height=ev_height, max_iou_inter_track=0.3)


DO_DISPLAY = True and os.environ.get("DOC_DISPLAY", 'ON') != "OFF" # display the result in a window
OUTPUT_VIDEO = os.path.join(file_dir, "detection_and_tracking.mp4") # output video (disabled if the string is empty. Set a file name to save the video)
print("DO_DISPLAY: ", DO_DISPLAY)
print("OUTPUT_VIDEO: ", OUTPUT_VIDEO)

import numpy as np
from skvideo.io import FFmpegWriter

def init_output():
    if OUTPUT_VIDEO:
        assert OUTPUT_VIDEO.lower().endswith(".mp4"), "Video should be mp4"

    if DO_DISPLAY:
        cv2.namedWindow("Detection and Tracking", cv2.WINDOW_NORMAL)
    
    return FFmpegWriter(OUTPUT_VIDEO) if OUTPUT_VIDEO else None

if OUTPUT_VIDEO or DO_DISPLAY:
    frame = np.zeros((ev_height, ev_width * 2, 3), dtype=np.uint8)

from metavision_ml.detection_tracking import draw_detections_and_tracklets
from metavision_sdk_core import BaseFrameGenerationAlgorithm

def generate_display(ts, ev, detections, tracklets, process_video):
    if OUTPUT_VIDEO or DO_DISPLAY:
        # build image frame
        BaseFrameGenerationAlgorithm.generate_frame(ev, frame[:, :ev_width])
        frame[:, ev_width:] = frame[:, :ev_width]
        draw_detections_and_tracklets(ts=ts, frame=frame, width=ev_width, height=ev_height,
                                         detections=detections, tracklets=tracklets)

    if DO_DISPLAY:
        # display image on screen
        cv2.imshow('Detection and Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    if OUTPUT_VIDEO:
        # write video
        process_video.writeFrame(frame[...,::-1].astype(np.uint8))
        
    return True
        
def end_display(process_video):
    # close video and window
    if OUTPUT_VIDEO:
        process_video.close()

    if DO_DISPLAY:
        cv2.destroyAllWindows()


mv_it = init_event_producer() # initialize the iterator to read the events
object_detector.reset() # reset the object detector internal memory before processing a sequence
data_assoc = init_data_association() #  initialize the data association block
data_assoc_buffer = data_assoc.get_empty_output_buffer()
process_video = init_output() # initialize the video generation

END_TS = 10 * 1e6 # process sequence until this timestamp (None to disable)

for ev in mv_it:
    ts = mv_it.get_current_time()

    if END_TS and ts > END_TS:
        break
    
    # run the detectors and get the output
    detections = generate_detection(ts, ev)

    # remove noisy events for processing with the data association block    
    noise_filtered_ev = noise_filter(ev)

    # compute tracklets
    data_assoc.process_events(ts, noise_filtered_ev, detections, data_assoc_buffer)
    tracklets = data_assoc_buffer.numpy()
    
    if not generate_display(ts, ev, detections, tracklets, process_video):
        # if the generation is stopped using `q`, break the loop
        break

# finalize the recording
end_display(process_video)
