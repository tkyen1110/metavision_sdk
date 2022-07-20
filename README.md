# metavision_sdk

## Markdown
* https://code.visualstudio.com/docs/languages/markdown
* https://www.mdeditor.tw

## Origin of Codes
* include is from /usr/include/metavision
* metavision is from /usr/share/metavision (except /usr/share/metavision/apps)
* metavision_core is from /usr/lib/python3/dist-packages/metavision_core
* metavision_core_ml is from /usr/lib/python3/dist-packages/metavision_core_ml
* metavision_ml is from /usr/lib/python3/dist-packages/metavision_ml

## [Metavision SDK >> Modules >> Machine Learning](https://docs.prophesee.ai/stable/metavision_sdk/modules/ml/index.html)

### [Samples and Applications](https://docs.prophesee.ai/stable/metavision_sdk/modules/ml/samples/index.html)

+ [Inference Pipeline of Detection and Tracking using Python](https://docs.prophesee.ai/stable/metavision_sdk/modules/ml/samples/detection_and_tracking_inference.html)
```bash
python3 metavision/sdk/ml/python_samples/detection_and_tracking_pipeline/detection_and_tracking_pipeline.py \
--object_detector_dir pre_trained_models/red_event_cube_05_2020 \
--record_file sample_recordings/driving_sample.raw --display 
```

+ [Inference Pipeline of Detection and Tracking using C++](https://docs.prophesee.ai/stable/metavision_sdk/modules/ml/samples/detection_and_tracking_inference_cpp.html)
```bash
cd metavision/sdk/ml/samples/metavision_detection_and_tracking_pipeline/
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$LIBTORCH_DIR_PATH -DTorch_DIR=$LIBTORCH_DIR_PATH
cmake --build .
cd ../../../../../../

./metavision/sdk/ml/samples/metavision_detection_and_tracking_pipeline/build/metavision_detection_and_tracking_pipeline \
--object-detector-dir pre_trained_models/red_event_cube_05_2020 \
--record-file sample_recordings/driving_sample.raw --display
```


## Tutorial
### 1. [Get Started using Python](https://docs.prophesee.ai/stable/metavision_sdk/get_started/index.html)
```bash
python3 metavision/sdk/core/python_samples/metavision_sdk_get_started/metavision_sdk_get_started.py 
-i sample_recordings/driving_sample.raw
```

### 2. ML Python Tutorials
+ [Quick Start](https://docs.prophesee.ai/stable/metavision_sdk/modules/ml/quick_start/index.html)
+ [Inference - Detection and Tracking Tutorial](https://docs.prophesee.ai/stable/metavision_sdk/modules/ml/inference/detection_and_tracking.html)
