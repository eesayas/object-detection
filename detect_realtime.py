import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from constants import API_MODEL_PATH, LABEL_MAP

'''----------------------------------------------------------------
find_latest_ckpt

Description: This will return the latest ckpt index from training
Arguments:
- model_path: the path to the trained model

Return: int 
-----------------------------------------------------------------'''
def find_latest_ckpt(model_path):
    max_ckpt = 0

    for subdir, dirs, files in os.walk(model_path):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if 'ckpt' in file_name:
                ckpt = int(file_name.split('-')[1])
                if ckpt > max_ckpt:
                    max_ckpt = ckpt

    return max_ckpt

'''----------------------------------------------------------------------------
detect_realtime

Description: Open webcam and show real time detection of trained model
Arguments:
- model_name: the name of trained model
------------------------------------------------------------------------------'''
def detect_realtime(model_name):
    model_path = os.path.join(API_MODEL_PATH, model_name)
    pipeline_config = os.path.join(model_path, 'pipeline.config')

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    latest_ckpt = 'ckpt-{}'.format(find_latest_ckpt(model_path))
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_path, latest_ckpt)).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP)
    cap = cv2.VideoCapture(0)
    
    while True: 
        ret, frame = cap.read()
        image_np = np.array(frame)
    
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
    
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.75,
            agnostic_mode=False)

        cv2.imshow("Object Detection Model: {} (Press 'q' to quit)".format(model_name),  image_np_with_detections)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break