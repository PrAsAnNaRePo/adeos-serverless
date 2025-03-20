import os
import base64
import cv2
import numpy as np
import onnxruntime
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image

class OBBModule: 
    def __init__(self, model_path, class_labels=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Set the maximum number of parallel workers
        self.max_workers = os.cpu_count() - 4
        
        # Create a session options object for optimizations
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1  # Set to 1 for better parallelism
        
        try:
            # Try to use CUDA if available, otherwise fall back to CPU
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(
                model_path, 
                sess_options=session_options,
                providers=providers
            )
            print(f"Successfully loaded ONNX model from {model_path}")
            print(f"Using provider: {self.session.get_providers()[0]}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the ONNX model: {e}")
        
        model_inputs = self.session.get_inputs()
        print("Model Inputs:", model_inputs)
        
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # [batch_size, channels, height, width]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        self.classes = class_labels if class_labels else {0: 'tables', 1: 'tilted', 2: 'empty'}

    def preprocess(self, image):
        """Preprocess a single image for inference"""
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            frame = image
            
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)  # CHW format
        batched = np.expand_dims(transposed, axis=0)  # Add batch dimension
        return batched

    def detect_single(self, 
                     low_res_image, 
                     high_res_image=None, 
                     confidence_threshold=0.35, 
                     iou_threshold=0.45) -> Dict:
        """
        Process a single image pair (low_res for detection, high_res for annotation)
        """
        # Convert PIL Image to OpenCV format if needed
        if isinstance(low_res_image, Image.Image):
            frame = cv2.cvtColor(np.array(low_res_image), cv2.COLOR_RGB2BGR)
        else:
            frame = low_res_image
            
        original_height, original_width = frame.shape[:2]
        
        if high_res_image is not None:
            if isinstance(high_res_image, Image.Image):
                img2 = cv2.cvtColor(np.array(high_res_image), cv2.COLOR_RGB2BGR)
            else:
                img2 = high_res_image
                
            target_height, target_width = img2.shape[:2]
            scale_factor_x = target_width / original_width
            scale_factor_y = target_height / original_height
        else:
            img2 = frame.copy()
            target_height, target_width = original_height, original_width
            scale_factor_x = scale_factor_y = 1
        
        # Preprocess and run inference
        preprocessed = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: preprocessed})
        
        # Postprocess the outputs to get bounding boxes
        obb_data = self.postprocess(outputs, img2, confidence_threshold, iou_threshold, 
                                   scale_factor_x, scale_factor_y)

        # Encode the output image
        _, buffer = cv2.imencode('.png', img2)
        base_img_string = base64.b64encode(buffer).decode('utf-8')

        # Create response
        response = {
            "bbox_data": obb_data,
            "actual_image": base_img_string,
            "height": int(img2.shape[0]),
            "width": int(img2.shape[1]),
            "num_tables": int(len(obb_data)),
        }
        return response

    def detect_bbox_batch(self, 
                         image_pairs: List[Tuple[Image.Image, Optional[Image.Image]]], 
                         confidence_threshold=0.35, 
                         iou_threshold=0.45) -> List[Dict]:
        """
        Process multiple image pairs in parallel
        
        Args:
            image_pairs: List of tuples (image1, image2) where:
                         - image1 is the low-res image for detection
                         - image2 is the high-res image for annotation (optional)
            confidence_threshold: Minimum confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
        
        Returns:
            List of detection results, one per image pair
        """
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for idx, (low_res, high_res) in enumerate(image_pairs):
                future = executor.submit(
                    self.detect_single, 
                    low_res, 
                    high_res, 
                    confidence_threshold, 
                    iou_threshold
                )
                future_to_idx[future] = idx
            
            # Collect results in the original order
            results = [None] * len(image_pairs)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f'Image at index {idx} generated an exception: {exc}')
                    # Provide an empty result on error
                    results[idx] = {
                        "bbox_data": [],
                        "actual_image": "",
                        "height": 0,
                        "width": 0,
                        "num_tables": 0,
                        "error": str(exc)
                    }
        
        return results

    def detect_bbox(self, image1, image2=None, confidence_threshold=0.35, iou_threshold=0.45):
        """
        Original method for backward compatibility
        """
        return self.detect_single(image1, image2, confidence_threshold, iou_threshold)

    def postprocess(self, outputs, img2, confidence_threshold, iou_threshold, scale_factor_x, scale_factor_y):
        img_height, img_width = img2.shape[:2]
        output_array = np.squeeze(outputs[0])

        if output_array.shape[0] < output_array.shape[1]:
            output_array = output_array.transpose()

        num_detections = output_array.shape[0]
        
        boxes = []
        scores = []
        class_ids = []

        # scaled based on model input size to img2
        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        for i in range(num_detections):
            row = output_array[i]
            objectness = row[4]
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores)) 
            confidence = float(class_scores[class_id]) 

            if confidence >= confidence_threshold:
                x, y, width, height = row[0], row[1], row[2], row[3]
                x1 = int((x - width / 2) * x_factor)
                y1 = int((y - height / 2) * y_factor)
                w = int(width * x_factor)
                h = int(height * y_factor)
                
                boxes.append([x1, y1, w, h])
                scores.append(float(confidence))
                class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)

        obb_data = []

        if len(indices) > 0:
            if isinstance(indices[0], (list, tuple, np.ndarray)):
                indices = [i[0] for i in indices]
            else:
                indices = list(indices)
            
            for idx in indices:
                box = boxes[idx]
                class_id = class_ids[idx]
                confidence = scores[idx]
                
                x1, y1, w, h = box
                x2 = x1 + w
                y2 = y1 + h
    
                x = x1 + w / 2
                y = y1 + h / 2

                obb_data.append({
                    "class_id": class_id,
                    "xyxy": [x1, y1, x2, y2],
                    "xywh": [x, y, w, h]
                })

        return obb_data