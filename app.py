import base64
import io
import json
import os
from PIL import Image
import runpod

# Import your models
from table_detection import OBBModule
from table_recognition import SuryaOCRAgent
import utils

# Initialize models (using singleton pattern to avoid reinitializing on each request)
detection_model = OBBModule(model_path='dynamic_quantized_21.onnx')
recognition_model = SuryaOCRAgent.get_instance()

def handler(job):
    """
    This is the handler function that processes all incoming jobs
    from RunPod's serverless platform.
    """
    job_input = job["input"]
    route = job_input.get("route", "")

    try:
        if route == "detect":
            batch_data = job_input["batch_data"]
            
            # Process each item in the batch
            image_pairs = []
            for item in batch_data:
                low_res_base64 = item["low_res"]
                high_res_base64 = item.get("high_res", low_res_base64)  # Use low_res if high_res not provided
                
                low_res_image = base64_to_pil_image(low_res_base64)
                high_res_image = base64_to_pil_image(high_res_base64)
                
                image_pairs.append((low_res_image, high_res_image))
            
            # Run batch detection
            detection_results = detection_model.detect_bbox_batch(image_pairs)
            
            return {"detection_results": detection_results}

        elif route == "recognize":
            # Handle recognize route
            image_base64 = job_input["image"]
            
            # Convert base64 to PIL Image
            image = base64_to_pil_image(image_base64)
            image_b64 = utils.pil_image_to_base64(image)
            
            # Recognize tables
            extracted_html = recognition_model(image_b64)
            
            return {
                "html": extracted_html
            }
            
        # elif route == "get_extract_info":
        #     # Handle get_extract_info route
        #     if os.path.exists('ocr_results.json'):
        #         with open('ocr_results.json', 'r') as f:
        #             return json.load(f)
        #     else:
        #         return []
        
        else:
            return {"error": f"Unknown route: {route}"}
            
    except Exception as e:
        return {"error": str(e)}

def base64_to_pil_image(base64_str):
    """Helper function to convert base64 string to PIL Image"""
    if "base64," in base64_str:  # Handle data URLs
        base64_str = base64_str.split("base64,")[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})