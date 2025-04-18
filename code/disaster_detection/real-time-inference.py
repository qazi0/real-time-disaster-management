import cv2
import time
import torch
import argparse
import logging
import numpy as np
from typing import Tuple, Optional
from PIL import Image
from imutils.video import WebcamVideoStream, FileVideoStream

from model.ernet import ErNET
from model.squeeze_ernet import Squeeze_ErNET
from model.squeeze_ernet_redconv import Squeeze_RedConv
from dataloaders.aider import aider_transforms, squeeze_transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_name: str, weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load model and weights."""
    # Initialize model based on name
    if model_name == 'ernet':
        model = ErNET()
    elif model_name == 'squeeze-ernet':
        model = Squeeze_ErNET()
    elif model_name == 'squeeze-redconv':
        model = Squeeze_RedConv()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Modern checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Legacy format (just the model)
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_frame(
    frame: np.ndarray,
    transform,
    input_shape: Tuple[int, int, int, int],
    device: torch.device
) -> torch.Tensor:
    """Preprocess frame for inference."""
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    pil_image = Image.fromarray(frame)
    tensor = transform(pil_image)
    
    # Reshape and move to device
    tensor = torch.reshape(tensor, input_shape).to(device)
    return tensor

def run_inference(
    model: torch.nn.Module,
    frame: np.ndarray,
    transform,
    input_shape: Tuple[int, int, int, int],
    device: torch.device,
    use_trt: bool = False,
    quant: str = 'fp16'
) -> Tuple[str, float]:
    """Run inference on a single frame."""
    # Preprocess frame
    tensor = preprocess_frame(frame, transform, input_shape, device)
    
    # Run inference
    with torch.no_grad():
        if use_trt and quant == 'fp16':
            tensor = tensor.half()
        output = model(tensor)
        predicted_class = output.data.max(1, keepdim=True)[1]
        
        # Get confidence
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item() * 100
    
    # Map class index to name
    classes = ['collapsed building', 'fire', 'flooded areas', 'normal', 'traffic incident']
    predicted_class_name = classes[predicted_class.item()]
    
    return predicted_class_name, confidence

def visualize_prediction(
    frame: np.ndarray,
    prediction: str,
    confidence: float,
    fps: float
) -> np.ndarray:
    """Visualize prediction on frame."""
    # Add prediction text
    text = f"{prediction} ({confidence:.1f}%)"
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    
    # Add FPS counter
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Real-time disaster detection inference')
    parser.add_argument('--model', type=str, default='ernet',
                        choices=['ernet', 'squeeze-ernet', 'squeeze-redconv'],
                        help='model architecture')
    parser.add_argument('--weights', type=str, required=True,
                        help='path to model weights')
    parser.add_argument('--video', type=str, default=None,
                        help='path to video file (if None, use webcam)')
    parser.add_argument('--width', type=int, default=640,
                        help='frame width')
    parser.add_argument('--height', type=int, default=480,
                        help='frame height')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--trt', action='store_true',
                        help='use TensorRT for inference')
    parser.add_argument('--quant', type=str, default='fp16',
                        choices=['fp16', 'fp32'],
                        help='quantization scheme for TensorRT')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = load_model(args.model, args.weights, device)
    
    # Get appropriate transforms and input shape
    transforms = aider_transforms if args.model == 'ernet' else squeeze_transforms
    input_shape = (1, 3, 240, 240) if args.model == 'ernet' else (1, 3, 140, 140)
    
    # Initialize video stream
    if args.video:
        vs = FileVideoStream(args.video).start()
    else:
        vs = WebcamVideoStream(src=0).start()
    
    # Initialize timing variables
    prev_frame_time = 0
    fps_list = []
    
    logger.info("Starting inference...")
    try:
        while True:
            # Read frame
            frame = vs.read()
            if frame is None:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (args.width, args.height))
            
            # Run inference
            prediction, confidence = run_inference(
                model, frame, transforms, input_shape,
                device, args.trt, args.quant
            )
            
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1.0 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_list.append(fps)
            
            # Visualize results
            frame = visualize_prediction(frame, prediction, confidence, fps)
            
            # Display frame
            cv2.imshow("Disaster Detection", frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        vs.stop()
        
        # Print statistics
        avg_fps = sum(fps_list) / len(fps_list)
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Min FPS: {min(fps_list):.2f}")
        logger.info(f"Max FPS: {max(fps_list):.2f}")

if __name__ == '__main__':
    main()
