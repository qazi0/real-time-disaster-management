import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from typing import Tuple, Dict
import logging

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

def predict(
	model: torch.nn.Module,
	image_path: str,
	transform,
	device: torch.device,
	use_trt: bool = False,
	quant: str = 'fp16'
) -> Tuple[str, float]:
	"""Make prediction on a single image."""
	# Load and preprocess image
	image = cv2.imread(image_path)
	if image is None:
		raise ValueError(f"Could not load image at {image_path}")
	
	# Convert BGR to RGB
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	# Apply transforms
	pil_image = Image.fromarray(image)
	data = transform(pil_image)
	
	# Reshape for model input
	shape = (1, 3, 240, 240) if isinstance(model, ErNET) else (1, 3, 140, 140)
	data = torch.reshape(data, shape).to(device)
	
	# Make prediction
	with torch.no_grad():
		if use_trt and quant == 'fp16':
			data = data.half()
		output = model(data)
		predicted_class = output.data.max(1, keepdim=True)[1]
		
		# Get confidence
		confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item() * 100
	
	# Map class index to name
	classes = ['collapsed building', 'fire', 'flooded areas', 'normal', 'traffic incident']
	predicted_class_name = classes[predicted_class.item()]
	
	return predicted_class_name, confidence

def visualize_prediction(
	image: np.ndarray,
	prediction: str,
	confidence: float,
	trt_prediction: str = None,
	trt_confidence: float = None
) -> None:
	"""Visualize prediction on image."""
	# Add main prediction
	text = f"{prediction} ({confidence:.1f}%)"
	image = cv2.putText(
		image, text, (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		(255, 255, 255), 2, cv2.LINE_AA
	)
	
	# Add TensorRT prediction if available
	if trt_prediction is not None:
		text = f"TRT: {trt_prediction} ({trt_confidence:.1f}%)"
		image = cv2.putText(
			image, text, (10, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7,
			(255, 255, 255), 2, cv2.LINE_AA
		)
	
	# Convert back to RGB for display
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	# Display image
	import matplotlib.pyplot as plt
	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	plt.axis('off')
	plt.show()

def main():
	parser = argparse.ArgumentParser(description='Predict disaster types from aerial images')
	parser.add_argument('--model', type=str, default='ernet',
						choices=['ernet', 'squeeze-ernet', 'squeeze-redconv'],
						help='model architecture')
	parser.add_argument('--image', type=str, required=True,
						help='path to input image')
	parser.add_argument('--weights', type=str, default=None,
						help='path to model weights')
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
	
	# Set default weights if not provided
	if args.weights is None:
		args.weights = f'weights/{args.model}.pt'
		if not os.path.exists(args.weights):
			raise FileNotFoundError(f"No weights found at {args.weights}")
	
	# Load model
	model = load_model(args.model, args.weights, device)
	
	# Get appropriate transforms
	transforms = aider_transforms if args.model == 'ernet' else squeeze_transforms
	
	# Make prediction
	prediction, confidence = predict(model, args.image, transforms, device)
	logger.info(f"Prediction: {prediction} ({confidence:.1f}%)")
	
	# TensorRT inference if requested
	trt_prediction = None
	trt_confidence = None
	if args.trt:
		try:
			from torch2trt import TRTModule
			trt_model = TRTModule()
			trt_model.load_state_dict(torch.load(f'tensorrt_state_dicts/{args.model}_{args.quant}_trt.pth'))
			trt_model = trt_model.to(device)
			trt_prediction, trt_confidence = predict(trt_model, args.image, transforms, device, True, args.quant)
			logger.info(f"TensorRT Prediction: {trt_prediction} ({trt_confidence:.1f}%)")
		except ImportError:
			logger.warning("TensorRT not available. Skipping TensorRT inference.")
	
	# Visualize results
	image = cv2.imread(args.image)
	visualize_prediction(image, prediction, confidence, trt_prediction, trt_confidence)

if __name__ == '__main__':
	main()


