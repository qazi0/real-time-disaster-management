import os
import torch
import argparse
import logging
from typing import Optional, Tuple
import traceback
from torch2trt import torch2trt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tensorrt_conversion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_name: str, weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load model and weights."""
    try:
        logger.info(f"Loading model {model_name} from {weights_path}")
        
        # Initialize model based on name
        if model_name == 'ernet':
            from model.ernet import ErNET
            model = ErNET()
        elif model_name == 'squeeze-ernet':
            from model.squeeze_ernet import Squeeze_ErNET
            model = Squeeze_ErNET()
        elif model_name == 'squeeze-redconv':
            from model.squeeze_ernet_redconv import Squeeze_RedConv
            model = Squeeze_RedConv()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Verify weights file exists
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load weights
        logger.info("Loading model weights...")
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Modern checkpoint format
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                # Alternative modern format
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Try loading the entire checkpoint as state dict
                model.load_state_dict(checkpoint)
        else:
            # Legacy format (just the model)
            model.load_state_dict(checkpoint)
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Verify model is properly loaded
        with torch.no_grad():
            test_input = torch.randn(1, 3, 140, 140).to(device)
            output = model(test_input)
            logger.info(f"Model loaded successfully. Test output shape: {output.shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def build_trt_model(model: torch.nn.Module, input_tensor: torch.Tensor, args) -> torch.nn.Module:
    """Convert PyTorch model to TensorRT model."""
    logger.info(f'Building TensorRT model [{args.quant}]...')
    
    try:
        # Move model and input to GPU
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        
        # Configure TensorRT settings
        # Configure TensorRT settings
        max_batch_size = 256
        max_workspace_size = 1 << 31  # 2GB
        
        # Log detailed GPU memory status before conversion
        logger.info(f"GPU memory allocated before conversion: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        logger.info(f"GPU memory reserved before conversion: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
        logger.info(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
        
        # Verify model is in eval mode
        if model.training:
            logger.warning("Model was in training mode, switching to eval mode")
            model.eval()
        
        # Verify input tensor
        # Verify input tensor
        logger.info(f"Input tensor shape: {input_tensor.shape}")
        logger.info(f"Input tensor dtype: {input_tensor.dtype}")
        
        # Check for NaN or Inf values in input
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logger.warning("Input tensor contains NaN or Inf values. Normalizing...")
            input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        def attempt_conversion(model, input_tensor, config, attempt_name):
            """Helper function to try conversion with different parameters"""
            # Log conversion attempt
            logger.info(f"Attempting TensorRT conversion - {attempt_name}")
            logger.info(f"Parameters: {config}")
            
            # Force clean CUDA cache before each attempt
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log memory status
            logger.info(f"GPU memory before attempt: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            
            try:
                # Create dedicated CUDA stream
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    # Clone input tensor to avoid side effects
                    input_clone = input_tensor.clone().detach()
                    
                    # Handle FP16 conversion if needed
                    if config.get('fp16_mode', False) and input_clone.dtype != torch.float16:
                        input_clone = input_clone.half()
                    
                    trt_model = torch2trt(
                        model, 
                        [input_clone],
                        **config
                    )
                    
                # Wait for operations to complete
                stream.synchronize()
                torch.cuda.synchronize()
                
                # Log success
                logger.info(f"TensorRT conversion '{attempt_name}' successful")
                return trt_model
            
            except Exception as e:
                # Log failure
                logger.warning(f"TensorRT conversion '{attempt_name}' failed: {str(e)}")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return None
        
        if args.quant == 'fp16':
            logger.info("Converting to FP16 precision")
            # Ensure model and input are in FP16
            model = model.half()
            input_tensor = input_tensor.half()
            
            # Verify FP16 conversion
            logger.info(f"Model dtype after FP16 conversion: {next(model.parameters()).dtype}")
            logger.info(f"Input tensor dtype after FP16 conversion: {input_tensor.dtype}")
            
            # Try multiple conversion approaches with fallbacks
            
            # First attempt - without strict type constraints
            trt_config_1 = {
                'max_batch_size': max_batch_size,
                'fp16_mode': True,
                'max_workspace_size': max_workspace_size,
                'strict_type_constraints': False
            }
            converted_model = attempt_conversion(model, input_tensor, trt_config_1, "FP16 - standard")
            
            # Second attempt - with smaller workspace
            if converted_model is None:
                logger.info("First attempt failed, trying with smaller workspace...")
                trt_config_2 = {
                    'max_batch_size': max_batch_size,
                    'fp16_mode': True,
                    'max_workspace_size': max_workspace_size // 2,
                    'strict_type_constraints': False
                }
                converted_model = attempt_conversion(model, input_tensor, trt_config_2, "FP16 - smaller workspace")
            
            # Third attempt - with strict type constraints
            if converted_model is None:
                logger.info("Second attempt failed, trying with strict type constraints...")
                trt_config_3 = {
                    'max_batch_size': max_batch_size,
                    'fp16_mode': True,
                    'max_workspace_size': max_workspace_size,
                    'strict_type_constraints': True
                }
                converted_model = attempt_conversion(model, input_tensor, trt_config_3, "FP16 - strict types")
            
            # Fourth attempt - with even smaller workspace and no strict constraints
            if converted_model is None:
                logger.info("Third attempt failed, trying with minimal settings...")
                trt_config_4 = {
                    'max_batch_size': 1,  # Minimal batch size
                    'fp16_mode': True,
                    'max_workspace_size': 1 << 26,  # 64MB
                    'strict_type_constraints': False
                }
                converted_model = attempt_conversion(model, input_tensor, trt_config_4, "FP16 - minimal")
            
            # Final check
            if converted_model is None:
                logger.error("All FP16 conversion attempts failed")
                raise RuntimeError("Failed to convert model to TensorRT with FP16 precision")
            else:
                model = converted_model
                logger.info("FP16 conversion completed successfully")
        elif args.quant == 'fp32':
            logger.info("Converting to FP32 precision")
            
            # Try multiple conversion approaches with fallbacks
            
            # First attempt - standard settings
            trt_config_1 = {
                'max_batch_size': max_batch_size,
                'fp16_mode': False,
                'max_workspace_size': max_workspace_size
            }
            converted_model = attempt_conversion(model, input_tensor, trt_config_1, "FP32 - standard")
            
            # Second attempt - with smaller workspace
            if converted_model is None:
                logger.info("First attempt failed, trying with smaller workspace...")
                trt_config_2 = {
                    'max_batch_size': max_batch_size,
                    'fp16_mode': False,
                    'max_workspace_size': max_workspace_size // 2
                }
                converted_model = attempt_conversion(model, input_tensor, trt_config_2, "FP32 - smaller workspace")
            
            # Third attempt - with minimal settings
            if converted_model is None:
                logger.info("Second attempt failed, trying with minimal settings...")
                trt_config_3 = {
                    'max_batch_size': 1,  # Minimal batch size
                    'fp16_mode': False,
                    'max_workspace_size': 1 << 26  # 64MB
                }
                converted_model = attempt_conversion(model, input_tensor, trt_config_3, "FP32 - minimal")
            
            # Final check
            if converted_model is None:
                logger.error("All FP32 conversion attempts failed")
                raise RuntimeError("Failed to convert model to TensorRT with FP32 precision")
            else:
                model = converted_model
                logger.info("FP32 conversion completed successfully")
        elif args.quant == 'int8':
            logger.info("Converting to INT8 precision")
            model = model.half()
            input
        else:
            raise ValueError(f"Unsupported quantization scheme: {args.quant}")
        # Verify the conversion was successful
        logger.info("Verifying TensorRT model with test inference...")
        try:
            with torch.no_grad():
                # Create a fresh input tensor to test with
                test_input = input_tensor.clone()
                output = model(test_input)
                logger.info(f"TensorRT conversion successful. Output shape: {output.shape}")
                logger.info(f"Output data type: {output.dtype}")
                logger.info(f"Output sample values: {output[0, 0:min(5, output.shape[1])].tolist()}")
        except Exception as e:
            logger.error(f"Error during TensorRT model verification: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"TensorRT model verification failed: {str(e)}")
        # Save the converted model
        output_dir = 'tensorrt_state_dicts'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(
            output_dir,
            f"{args.model}_{args.quant}_{args.output}_trt.pth" if args.output else f"{args.model}_{args.quant}_trt.pth"
        )
        
        logger.info(f"Saving TensorRT model to {output_path}")
        try:
            torch.cuda.synchronize()  # Ensure all operations are complete before saving
            torch.save(model.state_dict(), output_path)
            logger.info(f"TensorRT model successfully saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving TensorRT model: {str(e)}")
            raise
        
        return model
        
    except Exception as e:
        logger.error(f"Error during TensorRT conversion: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Try to clean up CUDA memory in case of error
        try:
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared after error")
        except:
            pass
        
        raise

def get_model_input_shape(model_name: str) -> Tuple[int, int, int, int]:
    """Get the input shape for the specified model."""
    if model_name == 'ernet':
        return (1, 3, 240, 240)
    elif model_name in ['squeeze-ernet', 'squeeze-redconv']:
        return (1, 3, 140, 140)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='TensorRT Model Conversion Script')
        parser.add_argument('--model', type=str, default='ernet', 
                            choices=['ernet', 'squeeze-ernet', 'squeeze-redconv'],
                            help='Model architecture to convert')
        parser.add_argument('--output', type=str, default=None, 
                            help='Suffix to append at the output file name')
        parser.add_argument('--weights', type=str, default=None, 
                            help='Path to pre-trained PyTorch weights (.pt) file')
        parser.add_argument('--quant', type=str, default='fp16', 
                            choices=['fp16', 'fp32', 'int8'],
                            help='Quantization scheme to use')
        args = parser.parse_args()
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available. Cannot build TensorRT engine!')
        
        # Log CUDA device info
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Set default weights if not provided
        if not args.weights:
            if args.model == 'ernet':
                args.weights = 'weights/ernet.pt'
            elif args.model == 'squeeze-ernet':
                args.weights = 'weights/Squeeze-ernet-92f1score.pt'
            elif args.model == 'squeeze-redconv':
                args.weights = 'weights/Squeeze-ernet-redconv92acc.pt'
        
        # Create input tensor
        input_shape = get_model_input_shape(args.model)
        logger.info(f"Creating input tensor with shape: {input_shape}")
        input_tensor = torch.randn(input_shape)
        
        # Load model
        device = torch.device('cuda')
        model = load_model(args.model, args.weights, device)
        
        # Convert to TensorRT
        trt_model = build_trt_model(model, input_tensor, args)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise