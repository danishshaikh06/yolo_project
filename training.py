# Memory-Optimized YOLO Training Script for 4GB GPU
from ultralytics import YOLO
import torch
import gc
import os
from multiprocessing import freeze_support

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_optimal_settings(gpu_memory_gb):
    """Get optimal settings based on GPU memory"""
    if gpu_memory_gb <= 4:
        return {
            'batch_size': 2,  # Very small batch size for 4GB GPU
            'imgsz': 640,     # Smaller image size
            'workers': 2,     # Fewer workers
            'cache': False,   # Disable caching to save memory
        }
    elif gpu_memory_gb <= 6:
        return {
            'batch_size': 4,
            'imgsz': 640,
            'workers': 4,
            'cache': 'ram',
        }
    elif gpu_memory_gb <= 8:
        return {
            'batch_size': 8,
            'imgsz': 832,
            'workers': 6,
            'cache': 'ram',
        }
    else:
        return {
            'batch_size': 16,
            'imgsz': 832,
            'workers': 8,
            'cache': 'ram',
        }

def main():
    print("Starting Memory-Optimized YOLOv10x Training...")
    
    # Clear memory before starting
    clear_memory()
    
    # Your dataset path
    dataset_path = r"C:\Users\HITESH SHUKLA\Downloads\exam cheating.v1-examdataset.yolov8"
    
    # Check if data.yaml exists
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml_path):
        print(f"ERROR: data.yaml not found at {data_yaml_path}")
        print("Please check your dataset path.")
        return
    
    # Load model - Use smaller model for 4GB GPU
    print("Loading YOLOv10n (nano) model for better memory efficiency...")
    model = YOLO('yolov10n.pt')  # Using nano instead of x for memory efficiency
    
    # Get GPU memory info
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable memory-efficient settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
    else:
        gpu_memory = 0
        print("CUDA not available, using CPU")
    
    # Get optimal settings
    settings = get_optimal_settings(gpu_memory)
    
    print(f"Optimal settings for your GPU:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    # Training configuration (memory optimized)
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=200,
            batch=settings['batch_size'],
            imgsz=settings['imgsz'],
            device='0' if torch.cuda.is_available() else 'cpu',
            workers=settings['workers'],
            
            # Memory optimization settings
            close_mosaic=100,  # Disable mosaic early to save memory
            cache=settings['cache'],
            amp=True,  # Mixed precision training
            
            # Core training settings
            optimizer='SGD',  # SGD uses less memory than AdamW
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            patience=30,
            
            # Performance settings
            save=True,
            plots=True,
            val=True,
            
            # Reduced data augmentation to save memory
            hsv_h=0.01,
            hsv_s=0.2,
            hsv_v=0.2,
            degrees=3.0,
            translate=0.05,
            scale=0.2,
            shear=1.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.3,
            mosaic=0.5,  # Reduced mosaic probability
            mixup=0.0,   # Disable mixup to save memory
            copy_paste=0.0,
            
            # Additional memory saving options
            save_period=50,  # Save less frequently
            verbose=True,
        )
        
        print("Training completed successfully!")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory Error: {e}")
        print("\nTrying with even smaller batch size...")
        clear_memory()
        
        # Emergency fallback settings
        try:
            results = model.train(
                data=data_yaml_path,
                epochs=200,
                batch=1,  # Minimum batch size
                imgsz=480,  # Even smaller image size
                device='0' if torch.cuda.is_available() else 'cpu',
                workers=1,
                cache=False,
                amp=False,  # Disable mixed precision
                optimizer='SGD',
                lr0=0.01,
                patience=30,
                save=True,
                plots=False,  # Disable plots to save memory
                val=True,
                mosaic=0.0,  # Disable mosaic
                mixup=0.0,
                copy_paste=0.0,
                verbose=True,
            )
            print("Training completed with emergency settings!")
            
        except Exception as e2:
            print(f"Training failed even with minimum settings: {e2}")
            print("Consider using CPU training or upgrading your GPU.")
            return
    
    except Exception as e:
        print(f"Training error: {e}")
        return
    
    # Clear memory before validation
    clear_memory()
    
    # Validate the model
    print("Starting validation...")
    try:
        val_results = model.val()
        
        # Print results
        print("=== VALIDATION RESULTS ===")
        if hasattr(val_results, 'box'):
            print(f"mAP50: {val_results.box.map50:.4f}")
            print(f"mAP50-95: {val_results.box.map:.4f}")
            print(f"Precision: {val_results.box.mp:.4f}")
            print(f"Recall: {val_results.box.mr:.4f}")
    
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Export model for deployment
    print("Exporting model to ONNX...")
    try:
        clear_memory()
        model.export(format='onnx', imgsz=settings['imgsz'])
        print("Model exported successfully!")
    except Exception as e:
        print(f"Export failed: {e}")
    
    print("=" * 50)
    print("TRAINING COMPLETED!")
    print("Check 'runs/detect/train' folder for results.")
    print("Best model: runs/detect/train/weights/best.pt")
    print("Last checkpoint: runs/detect/train/weights/last.pt")
    print("=" * 50)

def resume_training():
    """Function to resume training from last checkpoint"""
    last_checkpoint = "runs/detect/train/weights/last.pt"
    
    if not os.path.exists(last_checkpoint):
        print("No checkpoint found to resume from!")
        return
    
    print(f"Resuming training from: {last_checkpoint}")
    
    # Clear memory before loading
    clear_memory()
    
    # Load from checkpoint
    model = YOLO(last_checkpoint)
    
    # Resume training with memory optimization
    try:
        results = model.train(resume=True)
        print("Resume training completed!")
    except torch.cuda.OutOfMemoryError:
        print("Out of memory during resume. Try reducing batch size in the checkpoint.")

def test_model():
    """Test the trained model"""
    model_path = "runs/detect/train/weights/best.pt"
    
    if not os.path.exists(model_path):
        print("No trained model found!")
        return
    
    print("Testing trained model...")
    model = YOLO(model_path)
    
    # Test on a sample image (replace with your test image path)
    test_image = input("Enter path to test image (or press Enter to skip): ").strip()
    
    if test_image and os.path.exists(test_image):
        results = model(test_image)
        results[0].show()
        print(f"Detection results saved!")
    else:
        print("Test skipped - no valid image path provided.")

if __name__ == '__main__':
    freeze_support()
    
    # System check
    print("=== SYSTEM CHECK ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory <= 4:
            print("⚠️  WARNING: Low GPU memory detected!")
            print("   Using memory-optimized settings.")
    print("=" * 30)
    
    # Choose training mode
    print("Training Options:")
    print("1. Start new training (memory optimized)")
    print("2. Resume previous training")
    print("3. Test existing model")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "2":
        resume_training()
    elif choice == "3":
        test_model()
    else:
        main()
    
    print("\n=== QUICK TEST COMMANDS ===")
    print("To test your trained model:")
    print("from ultralytics import YOLO")
    print("model = YOLO('runs/detect/train/weights/best.pt')")
    print("results = model('path/to/test/image.jpg')")
    print("results[0].show()")
    
    print("\n=== MEMORY TIPS ===")
    print("If you still get memory errors:")
    print("1. Use batch_size=1")
    print("2. Use imgsz=480 or even 320")
    print("3. Use YOLOv10n instead of YOLOv10x")
    print("4. Set cache=False")
    print("5. Consider training on CPU (slower but works)")