import os  
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
from transformers.utils import logging as hf_logging
import time  
import gc
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import get_balanced_memory

hf_logging.set_verbosity_info()  
hf_logging.get_logger("transformers").setLevel(hf_logging.INFO)  

def get_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return []
    
    gpu_info = []
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs:")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        gc.collect()
        
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        cached = torch.cuda.memory_reserved(i) / (1024**3)
        free = gpu_memory - allocated - cached
        
        gpu_info.append({
            'id': i,
            'name': gpu_name,
            'total': gpu_memory,
            'allocated': allocated,
            'cached': cached,
            'free': free
        })
        
        print(f"  GPU {i}: {gpu_name}")
        print(f"    Total: {gpu_memory:.2f} GB")
        print(f"    Allocated: {allocated:.2f} GB")
        print(f"    Cached: {cached:.2f} GB") 
        print(f"    Free: {free:.2f} GB")
        print()
    
    return gpu_info

def set_multi_gpu(gpu_ids=None):
    if gpu_ids is not None:
        gpu_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        print(f"Using GPUs: {gpu_ids}")
    else:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Using all available GPUs: {list(range(num_gpus))}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        gc.collect()

def create_device_map(model_path, gpu_ids=None, max_memory_per_gpu=None):
    if not torch.cuda.is_available():
        return "cpu"
    
    try:
        if gpu_ids:
            available_devices = [f"cuda:{i}" for i in gpu_ids]
        else:
            available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        
        print(f"Creating device map for: {available_devices}")
        
        if max_memory_per_gpu:
            max_memory = {}
            for device in available_devices:
                max_memory[device] = f"{max_memory_per_gpu}GB"
        else:
            max_memory = get_balanced_memory(
                model_name=model_path,
                available_devices=available_devices,
                dtype=torch.bfloat16
            )
        
        print("Memory allocation per device:")
        for device, memory in max_memory.items():
            print(f"  {device}: {memory}")
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        
        device_map = infer_auto_device_map(
            model, 
            max_memory=max_memory,
            dtype=torch.bfloat16
        )
        
        print("Auto-generated device map:")
        for layer, device in device_map.items():
            print(f"  {layer}: {device}")
        
        return device_map
        
    except Exception as e:
        print(f"Device map creation failed: {e}")
        print("Falling back to single GPU mode")
        return "auto"

def download_model(model_name, model_specific_path, hf_token=None):  
    print(f"Target save path: {model_specific_path}")
    os.makedirs(model_specific_path, exist_ok=True)  
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    print(f"Downloading tokenizer for {model_name}...")  
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token,
            trust_remote_code=True,
            cache_dir=os.path.join(model_specific_path, "tokenizer_cache")
        )  
        tokenizer.save_pretrained(model_specific_path)  
        print(f"Tokenizer saved to {model_specific_path}")  
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        raise

    print(f"Downloading model {model_name}...")  
    try:
        model = AutoModelForCausalLM.from_pretrained(  
            model_name,  
            token=hf_token,
            torch_dtype=torch.bfloat16,  
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=os.path.join(model_specific_path, "model_cache")
        )  
        model.save_pretrained(model_specific_path)  
        print(f"Model saved to {model_specific_path}")  
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise
        
    return tokenizer, model  

def load_model_multi_gpu(model_specific_path, gpu_ids=None, max_memory_per_gpu=None, use_8bit=False, use_4bit=False):
    print(f"Loading model from: {model_specific_path}")
    
    set_multi_gpu(gpu_ids)
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_specific_path, trust_remote_code=True)
        print("Tokenizer loaded.")
        
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_specific_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            return tokenizer, model, "cpu"
        
        device_map = create_device_map(model_specific_path, gpu_ids, max_memory_per_gpu)
        
        print("Loading model with multi-GPU support...")
        
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
            "device_map": device_map
        }
        
        if use_8bit:
            print("Using 8-bit quantization...")
            model_kwargs["load_in_8bit"] = True
        elif use_4bit:
            print("Using 4-bit quantization...")
            model_kwargs["load_in_4bit"] = True
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_specific_path,
                **model_kwargs
            )
            
            print("Model loaded successfully across multiple GPUs!")
            
            if hasattr(model, 'hf_device_map'):
                print("\nModel layer distribution:")
                device_count = {}
                for layer, device in model.hf_device_map.items():
                    device_count[device] = device_count.get(device, 0) + 1
                
                for device, count in device_count.items():
                    print(f"  {device}: {count} layers")
            
            if torch.cuda.is_available():
                print("\nCurrent GPU memory usage:")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    print(f"  GPU {i}: {allocated:.2f} GB (allocated) + {cached:.2f} GB (cached)")
            
            return tokenizer, model, device_map
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"Multi-GPU loading failed, out of memory: {e}")
            print("Try using more aggressive config or reduce GPU count...")
            raise
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def test_model_multi_gpu(tokenizer, model, device_info):
    if not tokenizer or not model:
        print("Model not loaded, skipping test")
        return

    print("\n===== Starting Multi-GPU Model Test =====")
    
    test_prompts = [
        "Explain the basic principles of quantum computing",
        "Write a Python function to calculate Fibonacci sequence",
        "Compare differences between classical and quantum computers"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n----- Test Case {i+1} -----")
        print(f"Input: {prompt}")
        
        start_time = time.time()
        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            if torch.cuda.is_available() and hasattr(model, 'hf_device_map'):
                first_device = None
                for layer_name, device in model.hf_device_map.items():
                    if 'embed' in layer_name.lower() or layer_name == '0':
                        first_device = device
                        break
                
                if first_device and first_device != 'cpu':
                    inputs = inputs.to(first_device)
                elif torch.cuda.is_available():
                    inputs = inputs.to('cuda:0')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            elapsed_time = time.time() - start_time
            
            print(f"Output: {response}")
            print(f"Generation time: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-GPU model loader and tester')
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-4-reasoning-plus',
                       help='Model name from Hugging Face (default: microsoft/Phi-4-reasoning-plus)')
    parser.add_argument('--base_model_path', type=str, default='./models',
                       help='Base path to save models (default: ./models)')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='Hugging Face API token for accessing gated models')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None, 
                       help='Specify GPU IDs to use (e.g., --gpu_ids 0 1 2)')
    parser.add_argument('--max_memory_per_gpu', type=int, default=None,
                       help='Maximum memory per GPU in GB (e.g., --max_memory_per_gpu 20)')
    parser.add_argument('--use_8bit', action='store_true', help='Use 8-bit quantization')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--download_only', action='store_true', help='Only download model')
    parser.add_argument('--show_gpu_info', action='store_true', help='Only show GPU information')
    
    args = parser.parse_args()
    
    # Setup paths based on arguments
    model_save_subdir = args.model_name.replace("/", "_")
    model_specific_path = os.path.join(args.base_model_path, model_save_subdir)
    
    print("=== GPU Information ===")
    gpu_info = get_gpu_info()
    
    if args.show_gpu_info:
        return
    
    config_file = os.path.join(model_specific_path, "config.json")
    
    if os.path.exists(config_file):
        print(f"Model exists at: {model_specific_path}")
        
        if not args.download_only:
            try:
                tokenizer, model, device_info = load_model_multi_gpu(
                    model_specific_path=model_specific_path,
                    gpu_ids=args.gpu_ids,
                    max_memory_per_gpu=args.max_memory_per_gpu,
                    use_8bit=args.use_8bit,
                    use_4bit=args.use_4bit
                )
                test_model_multi_gpu(tokenizer, model, device_info)
            except Exception as e:
                print(f"Loading or testing failed: {e}")
    else:
        print("Model does not exist, starting download...")
        try:
            download_model(
                model_name=args.model_name,
                model_specific_path=model_specific_path,
                hf_token=args.hf_token
            )
            
            if not args.download_only:
                tokenizer, model, device_info = load_model_multi_gpu(
                    model_specific_path=model_specific_path,
                    gpu_ids=args.gpu_ids,
                    max_memory_per_gpu=args.max_memory_per_gpu,
                    use_8bit=args.use_8bit,
                    use_4bit=args.use_4bit
                )
                test_model_multi_gpu(tokenizer, model, device_info)
        except Exception as e:
            print(f"Error during download or loading: {e}")
    
    print("\n===== Processing Complete =====")

if __name__ == "__main__":
    main()
