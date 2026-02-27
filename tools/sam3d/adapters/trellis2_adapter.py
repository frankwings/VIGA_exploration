
import os
import torch
import numpy as np
from PIL import Image
import trimesh
import json
import tempfile
from huggingface_hub import hf_hub_download

print("Adapting Trellis2... v3", flush=True)

# Define DummyBiRefNet globally
class DummyBiRefNet:
    def __init__(self, *args, **kwargs):
        print("Initialized DummyBiRefNet (Bypass Gated Model)")
        self.device = 'cpu'
    def to(self, device): self.device = device
    def cuda(self): self.device = 'cuda'
    def cpu(self): self.device = 'cpu'
    def eval(self): pass
    def __call__(self, image): return image

# Variables to hold classes if imports succeed
Trellis2ImageTo3DPipeline_Class = None
rembg_module = None
models_module = None
image_feature_extractor_module = None
samplers_module = None

try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import rembg
    from trellis2 import models
    from trellis2.pipelines import samplers
    from trellis2.modules import image_feature_extractor
    
    Trellis2ImageTo3DPipeline_Class = Trellis2ImageTo3DPipeline
    rembg_module = rembg
    models_module = models
    samplers_module = samplers
    image_feature_extractor_module = image_feature_extractor

    # Inject into rembg module so getattr(rembg, 'DummyBiRefNet') works
    if not hasattr(rembg, 'DummyBiRefNet'):
        setattr(rembg, 'DummyBiRefNet', DummyBiRefNet)
    
    print("DummyBiRefNet injected into trellis2.pipelines.rembg")

except ImportError as e:
    print(f"Warning: trellis2 package not found or import failed: {e}")
    import traceback
    traceback.print_exc()

# Define Custom Pipeline ONLY if base class is available
if Trellis2ImageTo3DPipeline_Class:
    class CustomTrellis2Pipeline(Trellis2ImageTo3DPipeline_Class):
        @classmethod
        def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "CustomTrellis2Pipeline":
            # Check if config_file is local path
            if os.path.exists(config_file):
                local_config = config_file
            else:
                local_config = hf_hub_download(path, config_file)

            with open(local_config, 'r') as f:
                args = json.load(f)['args']

            _models = {}
            for k, v in args['models'].items():
                if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                    continue
                try:
                    # Logic from base pipeline: try path/v, else v
                    full_path = f"{path}/{v}"
                    print(f"Loading sub-model: {k} from {full_path}")
                    _models[k] = models_module.from_pretrained(full_path)
                except Exception as e:
                    print(f"Failed to load {k} from {full_path}: {e}")
                    print(f"Fallback loading {k} from {v}")
                    # fallback
                    _models[k] = models_module.from_pretrained(v)

            new_pipeline = cls(_models)
            new_pipeline._pretrained_args = args
            
            pipeline = new_pipeline
            # Setup sampler and models as in Trellis2ImageTo3DPipeline.from_pretrained
            if 'sparse_structure_sampler' in args:
                pipeline.sparse_structure_sampler = getattr(samplers_module, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
                pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

            if 'shape_slat_sampler' in args:
                pipeline.shape_slat_sampler = getattr(samplers_module, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
                pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

            if 'tex_slat_sampler' in args:
                pipeline.tex_slat_sampler = getattr(samplers_module, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
                pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

            if 'shape_slat_normalization' in args:
                pipeline.shape_slat_normalization = args['shape_slat_normalization']
            if 'tex_slat_normalization' in args:
                pipeline.tex_slat_normalization = args['tex_slat_normalization']

            if 'image_cond_model' in args:
                pipeline.image_cond_model = getattr(image_feature_extractor_module, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
            
            if 'rembg_model' in args:
                pipeline.rembg_model = getattr(rembg_module, args['rembg_model']['name'])(**args['rembg_model']['args'])
            
            pipeline.low_vram = args.get('low_vram', True)
            pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
            pipeline.pbr_attr_layout = {
                'base_color': slice(0, 3),
                'metallic': slice(3, 4),
                'roughness': slice(4, 5),
                'alpha': slice(5, 6),
            }
            pipeline._device = 'cpu'

            return pipeline
else:
    print("CRITICAL: Trellis2 Base Pipeline not loaded. CustomTrellis2Pipeline not defined.")
    CustomTrellis2Pipeline = None

class Trellis2Inference:
    def __init__(self, config_path=None, compile=False, model_name="microsoft/TRELLIS.2-4B"):
        print(f"Loading TRELLIS.2 model: {model_name}...")
        
        if CustomTrellis2Pipeline is None:
            raise ImportError("Trellis2 Pipeline class missing. Check imports.")

        # Download and modify config to avoid gated rembg model
        fd = None
        try:
            config_file = hf_hub_download(repo_id=model_name, filename="pipeline.json")
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Modify config to use DummyBiRefNet
            if 'args' in config and 'rembg_model' in config['args']:
                print("Patching pipeline config to use DummyBiRefNet...")
                config['args']['rembg_model']['name'] = 'DummyBiRefNet'
                config['args']['rembg_model']['args'] = {} # No args needed
            
            # Save modified config to temp file
            fd, temp_config_path = tempfile.mkstemp(suffix='.json', text=True)
            with os.fdopen(fd, 'w') as f:
                json.dump(config, f)
            
            print(f"Using modified config at: {temp_config_path}")
            # Use Custom Pipeline
            self.pipeline = CustomTrellis2Pipeline.from_pretrained(model_name, config_file=temp_config_path)
            
        except Exception as e:
            print(f"Error initializing custom pipeline: {e}. Aborting.")
            import traceback
            traceback.print_exc()
            raise e
        finally:
            if fd:
                try: 
                    # Don't unlink yet if pipeline still needs it?
                    # Pipeline reads it in from_pretrained. Safe to unlink after return.
                    # But os.fdopen creates file object.
                    pass
                except: pass

        self.pipeline.cuda()
        print("TRELLIS.2 model loaded.")

    def __call__(self, image: np.ndarray, mask: np.ndarray = None, seed: int = 42):
        # Prepare Input
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        if mask is not None:
             if isinstance(mask, np.ndarray):
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
             else:
                mask_pil = mask
             image_pil = image_pil.convert("RGBA")
             image_pil.putalpha(mask_pil)
        
        print(f"Running TRELLIS.2 inference with seed {seed}...")
        outputs = self.pipeline.run(image_pil, seed=seed)
        
        mesh = outputs[0]
        
        # Helper to convert mesh items if needed
        vertices = mesh.vertices # usually torch tensor or numpy
        faces = mesh.faces
        
        if hasattr(vertices, 'cpu'): vertices = vertices.cpu().numpy()
        if hasattr(faces, 'cpu'): faces = faces.cpu().numpy()
        
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return {
            "glb": mesh_trimesh,
            "scale": torch.ones(1, 3, dtype=torch.float32).cuda(),
            "translation": torch.zeros(1, 3, dtype=torch.float32).cuda(),
            "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).cuda() 
        }

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3] 
    image = image.astype(np.uint8)
    return image
