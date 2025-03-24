import os
import shutil
import time
from glob import glob
from pathlib import Path
import torch
from mmgp import offload
from IPython.display import display, HTML

# Define constants
SAVE_DIR = 'gradio_cache'  # Cache directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
ENABLE_T23D = True  # Enable text-to-3D
PROFILE = 5  # Memory profile for optimization
VERBOSE = 1  # Verbosity level
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output')  # New output directory

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions
def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/*.png')) if os.path.exists('./assets/example_images') else []

def get_example_txt_list():
    print('Loading example txt list ...')
    txt_file = './assets/example_prompts.txt'
    txt_list = []
    if os.path.exists(txt_file):
        try:
            with open(txt_file, encoding='utf-8') as f:
                for line in f:
                    txt_list.append(line.strip())
        except Exception as e:
            print(f"Failed to load example prompts: {e}")
    else:
        print(f"Example prompts file not found at {txt_file}. Using default prompts.")
        # Default prompts as fallback
        txt_list = [
            "A 3D model of a cute cat, white background",
            "A futuristic car with glowing lights",
            "A medieval castle on a hill"
        ]
    return txt_list

def gen_save_folder(max_size=60):
    os.makedirs(SAVE_DIR, exist_ok=True)
    exists = set(int(_) for _ in os.listdir(SAVE_DIR) if _.isdigit())
    cur_id = min(set(range(max_size)) - exists) if len(exists) < max_size else -1
    if os.path.exists(f"{SAVE_DIR}/{(cur_id + 1) % max_size}"):
        shutil.rmtree(f"{SAVE_DIR}/{(cur_id + 1) % max_size}")
        print(f"remove {SAVE_DIR}/{(cur_id + 1) % max_size} success !!!")
    save_folder = f"{SAVE_DIR}/{max(0, cur_id)}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"mkdir {save_folder} success !!!")
    return save_folder

def export_mesh(mesh, save_folder, textured=False):
    if textured:
        temp_path = os.path.join(save_folder, f'textured_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'textured_mesh_{int(time.time())}.glb')
    else:
        temp_path = os.path.join(save_folder, f'white_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'white_mesh_{int(time.time())}.glb')
    
    # Export to temporary location
    mesh.export(temp_path, include_normals=textured)
    # Copy to output directory with timestamp to avoid overwriting
    shutil.copy2(temp_path, output_path)
    return temp_path  # Return original path for compatibility

def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')

    template_path = os.path.join(CURRENT_DIR, template_name)
    if not os.path.exists(template_path):
        print(f"Template file {template_path} not found. Skipping HTML generation.")
        return None

    with open(template_path, 'r', encoding='utf-8') as f:
        template_html = f.read()
        obj_html = f"""
            <div class="column is-mobile is-centered">
                <model-viewer style="height: {height - 10}px; width: {width}px;" rotation-per-second="10deg" id="modelViewer"
                    src="{related_path}/" disable-tap 
                    environment-image="neutral" auto-rotate camera-target="0m 0m 0m" orientation="0deg 0deg 170deg" shadow-intensity=".9"
                    ar auto-rotate camera-controls>
                </model-viewer>
            </div>
            """

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(template_html.replace('<model-viewer>', obj_html))

    return output_html_path

def _gen_shape(caption, image, steps=50, guidance_scale=7.5, seed=1234, octree_resolution=256, remove_background=False):
    if caption: print('prompt is', caption)
    save_folder = gen_save_folder()
    stats = {}
    time_meta = {}
    start_time_0 = time.time()

    if image is None and ENABLE_T23D:
        start_time = time.time()
        try:
            image = t2i_worker(caption)
        except Exception as e:
            raise Exception(f"Text to 3D failed: {e}")
        time_meta['text2image'] = time.time() - start_time

    if image is None:
        raise Exception("No image provided and text-to-image is disabled or failed.")

    image.save(os.path.join(save_folder, 'input.png'))

    if remove_background or image.mode == "RGB":
        start_time = time.time()
        image = rmbg_worker(image.convert('RGB'))
        time_meta['rembg'] = time.time() - start_time

    image.save(os.path.join(save_folder, 'rembg.png'))

    start_time = time.time()
    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    mesh = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution
    )[0]

    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    time_meta['image_to_textured_3d'] = {'total': time.time() - start_time}
    time_meta['total'] = time.time() - start_time_0
    stats['time'] = time_meta
    return mesh, image, save_folder

def generate_shape_only(caption=None, image=None, steps=50, guidance_scale=7.5, seed=1234, octree_resolution=256, remove_background=False):
    mesh, image, save_folder = _gen_shape(caption, image, steps, guidance_scale, seed, octree_resolution, remove_background)
    path = export_mesh(mesh, save_folder, textured=False)
    html_path = build_model_viewer_html(save_folder, height=596, width=700, textured=False)
    return path, html_path

def generate_shape_and_texture(caption=None, image=None, steps=50, guidance_scale=7.5, seed=1234, octree_resolution=256, remove_background=False):
    mesh, image, save_folder = _gen_shape(caption, image, steps, guidance_scale, seed, octree_resolution, remove_background)
    path = export_mesh(mesh, save_folder, textured=False)
    html_path = build_model_viewer_html(save_folder, height=596, width=700, textured=False)
    textured_mesh = texgen_worker(mesh, image)
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    html_path_textured = build_model_viewer_html(save_folder, height=596, width=700, textured=True)
    return path, path_textured, html_path, html_path_textured

# Setup environment and models
os.makedirs(SAVE_DIR, exist_ok=True)
example_is = get_example_img_list()
example_ts = get_example_txt_list()
torch.set_default_device("cpu")

try:
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    HAS_TEXTUREGEN = True
except Exception as e:
    print(f"Failed to load texture generator: {e}")
    HAS_TEXTUREGEN = False

HAS_T2I = False
if ENABLE_T23D:
    try:
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        HAS_T2I = True
    except Exception as e:
        print(f"Failed to load text-to-image pipeline: {e}")

from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

rmbg_worker = BackgroundRemover()
i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', device="cpu", use_safetensors=True)

# Memory optimization
pipe = offload.extract_models("i23d_worker", i23d_worker)
if HAS_TEXTUREGEN:
    pipe.update(offload.extract_models("texgen_worker", texgen_worker))
    texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True
if HAS_T2I:
    pipe.update(offload.extract_models("t2i_worker", t2i_worker))

kwargs = {}
if PROFILE < 5:
    kwargs["pinnedMemory"] = "i23d_worker/model"
if PROFILE != 1 and PROFILE != 3:
    kwargs["budgets"] = {"*": 2200}

offload.profile(pipe, profile_no=PROFILE, verboseLevel=VERBOSE, **kwargs)

# Example usage in notebook
def run_example(caption=None, image_path=None, generate_texture=False):
    image = None
    if image_path:
        from PIL import Image
        image = Image.open(image_path).convert('RGBA')
    
    try:
        if generate_texture and HAS_TEXTUREGEN:
            path, path_textured, html_path, html_path_textured = generate_shape_and_texture(
                caption=caption,
                image=image,
                steps=30,
                guidance_scale=5.5,
                seed=1234,
                octree_resolution=256,
                remove_background=True
            )
            print(f"Generated mesh: {path}")
            print(f"Generated textured mesh: {path_textured}")
            print(f"GLB files exported to: {OUTPUT_DIR}")
            if html_path: display(HTML(f"<a href='file://{html_path}' target='_blank'>View Shape</a>"))
            if html_path_textured: display(HTML(f"<a href='file://{html_path_textured}' target='_blank'>View Textured Shape</a>"))
        else:
            path, html_path = generate_shape_only(
                caption=caption,
                image=image,
                steps=30,
                guidance_scale=5.5,
                seed=1234,
                octree_resolution=256,
                remove_background=True
            )
            print(f"Generated mesh: {path}")
            print(f"GLB file exported to: {OUTPUT_DIR}")
            if html_path: display(HTML(f"<a href='file://{html_path}' target='_blank'>View Shape</a>"))
    except Exception as e:
        print(f"Error: {e}")

# Run an example (modify inputs as needed)
run_example(caption="A 3D model of a cute cat, white background" if ENABLE_T23D else None,
            image_path=None,  # Replace with a path like './assets/example_images/example.png' if desired
            generate_texture=True)
