#!/usr/bin/env python3
import os
import json
import atexit
import shutil
import time
import glob
import torch
from pathlib import Path
import argparse
from PIL import Image
from mmgp import offload
from Voxelization import load_block_colors, ModelViewer
from aieditor import analyze_images_and_voxel
from openai import OpenAI
from txt2sc import text_to_schematic

# 常量定义 (Constants)
BLOCK_COLORS_PATH = 'blockids.json'
API_KEY_PATH = 'api_key.json'
SAVE_DIR = 'cache'  # 缓存目录 (Cache directory)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output')  # 输出目录 (Output directory)
PROFILE = 5  # 内存优化配置 (Memory optimization profile)
VERBOSE = 1  # 详细程度 (Verbosity level)

# 创建输出目录 (Create output directory)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# 全局加载 blockids.json (Globally load blockids.json)
block_colors = load_block_colors(BLOCK_COLORS_PATH)
viewer = ModelViewer(block_colors)

def cleanup_image_files():
    """清理当前目录下的临时图片文件 (Clean up temporary image files in the current directory)"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    for filename in os.listdir(CURRENT_DIR):
        if filename.lower().endswith(image_extensions):
            try:
                file_path = os.path.join(CURRENT_DIR, filename)
                os.remove(file_path)
                print(f"已删除临时图片文件：{filename} (Temporary image file deleted: {filename})")
            except Exception as e:
                print(f"删除文件 {filename} 时出错：{e} (Error deleting file {filename}: {e})")

def cleanup_glb_files():
    """清理output目录下的.glb模型文件 (Clean up .glb model files in the output directory)"""
    glb_files = glob.glob(os.path.join(OUTPUT_DIR, "*.glb"))
    for glb_file in glb_files:
        try:
            os.remove(glb_file)
            print(f"已删除GLB模型文件：{glb_file} (GLB model file deleted: {glb_file})")
        except Exception as e:
            print(f"删除GLB文件 {glb_file} 时出错：{e} (Error deleting GLB file {glb_file}: {e})")

# 注册程序退出时的清理函数 (Register cleanup function on program exit)
atexit.register(cleanup_image_files)

def save_api_key(api_key):
    """保存API密钥到本地文件 (Save API key to local file)"""
    try:
        with open(API_KEY_PATH, 'w') as f:
            json.dump({'api_key': api_key}, f)
        print(f"API密钥已保存 (API key saved)")
    except Exception as e:
        print(f"保存API密钥时出错：{e} (Error saving API key: {e})")

def load_api_key():
    """从本地文件加载API密钥 (Load API key from local file)"""
    try:
        if os.path.exists(API_KEY_PATH):
            with open(API_KEY_PATH, 'r') as f:
                data = json.load(f)
                return data.get('api_key', '')
    except Exception as e:
        print(f"加载API密钥时出错：{e} (Error loading API key: {e})")
    return ''

def verify_api_key(api_key):
    """验证Gemini API密钥是否有效 (Verify if Gemini API key is valid)"""
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/"
        )
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            max_tokens=10
        )
        print(f"API密钥验证成功 (API key verification successful)")
        return True
    except Exception as e:
        print(f"Gemini API密钥验证失败：{e} (Gemini API key verification failed: {e})")
        return False

def gen_save_folder(max_size=60):
    """生成保存文件夹 (Generate save folder)"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    exists = set(int(_) for _ in os.listdir(SAVE_DIR) if _.isdigit())
    cur_id = min(set(range(max_size)) - exists) if len(exists) < max_size else -1
    if os.path.exists(f"{SAVE_DIR}/{(cur_id + 1) % max_size}"):
        shutil.rmtree(f"{SAVE_DIR}/{(cur_id + 1) % max_size}")
        print(f"移除 {SAVE_DIR}/{(cur_id + 1) % max_size} 成功 (Removed {SAVE_DIR}/{(cur_id + 1) % max_size} successfully)")
    save_folder = f"{SAVE_DIR}/{max(0, cur_id)}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"创建 {save_folder} 成功 (Created {save_folder} successfully)")
    return save_folder

def export_mesh(mesh, save_folder, textured=False):
    """导出模型文件 (Export model file)"""
    if textured:
        temp_path = os.path.join(save_folder, f'textured_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'textured_mesh_{int(time.time())}.glb')
    else:
        temp_path = os.path.join(save_folder, f'white_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'white_mesh_{int(time.time())}.glb')
    
    mesh.export(temp_path, include_normals=textured)
    shutil.copy2(temp_path, output_path)
    print(f"模型导出到：{output_path} (Model exported to: {output_path})")
    return output_path

def setup_hunyuan_model():
    """初始化Hunyuan模型 (Initialize Hunyuan model)"""
    print(f"正在加载Hunyuan模型... (Loading Hunyuan model...)")
    
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        has_texturegen = True
        print(f"纹理生成器加载成功 (Texture generator loaded successfully)")
    except Exception as e:
        print(f"加载纹理生成器失败: {e} (Failed to load texture generator: {e})")
        texgen_worker = None
        has_texturegen = False

    try:
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        has_t2i = True
        print(f"文本到图像模型加载成功 (Text-to-image model loaded successfully)")
    except Exception as e:
        print(f"加载文本到图像模型失败: {e} (Failed to load text-to-image model: {e})")
        t2i_worker = None
        has_t2i = False

    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', device="cpu", use_safetensors=True)

    pipe = offload.extract_models("i23d_worker", i23d_worker)
    if has_texturegen:
        pipe.update(offload.extract_models("texgen_worker", texgen_worker))
        texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True
    if has_t2i:
        pipe.update(offload.extract_models("t2i_worker", t2i_worker))

    kwargs = {}
    if PROFILE < 5:
        kwargs["pinnedMemory"] = "i23d_worker/model"
    if PROFILE != 1 and PROFILE != 3:
        kwargs["budgets"] = {"*": 2200}

    offload.profile(pipe, profile_no=PROFILE, verboseLevel=VERBOSE, **kwargs)
    
    return t2i_worker, i23d_worker, texgen_worker, rmbg_worker, FloaterRemover, DegenerateFaceRemover, FaceReducer

def generate_3d_model(prompt, seed=None, t2i_worker=None, i23d_worker=None, texgen_worker=None, 
                     rmbg_worker=None, FloaterRemover=None, DegenerateFaceRemover=None, FaceReducer=None):
    """生成3D模型 (Generate 3D model)"""
    print(f"正在生成3D模型，提示词: {prompt} (Generating 3D model, prompt: {prompt})")
    
    if seed is None or seed == "":
        seed = int(time.time()) % 10000
        print(f"使用随机种子: {seed} (Using random seed: {seed})")
    else:
        seed = int(seed)
        print(f"使用指定种子: {seed} (Using specified seed: {seed})")
    
    save_folder = gen_save_folder()
    generator = torch.Generator().manual_seed(seed)
    
    if t2i_worker:
        print(f"正在从文本生成图像... (Generating image from text...)")
        try:
            image = t2i_worker(prompt)
            input_image_path = os.path.join(CURRENT_DIR, f"input_{int(time.time())}.png")
            image.save(input_image_path)
            print(f"生成的参考图像已保存到: {input_image_path} (Generated reference image saved to: {input_image_path})")
            image.save(os.path.join(save_folder, 'input.png'))
        except Exception as e:
            print(f"文本生成图像失败: {e} (Text-to-image generation failed: {e})")
            return None
    else:
        print(f"文本到图像模型未加载，无法生成图像 (Text-to-image model not loaded, cannot generate image)")
        return None
    
    print(f"正在移除图像背景... (Removing image background...)")
    image = rmbg_worker(image.convert('RGB'))
    image.save(os.path.join(save_folder, 'rembg.png'))
    
    print(f"正在生成3D模型... (Generating 3D model...)")
    mesh = i23d_worker(
        image=image,
        num_inference_steps=30,
        guidance_scale=5.5,
        generator=generator,
        octree_resolution=256
    )[0]
    
    print(f"正在优化3D模型... (Optimizing 3D model...)")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    
    if texgen_worker:
        print(f"正在生成纹理... (Generating texture...)")
        textured_mesh = texgen_worker(mesh, image)
        output_path = export_mesh(textured_mesh, save_folder, textured=True)
        print(f"带纹理的3D模型已保存到: {output_path} (Textured 3D model saved to: {output_path})")
        return output_path
    else:
        output_path = export_mesh(mesh, save_folder, textured=False)
        print(f"3D模型已保存到: {output_path} (3D model saved to: {output_path})")
        return output_path

def process_model(glb_file):
    """处理GLB文件，进行体素化 (Process GLB file for voxelization)"""
    print(f"正在处理模型: {glb_file} (Processing model: {glb_file})")
    
    output_dir = os.getcwd()
    viewer.view_model(glb_file, output_dir)
    
    ply_path = os.path.join(output_dir, "voxel_model_32x32x32.ply")
    txt_path = os.path.join(output_dir, "voxel_model.txt")
    
    if not os.path.exists(ply_path):
        print(f"体素化模型生成失败！(Voxel model generation failed!)")
        return None, None
    if not os.path.exists(txt_path):
        print(f"体素文本生成失败！(Voxel text generation failed!)")
        return None, None
    
    print(f"模型已体素化，PLY文件: {ply_path}, TXT文件: {txt_path} (Model voxelized, PLY file: {ply_path}, TXT file: {txt_path})")
    return ply_path, txt_path

def analyze_images_and_voxel_with_key(api_key):
    """使用API密钥调用analyze_images_and_voxel函数 (Call analyze_images_and_voxel with API key)"""
    if not api_key or not api_key.strip():
        print(f"请提供有效的Gemini API密钥！(Please provide a valid Gemini API key!)")
        return None
    
    from aieditor import client
    client.api_key = api_key
    client.base_url = "https://generativelanguage.googleapis.com/v1beta/"
    
    try:
        print(f"正在进行AI分析... (Performing AI analysis...)")
        original_image_filenames = [f for f in os.listdir(".") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        filtered_image_filenames = [f for f in original_image_filenames if not f.startswith('extracted')]
        if len(original_image_filenames) > len(filtered_image_filenames):
            print(f"已过滤掉以下以 'extracted' 开头的图片: {', '.join(set(original_image_filenames) - set(filtered_image_filenames))} "
                  f"(Filtered out the following images starting with 'extracted': {', '.join(set(original_image_filenames) - set(filtered_image_filenames))})")
        
        result = analyze_images_and_voxel(".")
        if isinstance(result, str) and (result.startswith("错误") or result.startswith("处理错误")):
            print(f"AI分析过程出错：{result} (AI analysis process failed: {result})")
            return None
        
        working_file = os.path.join(os.getcwd(), "working.txt")
        if not os.path.exists(working_file):
            print(f"配置文件生成失败！(Configuration file generation failed!)")
            return None

        # 将 working.txt 同时保存到 output 文件夹（如果有同名则覆盖）
        output_working_file = os.path.join(OUTPUT_DIR, "working.txt")
        try:
            shutil.copy2(working_file, output_working_file)
            print(f"工作文件也已保存到output文件夹：{output_working_file} (Working file also saved to output folder: {output_working_file})")
        except Exception as e:
            print(f"保存工作文件到output文件夹失败：{e} (Failed to save working file to output folder: {e})")
        
        print(f"AI分析完成，配置文件: {working_file} (AI analysis completed, configuration file: {working_file})")
        return output_working_file
    except Exception as e:
        print(f"AI分析过程出错：{str(e)} (AI analysis process failed: {str(e)})")
        return None

def convert_to_schematic(working_file):
    """将working.txt转换为schematic格式 (Convert working.txt to schematic format)"""
    try:
        if not working_file:
            print(f"请先生成配置文件！(Please generate configuration file first!)")
            return None
        
        print(f"正在转换为Schematic: {working_file} (Converting to Schematic: {working_file})")
        output_file = os.path.join(OUTPUT_DIR, f"output_{int(time.time())}.schematic")
        
        result_path = text_to_schematic(working_file, output_file)
        print(f"转换完成，Schematic文件: {result_path} (Conversion completed, Schematic file: {result_path})")
        return result_path
    except Exception as e:
        print(f"转换失败：{str(e)} (Conversion failed: {str(e)})")
        return None

def check_existing_models():
    """检查output文件夹中是否已有.glb模型文件 (Check if there are existing .glb model files in output folder)"""
    glb_files = glob.glob(os.path.join(OUTPUT_DIR, "*.glb"))
    if glb_files:
        print(f"发现现有模型文件：{glb_files[0]}，跳过Hunyuan模型生成 (Found existing model file: {glb_files[0]}, skipping Hunyuan model generation)")
        return glb_files[0]
    return None

def main(args):
    """主函数 (Main function)"""
    print(f"==========================================\n"
          f"Minecraft 3D模型生成工具 (Minecraft 3D Model Generation Tool)\n"
          f"==========================================")
    
    prompt = args.prompt
    seed = args.seed
    api_key = args.key if args.key else load_api_key()

    if not prompt or not prompt.strip():
        print(f"错误：提示词不能为空 (Error: Prompt cannot be empty)")
        return

    api_valid = False
    while not api_valid:
        if not api_key:
            print(f"错误：未提供Gemini API密钥，且本地未找到有效密钥 (Error: No Gemini API key provided, and no valid key found locally)")
            return
        
        if verify_api_key(api_key):
            api_valid = True
            save_api_key(api_key)
        else:
            print(f"错误：Gemini API密钥无效 (Error: Gemini API key is invalid)")
            return
    
    glb_file = check_existing_models()
    
    if not glb_file:
        t2i_worker, i23d_worker, texgen_worker, rmbg_worker, FloaterRemover, DegenerateFaceRemover, FaceReducer = setup_hunyuan_model()
        glb_file = generate_3d_model(
            prompt, 
            seed, 
            t2i_worker, 
            i23d_worker, 
            texgen_worker, 
            rmbg_worker, 
            FloaterRemover, 
            DegenerateFaceRemover, 
            FaceReducer
        )
        
        if not glb_file:
            print(f"3D模型生成失败 (3D model generation failed)")
            return
    
    print(f"开始体素化处理... (Starting voxelization process...)")
    ply_file, txt_file = process_model(glb_file)
    
    if not ply_file or not txt_file:
        print(f"体素化失败 (Voxelization failed)")
        return
    
    print(f"开始AI颜色映射... (Starting AI color mapping...)")
    working_file = analyze_images_and_voxel_with_key(api_key)
    
    if not working_file:
        print(f"AI颜色映射失败 (AI color mapping failed)")
        return
    
    print(f"转换为Schematic... (Converting to Schematic...)")
    schematic_file = convert_to_schematic(working_file)
    
    if not schematic_file:
        print(f"Schematic转换失败 (Schematic conversion failed)")
        return
    
    # 在成功生成schematic后清理.glb文件 (Clean up .glb files after successful schematic generation)
    cleanup_glb_files()
    
    print(f"==========================================\n"
          f"处理完成！(Processing completed!)\n"
          f"Schematic文件已保存到: {schematic_file} (Schematic file saved to: {schematic_file})\n"
          f"==========================================")
    
    cleanup_image_files()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft 3D模型生成工具 (Minecraft 3D Model Generation Tool)")
    parser.add_argument("-prompt", type=str, required=True, help="生成3D模型的提示词 (Prompt for generating 3D model)")
    parser.add_argument("-seed", type=str, default="", help="随机种子（可选，留空则随机生成）(Random seed (optional, leave empty for random generation))")
    parser.add_argument("-key", type=str, default="", help="Gemini API密钥（可选，若未提供则尝试加载本地密钥）(Gemini API key (optional, attempts to load local key if not provided))")
    
    args = parser.parse_args()
    main(args)
