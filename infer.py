#!/usr/bin/env python3
import os
import json
import atexit
import shutil
import time
import glob
import torch
import re
from pathlib import Path
import argparse
from PIL import Image
from mmgp import offload
from Voxelization import load_block_colors, ModelViewer
from aieditor import analyze_images_and_voxel_with_key, make_openrouter_api_call
import nbt.nbt

# 检查是否在 Notebook 环境中运行
try:
    from IPython import get_ipython
    IN_NOTEBOOK = 'ipykernel' in str(get_ipython())
except (ImportError, AttributeError):
    IN_NOTEBOOK = False

# 常量定义
BLOCK_COLORS_PATH = 'blockids.json'
API_KEY_PATH = 'api_key.json'
SAVE_DIR = 'cache'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output')
PROFILE = 5
VERBOSE = 1

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# 全局加载 blockids.json
block_colors = load_block_colors(BLOCK_COLORS_PATH)
viewer = ModelViewer(block_colors)

# 全局模型变量
t2i_worker = None
i23d_worker = None
texgen_worker = None
rmbg_worker = None
FloaterRemover = None
DegenerateFaceRemover = None
FaceReducer = None

def cleanup_image_files():
    """清理当前目录下的临时图片文件"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    for filename in os.listdir(CURRENT_DIR):
        if filename.lower().endswith(image_extensions):
            try:
                file_path = os.path.join(CURRENT_DIR, filename)
                os.remove(file_path)
                print(f"已删除临时图片文件：{filename}")
            except Exception as e:
                print(f"删除文件 {filename} 时出错：{e}")

def cleanup_glb_files():
    """清理output目录下的.glb模型文件"""
    glb_files = glob.glob(os.path.join(OUTPUT_DIR, "*.glb"))
    for glb_file in glb_files:
        try:
            os.remove(glb_file)
            print(f"已删除GLB模型文件：{glb_file}")
        except Exception as e:
            print(f"删除GLB文件 {glb_file} 时出错：{e}")

# 注册程序退出时的清理函数
atexit.register(cleanup_image_files)

def save_api_key(api_key):
    """保存API密钥到本地文件"""
    try:
        with open(API_KEY_PATH, 'w') as f:
            json.dump({'api_key': api_key}, f)
        print(f"API密钥已保存")
    except Exception as e:
        print(f"保存API密钥时出错：{e}")

def load_api_key():
    """从本地文件加载API密钥"""
    try:
        if os.path.exists(API_KEY_PATH):
            with open(API_KEY_PATH, 'r') as f:
                data = json.load(f)
                return data.get('api_key', '')
    except Exception as e:
        print(f"加载API密钥时出错：{e}")
    return ''

def gen_save_folder(max_size=60):
    """生成保存文件夹"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    exists = set(int(_) for _ in os.listdir(SAVE_DIR) if _.isdigit())
    cur_id = min(set(range(max_size)) - exists) if len(exists) < max_size else -1
    if os.path.exists(f"{SAVE_DIR}/{(cur_id + 1) % max_size}"):
        shutil.rmtree(f"{SAVE_DIR}/{(cur_id + 1) % max_size}")
        print(f"移除 {SAVE_DIR}/{(cur_id + 1) % max_size} 成功")
    save_folder = f"{SAVE_DIR}/{max(0, cur_id)}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"创建 {save_folder} 成功")
    return save_folder

def export_mesh(mesh, save_folder, textured=False):
    """导出模型文件"""
    if textured:
        temp_path = os.path.join(save_folder, f'textured_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'textured_mesh_{int(time.time())}.glb')
    else:
        temp_path = os.path.join(save_folder, f'white_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'white_mesh_{int(time.time())}.glb')
    
    mesh.export(temp_path, include_normals=textured)
    shutil.copy2(temp_path, output_path)
    print(f"模型导出到：{output_path}")
    return output_path

def setup_hunyuan_model():
    """初始化Hunyuan模型"""
    global t2i_worker, i23d_worker, texgen_worker, rmbg_worker, FloaterRemover, DegenerateFaceRemover, FaceReducer
    print(f"正在加载Hunyuan模型...")
    
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        print(f"纹理生成器加载成功")
    except Exception as e:
        print(f"加载纹理生成器失败: {e}")
        texgen_worker = None

    try:
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        print(f"文本到图像模型加载成功")
    except Exception as e:
        print(f"加载文本到图像模型失败: {e}")
        t2i_worker = None

    from hy3dgen.shapegen import FaceReducer as FR, FloaterRemover as FLR, DegenerateFaceRemover as DFR
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', device="cpu", use_safetensors=True)
    FloaterRemover = FLR
    DegenerateFaceRemover = DFR
    FaceReducer = FR

    pipe = offload.extract_models("i23d_worker", i23d_worker)
    if texgen_worker:
        pipe.update(offload.extract_models("texgen_worker", texgen_worker))
        texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True
    if t2i_worker:
        pipe.update(offload.extract_models("t2i_worker", t2i_worker))

    kwargs = {}
    if PROFILE < 5:
        kwargs["pinnedMemory"] = "i23d_worker/model"
    if PROFILE != 1 and PROFILE != 3:
        kwargs["budgets"] = {"*": 2200}

    offload.profile(pipe, profile_no=PROFILE, verboseLevel=VERBOSE, **kwargs)
    print("模型加载完成")

def generate_3d_model(prompt, seed=None):
    """生成3D模型，使用全局模型实例"""
    print(f"正在生成3D模型，提示词: {prompt}")
    
    if seed is None or seed == "":
        seed = int(time.time()) % 10000
        print(f"使用随机种子: {seed}")
    else:
        seed = int(seed)
        print(f"使用指定种子: {seed}")
    
    save_folder = gen_save_folder()
    generator = torch.Generator().manual_seed(seed)
    
    if t2i_worker:
        print(f"正在从文本生成图像...")
        try:
            image = t2i_worker(prompt)
            input_image_path = os.path.join(CURRENT_DIR, f"input_{int(time.time())}.png")
            image.save(input_image_path)
            print(f"生成的参考图像已保存到: {input_image_path}")
            image.save(os.path.join(save_folder, 'input.png'))
        except Exception as e:
            print(f"文本生成图像失败: {e}")
            return None
    else:
        print(f"文本到图像模型未加载，无法生成图像")
        return None
    
    print(f"正在移除图像背景...")
    image = rmbg_worker(image.convert('RGB'))
    image.save(os.path.join(save_folder, 'rembg.png'))
    
    print(f"正在生成3D模型...")
    mesh = i23d_worker(
        image=image,
        num_inference_steps=50,
        guidance_scale=5.5,
        generator=generator,
        octree_resolution=256
    )[0]
    
    print(f"正在优化3D模型...")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    
    if texgen_worker:
        print(f"正在生成纹理...")
        textured_mesh = texgen_worker(mesh, image)
        output_path = export_mesh(textured_mesh, save_folder, textured=True)
        print(f"带纹理的3D模型已保存到: {output_path}")
        return output_path
    else:
        output_path = export_mesh(mesh, save_folder, textured=False)
        print(f"3D模型已保存到: {output_path}")
        return output_path

def process_model(glb_file):
    """处理GLB文件，进行体素化"""
    print(f"正在处理模型: {glb_file}")
    
    output_dir = os.getcwd()
    ply_path = os.path.join(output_dir, "voxel_model_32x32x32.ply")
    txt_path = os.path.join(output_dir, "voxel_model.txt")
    
    ply_file = None
    txt_file = None
    
    try:
        viewer.view_model(glb_file, output_dir)
        
        if os.path.exists(ply_path):
            ply_file = ply_path
        else:
            print(f"体素化模型生成失败！PLY 文件未找到: {ply_path}")
        
        if os.path.exists(txt_path):
            txt_file = txt_path
        else:
            print(f"体素文本生成失败！TXT 文件未找到: {txt_path}")
        
        if ply_file and txt_file:
            print(f"模型已体素化，PLY文件: {ply_file}, TXT文件: {txt_file}")
        else:
            print("体素化过程部分失败")
            
    except Exception as e:
        print(f"体素化过程中发生错误: {e}")
    
    return ply_file, txt_file

def parse_voxel_text(text):
    """解析分层文本为32x32x32三维数组"""
    layers = []
    current_layer = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("---"):
            if current_layer:
                layers.append(current_layer)
                current_layer = []
            continue
        if not line:
            continue
        
        tokens = line.split()
        row = []
        for token in tokens:
            if token == '.':
                row.append(0)
            else:
                try:
                    row.append(int(token))
                except ValueError:
                    row.append(0)
        
        row = row[:32] + [0]*(32 - len(row)) if len(row) < 32 else row[:32]
        current_layer.append(row)
        
        if len(current_layer) == 32:
            layers.append(current_layer)
            current_layer = []
    
    if current_layer:
        layers.append(current_layer)
    
    layers = layers[:32]
    while len(layers) < 32:
        layers.append([[0]*32 for _ in range(32)])
    
    for y in range(32):
        layer = layers[y][:32]
        while len(layer) < 32:
            layer.append([0]*32)
        for z in range(32):
            layer[z] = layer[z][:32] + [0]*(32 - len(layer[z])) if len(layer[z]) < 32 else layer[z][:32]
        layers[y] = layer

    return 32, 32, 32, layers

def text_to_schematic(input_file, prompt, seed):
    """转换文本为32x32x32 schematic文件"""
    safe_prompt = re.sub(r'[<>:"/\\|?*]', '', prompt).replace(' ', '_')
    output_file = os.path.join(OUTPUT_DIR, f"{safe_prompt}_{seed}.schematic")
    
    with open(input_file, 'r', encoding="utf-8") as f:
        text = f.read()

    width, height, length, layers = parse_voxel_text(text)

    nbtfile = nbt.nbt.NBTFile()
    nbtfile.name = "Schematic"
    
    nbtfile.tags.extend([
        nbt.nbt.TAG_Short(name="Width", value=width),
        nbt.nbt.TAG_Short(name="Height", value=height),
        nbt.nbt.TAG_Short(name="Length", value=length),
    ])

    blocks = []
    for y in range(32):
        for z in range(32):
            for x in range(32):
                blocks.append(layers[y][z][x])

    blocks_tag = nbt.nbt.TAG_Byte_Array(name="Blocks")
    blocks_tag.value = bytearray(blocks)

    data_tag = nbt.nbt.TAG_Byte_Array(name="Data")
    data_tag.value = bytearray([0] * 32768)

    nbtfile.tags.extend([
        blocks_tag,
        data_tag,
        nbt.nbt.TAG_String(name="Materials", value="Alpha"),
        nbt.nbt.TAG_List(name="Entities", type=nbt.nbt.TAG_Compound),
        nbt.nbt.TAG_List(name="TileEntities", type=nbt.nbt.TAG_Compound)
    ])

    nbtfile.write_file(output_file)
    print(f"成功生成 schematic 文件：{output_file}")
    return output_file

def check_existing_models(skip=False):
    """检查output文件夹中是否已有.glb模型文件，根据skip参数决定行为"""
    glb_files = glob.glob(os.path.join(OUTPUT_DIR, "*.glb"))
    if glb_files:
        if skip:
            print(f"发现现有模型文件：{glb_files[0]}，由于指定 -skip，跳过Hunyuan模型生成")
            return glb_files[0]
        else:
            print(f"发现现有模型文件：{glb_files[0]}，正在删除以生成新模型")
            try:
                os.remove(glb_files[0])
                print(f"已删除现有模型文件：{glb_files[0]}")
            except Exception as e:
                print(f"删除现有模型文件 {glb_files[0]} 失败：{e}")
    return None

def get_user_input(prompt_text):
    """获取用户输入，支持终端和Notebook环境"""
    if IN_NOTEBOOK:
        from IPython.display import display
        from ipywidgets import Text, Button
        import threading

        response = [None]
        event = threading.Event()

        text_input = Text(value='', placeholder=prompt_text, description='输入:')
        submit_button = Button(description='提交')

        def on_button_clicked(b):
            response[0] = text_input.value.strip()
            event.set()

        submit_button.on_click(on_button_clicked)
        display(text_input)
        display(submit_button)

        event.wait()
        print("消息已收到")
        return response[0]
    else:
        response = input(prompt_text).strip()
        print("消息已收到")
        return response

def run_generation(prompt, seed, api_key, skip_existing, ask_key=True):
    """执行生成流程"""
    print(f"==========================================\n"
          f"Minecraft 3D模型生成工具\n"
          f"==========================================")
    
    if not prompt or not prompt.strip():
        print(f"错误：提示词不能为空")
        return False, api_key
    
    if ask_key or not api_key:
        api_key = get_user_input("请输入API密钥（留空尝试使用本地密钥）：")
        if not api_key:
            api_key = load_api_key()
            if not api_key:
                print(f"警告：未提供API密钥，且本地未找到密钥，将尝试继续运行")
    
    if not seed or seed == "":
        seed = str(int(time.time()) % 10000)
    else:
        seed = str(seed)

    glb_file = check_existing_models(skip=skip_existing)
    
    if not glb_file:
        glb_file = generate_3d_model(prompt, seed)
        
        if not glb_file:
            print(f"3D模型生成失败")
            return False, api_key
    
    print(f"开始体素化处理...")
    ply_file, txt_file = process_model(glb_file)
    
    if not ply_file or not txt_file:
        print(f"体素化失败")
        return False, api_key
    
    print(f"开始AI颜色映射...")
    color_mapping = analyze_images_and_voxel_with_key(api_key)
    
    if not color_mapping:
        print(f"AI颜色映射失败")
        return False, api_key
    
    working_file = "working.txt"
    if not os.path.exists(working_file):
        print(f"错误：{working_file} 文件未生成")
        return False, api_key
    
    print(f"转换为Minecraft Schematic 文件...")
    schematic_file = text_to_schematic(working_file, prompt, seed)
    
    if not schematic_file:
        print(f"Schematic 文件生成失败")
        return False, api_key
    
    cleanup_glb_files()
    
    print(f"==========================================\n"
          f"处理完成！\n"
          f"Schematic 文件已保存到: {schematic_file}\n"
          f"==========================================")
    
    cleanup_image_files()
    return True, api_key

def main(args):
    """主函数"""
    global t2i_worker, i23d_worker, texgen_worker, rmbg_worker, FloaterRemover, DegenerateFaceRemover, FaceReducer
    
    # 从命令行获取参数，若未提供则设为 None
    prompt = args.prompt if args.prompt else None
    api_key = args.key if args.key else None
    seed = args.seed if args.seed else ""
    skip_existing = args.skip
    
    # 加载模型（仅一次）
    setup_hunyuan_model()
    
    if not t2i_worker or not i23d_worker or not rmbg_worker or not FloaterRemover or not DegenerateFaceRemover or not FaceReducer:
        print("错误：模型加载失败，程序退出")
        return
    
    # 如果命令行未提供 prompt，则询问
    if not prompt:
        prompt = get_user_input("请输入生成3D模型的提示词：")
    
    # 首次运行
    success, api_key = run_generation(prompt, seed, api_key, skip_existing, ask_key=(not args.key))
    
    if not success:
        print("首次运行失败，程序退出")
        return
    
    # 交互循环
    while True:
        user_input = get_user_input("是否继续运行？请输入新的提示词（按回车继续），或 'exit' 退出：")
        
        if user_input.lower() == 'exit':
            print("用户选择退出，程序结束")
            break
        
        if not user_input:
            print("提示词不能为空，请重新输入")
            continue
        
        success, api_key = run_generation(user_input, "", api_key, skip_existing, ask_key=False)
        if not success:
            print("生成失败，将继续询问用户是否继续")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft 3D模型生成工具")
    parser.add_argument("-prompt", type=str, help="生成3D模型的提示词（可选）")
    parser.add_argument("-seed", type=str, default="", help="随机种子（可选，留空则随机生成）")
    parser.add_argument("-key", type=str, default="", help="API密钥（可选，若未提供则询问）")
    parser.add_argument("-skip", action="store_true", help="跳过模型生成，使用现有模型文件")
    
    args = parser.parse_args()
    main(args)
