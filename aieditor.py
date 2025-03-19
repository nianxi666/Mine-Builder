import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO
import time
import traceback

API_KEY_PATH = 'api_key.json'

def read_file(filename: str):
    """读取指定文件内容，如果文件不存在则报错"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found!")
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

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

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_block_id",
            "description": "根据给定的颜色，返回对应的Minecraft方块ID。",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {"type": "string", "description": "体素文件中的颜色名称"},
                    "block_id": {"type": "string", "description": "对应的Minecraft方块ID (如 '1' 或 '251:1')"}
                },
                "required": ["color", "block_id"]
            }
        }
    }
]

def extract_colors_from_voxel_text(file_content):
    """从体素文本内容中提取所有不同的颜色名称"""
    colors = set()
    lines = file_content.splitlines()
    for line_number, line in enumerate(lines):
        mod = line_number % 34
        if 1 <= mod <= 32:
            components = line.strip().split()
            for comp in components:
                if comp != ".":
                    colors.add(comp)
    return sorted(colors)

def preserve_voxel_structure(file_content):
    """保留voxel_model.txt的原始结构，返回行列表"""
    return file_content.splitlines()

def replace_colors_with_ids(lines, color_to_id):
    """将原始行中的颜色替换为对应的block_id"""
    new_lines = []
    for line_number, line in enumerate(lines):
        mod = line_number % 34
        if 1 <= mod <= 32:
            components = line.strip().split()
            new_components = [color_to_id.get(comp, comp) if comp != "." else "." for comp in components]
            new_lines.append(" ".join(new_components))
        else:
            new_lines.append(line)
    return new_lines

def write_working_file(lines, output_filename="working.txt"):
    """将替换后的内容写入working.txt"""
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✅ 已生成输出文件: {output_filename}")

def make_openrouter_api_call(messages, api_key=None, tools=None):
    """调用 OpenRouter API，使用 google/gemini-2.0-pro-exp-02-05:free 模型"""
    if not api_key:
        api_key = load_api_key()
        if not api_key:
            print("警告：未提供API密钥，将尝试使用免费模型")

    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "YOUR_SITE_URL",  # 请替换为您的应用 URL
        "X-Title": "YOUR_SITE_NAME"       # 请替换为您的应用名称
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = {
        "model": "google/gemini-2.0-flash-exp:free",  # 更换为指定模型
        "messages": messages,
        "temperature": 0.75,
        "top_p": 0.95,
        "max_tokens": 8192
    }
    if tools:
        data["tools"] = tools
        data["tool_choice"] = "auto"

    try:
        response = requests.post(endpoint, headers=headers, json=data)
        if response.status_code == 200:
            try:
                json_data = response.json()
                print(f"调试：API返回的JSON数据: {json.dumps(json_data, indent=2)}")
                return json_data
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                print(f"原始响应内容: {response.text}")
                with open("debug_response.txt", "w", encoding="utf-8") as f:
                    f.write(response.text)
                print("已将响应保存到 debug_response.txt")
                return None
        else:
            print(f"❌ API调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ API调用失败: {e}")
        return None

def analyze_images_and_voxel(image_dir=".", max_retries=3, api_key=None):
    """分析图像和体素数据，不验证密钥"""
    if not api_key:
        api_key = load_api_key()

    try:
        with open("blockids.json", 'r', encoding='utf-8-sig') as f:
            blockids_data = json.load(f)
        print("\n✅ blockids.json 文件读取成功.")
    except FileNotFoundError:
        error_msg = "blockids.json 文件未找到！请确保该文件存在于当前目录。"
        print(f"\n❌ {error_msg}")
        return f"错误：{error_msg}"
    except json.JSONDecodeError as e:
        error_msg = f"blockids.json 文件格式错误：{str(e)}"
        print(f"\n❌ {error_msg}")
        return f"错误：{error_msg}"

    try:
        voxel_file_path = os.path.join(image_dir, "voxel_model.txt")
        voxel_content = read_file(voxel_file_path)
        print(f"\n✅ {voxel_file_path} 文件读取成功.")
        colors = extract_colors_from_voxel_text(voxel_content)
        original_lines = preserve_voxel_structure(voxel_content)
        print(f"\n发现以下颜色需要映射：{', '.join(colors)}")
    except FileNotFoundError as e:
        error_msg = f"{voxel_file_path} 文件未找到！请确保已完成模型体素化步骤。"
        print(f"\n❌ 错误：{error_msg}")
        return f"错误：{error_msg}"

    image_filenames = [
        filename for filename in os.listdir(image_dir)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        and not filename.lower().startswith('extracted')
    ]
    image_data = []
    for filename in image_filenames:
        image_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(image_path)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data.append({
                "filename": filename,
                "image_url": f"data:image/jpeg;base64,{img_str}"
            })
            print(f"\n✅ 图片 {filename} 加载成功.")
        except Exception as e:
            print(f"❌ 图片 {filename} 加载失败：{str(e)}")

    color_to_id = {}
    image_description = None
    retry_count = 0

    while retry_count <= max_retries:
        if retry_count == 0:
            messages = [
                {
                    "role": "user",
                    "content": f"""
                    请根据以下图片和文件内容完成以下任务：
                    1. 分析每张图片中的建筑结构、材料和可能的用途，并生成一个详细的描述。
                    2. 为 voxel_model.txt 中的颜色列表分配方块ID。

                    参考以下 blockids.json 内容：
                    {json.dumps(blockids_data, indent=2)}

                    需要映射的颜色：
                    {', '.join(colors)}

                    请先提供图片描述，然后为每个颜色调用 'get_block_id' 函数返回结果，
                    参数中明确指定 'color' 和 'block_id'。
                    'block_id' 为字符串，如 '1' 或 '251:1'。
                    优先从 blockids.json 中提取映射，若有多个选项，根据图片描述选择最合适的值，
                    若无明确映射，则根据描述推测合理ID。
                    注意：以下是图片的base64编码数据。
                    """.strip() + "\n" + "\n".join([f"图片 {img['filename']}: {img['image_url']}" for img in image_data])
                }
            ]
        else:
            missing_colors = [color for color in colors if color not in color_to_id]
            if not missing_colors:
                break
            messages = [
                {
                    "role": "user",
                    "content": f"""
                    以下是图片的描述：
                    {image_description}

                    以下颜色尚未映射：
                    {', '.join(missing_colors)}

                    请为这些颜色提供Minecraft方块ID。
                    参考 blockids.json：
                    {json.dumps(blockids_data, indent=2)}

                    为每个颜色调用 'get_block_id' 函数返回结果，
                    参数中明确指定 'color' 和 'block_id'。
                    'block_id' 为字符串，如 '1' 或 '251:1'。
                    优先从 blockids.json 中提取映射，若有多个选项，根据描述选择最合适的值，
                    若无明确映射，则根据描述推测合理ID。
                    """.strip()
                }
            ]

        print(f"\n🔄 第 {retry_count + 1} 次调用API进行分析...")
        response = make_openrouter_api_call(messages, api_key=api_key, tools=tools)
        if not response:
            retry_count += 1
            continue
        print("\n✅ API调用成功")

        if retry_count == 0:
            image_description = response["choices"][0]["message"]["content"]
            print(f"\n📝 图片描述:\n{image_description}")

        if "tool_calls" in response["choices"][0]["message"]:
            for tool_call in response["choices"][0]["message"]["tool_calls"]:
                func_name = tool_call["function"]["name"]
                arguments_str = tool_call["function"]["arguments"]
                print(f"调试：原始 arguments 字符串: {arguments_str}")
                try:
                    func_args = json.loads(arguments_str)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析失败: {e}")
                    try:
                        end_idx = arguments_str.find('}') + 1
                        if end_idx > 0:
                            fixed_args = arguments_str[:end_idx]
                            func_args = json.loads(fixed_args)
                            print(f"修复：截取第一个JSON对象成功: {fixed_args}")
                        else:
                            print("无法修复：未找到完整的JSON对象")
                            continue
                    except json.JSONDecodeError as e2:
                        print(f"修复失败: {e2}")
                        continue

                if func_name == "get_block_id" and "color" in func_args and "block_id" in func_args:
                    color = func_args["color"]
                    block_id = func_args["block_id"]
                    if block_id in blockids_data:
                        color_to_id[color] = block_id
                        print(f"✅ 颜色 '{color}' 成功映射为方块ID: {block_id}")
                    else:
                        print(f"❌ 无效的block_id: {block_id}，跳过颜色 '{color}'")

        missing_colors = [color for color in colors if color not in color_to_id]
        if not missing_colors:
            break
        print(f"⚠️ 以下颜色未被映射: {', '.join(missing_colors)}")
        retry_count += 1

    if missing_colors:
        print(f"\n⚠️ 经过 {max_retries} 次重试，仍有以下颜色未映射: {', '.join(missing_colors)}")
        print("这些颜色将保留原始值。")

    print("\n🎉 颜色到方块ID的映射结果:")
    for color, block_id in color_to_id.items():
        print(f"颜色 '{color}' -> 方块ID: {block_id}")

    new_lines = replace_colors_with_ids(original_lines, color_to_id)
    write_working_file(new_lines)

    return color_to_id

def analyze_images_and_voxel_with_key(api_key):
    """为 generate_model.py 提供兼容接口"""
    result = analyze_images_and_voxel(api_key=api_key)
    if isinstance(result, str) and result.startswith("错误"):
        return None
    return result

if __name__ == "__main__":
    api_key = load_api_key()
    result = analyze_images_and_voxel(api_key=api_key)
    print(f"\n\n最终结果:\n{result}")
