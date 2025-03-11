import subprocess
import threading
import os
import queue
import requests
import json
import zipfile
import shutil
import time
import re

# 全局日志队列和日志存储
log_queue = queue.Queue()
full_logs = ""
server_process = None  # 将保存服务器进程

def setup_server():
    global full_logs
    # 使用当前目录作为工作目录
    os.makedirs(os.getcwd(), exist_ok=True)
    os.chdir(os.getcwd())

    log_queue.put("获取最新官方Minecraft版本信息...")
    manifest_url = "https://launchermeta.mojang.com/mc/game/version_manifest.json"
    try:
        manifest_response = requests.get(manifest_url)
        manifest_data = manifest_response.json()
    except Exception as e:
        log_queue.put(f"获取版本信息失败：{e}")
        return
    latest_release = manifest_data["latest"]["release"]

    version_url = ""
    for version in manifest_data["versions"]:
        if version["id"] == latest_release:
            version_url = version["url"]
            break
    if not version_url:
        log_queue.put("错误：无法找到最新版本信息！")
        return

    log_queue.put(f"最新官方Minecraft版本: {latest_release}")
    try:
        version_response = requests.get(version_url)
        version_data = version_response.json()
        server_url = version_data["downloads"]["server"]["url"]
    except Exception as e:
        log_queue.put(f"获取服务器下载链接失败：{e}")
        return

    if os.path.exists("server.jar"):
        os.remove("server.jar")
    log_queue.put("下载官方Minecraft服务器文件...")
    try:
        response = requests.get(server_url)
        with open("server.jar", "wb") as f:
            f.write(response.content)
        log_queue.put(f"官方Minecraft服务器文件下载完成。版本: {latest_release}")
    except Exception as e:
        log_queue.put(f"下载服务器文件失败：{e}")

    # 生成或修改 EULA 文件
    if not os.path.exists("eula.txt"):
        with open("eula.txt", "w") as f:
            f.write("eula=true\n")
        log_queue.put("EULA 文件已生成并同意。")
    else:
        with open("eula.txt", "r") as f:
            content = f.read().replace("false", "true")
        with open("eula.txt", "w") as f:
            f.write(content)
        log_queue.put("EULA 已同意。")

    # 如果存在 world.zip，则解压地图
    if os.path.exists("world.zip"):
        if os.path.exists("world"):
            shutil.rmtree("world")
        with zipfile.ZipFile("world.zip", "r") as zip_ref:
            zip_ref.extractall("world")
        log_queue.put("服务器地图已从 world.zip 提取至 world 文件夹。")

def start_server():
    global server_process
    setup_server()
    log_queue.put("服务器启动中...（可能需要几分钟时间生成世界）")

    try:
        server_process = subprocess.Popen(
            ["java", "-Xmx1024M", "-Xms1024M", "-jar", "server.jar", "--nogui"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd()
        )
    except Exception as e:
        log_queue.put(f"服务器启动失败：{e}")
        return "服务器启动失败"

    def read_output():
        while True:
            line = server_process.stdout.readline()
            if not line:
                break
            log_queue.put(line.strip())
    threading.Thread(target=read_output, daemon=True).start()
    return "服务器已启动！请注意：如果是首次生成世界，可能需要几分钟才能完全启动。"

def execute_commands_from_file():
    if not os.path.exists("commands.txt"):
        log_queue.put("错误：命令文件 commands.txt 不存在！")
        return "错误：命令文件 commands.txt 不存在！"
    if server_process is None or server_process.poll() is not None:
        log_queue.put("错误：服务器未运行，请先启动服务器。")
        return "错误：服务器未运行，请先启动服务器。"

    with open("commands.txt", "r") as f:
        commands = f.readlines()
    commands = [cmd.strip() for cmd in commands if cmd.strip()]
    total = len(commands)
    if total == 0:
        log_queue.put("错误：命令文件中没有有效命令！")
        return "错误：命令文件中没有有效命令！"

    executed_count = 0
    log_queue.put(f"开始执行 {total} 条命令...")

    for idx, cmd in enumerate(commands, 1):
        try:
            server_process.stdin.write(cmd + "\n")
            server_process.stdin.flush()
            executed_count += 1
            remaining = total - executed_count
            log_queue.put(f"命令 [{idx}/{total}]: '{cmd}' 发送成功。已发送: {executed_count}, 剩余: {remaining}.")
        except Exception as e:
            log_queue.put(f"命令 [{idx}/{total}]: '{cmd}' 发送失败，错误信息: {e}")
    log_queue.put("所有命令发送完成。")
    return f"已执行 {executed_count} 条命令。"

def zip_world(output_dir):
    zip_file = os.path.join(output_dir, "world_download.zip")
    if not os.path.exists("world"):
        log_queue.put("错误：world 文件夹不存在！")
        return "错误：world 文件夹不存在！"
    if os.path.exists(zip_file):
        os.remove(zip_file)
    try:
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk("world"):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, "world")
                    zipf.write(file_path, arcname)
        log_queue.put("地图打包完成，保存至 " + zip_file)
    except Exception as e:
        log_queue.put(f"打包地图失败: {e}")
        return f"打包地图失败: {e}"
    return zip_file

def parse_voxel_text(text):
    layers = []
    current_layer = []
    line_count = 0
    for line in text.splitlines():
        line_count += 1
        line = line.strip()
        if line.startswith("---"):
            if current_layer:
                layers.append(current_layer)
                print(f"已读取一个层，共 {len(current_layer)} 行（截止到行 {line_count}）")
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
        if len(row) < 32:
            row = row[:32] + [0] * (32 - len(row))
        else:
            row = row[:32]
        current_layer.append(row)
        if len(current_layer) == 32:
            layers.append(current_layer)
            print(f"完成一个 32x32 层（截止到行 {line_count}）")
            current_layer = []
    
    if current_layer:
        layers.append(current_layer)
        print("保存了最后一个未满32行的层。")
    
    layers = layers[:32]
    while len(layers) < 32:
        layers.append([[0] * 32 for _ in range(32)])
        print("补充了一个全0的层以达到32层。")
    
    for y in range(32):
        layer = layers[y][:32]
        while len(layer) < 32:
            layer.append([0] * 32)
        for z in range(32):
            if len(layer[z]) < 32:
                layer[z] = layer[z][:32] + [0] * (32 - len(layer[z]))
            else:
                layer[z] = layer[z][:32]
        layers[y] = layer

    print("体素文本解析完成，获得 32x32x32 数据结构。")
    return 32, 32, 32, layers

def load_legacy_dict(legacy_file):
    legacy = {}
    with open(legacy_file, 'r', encoding="utf-8") as f:
        content = f.read()
    
    pattern = re.compile(r'["\'](\d+:\d+)["\']\s*:\s*["\']([\w:]+)["\']')
    matches = pattern.findall(content)
    
    for key, value in matches:
        legacy[key] = value

    print(f"加载字典: 共加载 {len(legacy)} 个映射。")
    return legacy

def text_to_setblock_commands(input_file, legacy_file, start_coord=(753, 5, 25), output_file="commands.txt"):
    with open(input_file, 'r', encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        print("警告：输入文件为空！")
    print("开始解析体素文本...")
    width, height, length, layers = parse_voxel_text(text)
    print(f"解析尺寸：宽={width} 高={height} 长={length}")

    print("开始加载块映射字典...")
    legacy_mapping = load_legacy_dict(legacy_file)
    
    start_x, start_y, start_z = start_coord
    commands = []
    counter = 0

    print("开始生成 /setblock 指令...")
    for y in range(height):
        for z in range(length):
            for x in range(width):
                counter += 1
                block_id = layers[y][z][x]
                key = f"{block_id}:0"
                block_name = legacy_mapping.get(key, "minecraft:air")
                if block_name == "minecraft:air":
                    continue
                mc_x = start_x + x
                mc_y = start_y + y
                mc_z = start_z + z
                command = f"/setblock {mc_x} {mc_y} {mc_z} {block_name}"
                commands.append(command)
    
    print(f"遍历了 {counter} 个块，共生成 {len(commands)} 条指令。")
    
    with open(output_file, 'w', encoding="utf-8") as f:
        f.write("\n".join(commands))
    
    print(f"成功生成指令文件：{os.path.abspath(output_file)}")
    print(f"文件大小：{os.path.getsize(output_file):,} 字节")
    return os.path.abspath(output_file)

def process_voxel_file():
    input_file = "working.txt"
    legacy_file = "legacy.txt"
    output_file = "commands.txt"
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return None
    elif not os.path.exists(legacy_file):
        print(f"错误：字典文件 {legacy_file} 不存在")
        return None
    else:
        print("开始转换体素数据为 /setblock 指令...")
        text_to_setblock_commands(input_file, legacy_file, start_coord=(753, 5, 25), output_file=output_file)
        return output_file

def log_printer():
    while True:
        while not log_queue.empty():
            log_line = log_queue.get()
            print(log_line)
        time.sleep(1)

def periodic_zip_world(output_dir):
    while True:
        time.sleep(60)  # 每60秒打包一次
        try:
            zip_world(output_dir)
        except Exception as e:
            log_queue.put("定时打包: 打包地图失败: " + str(e))

def main():
    # 启动日志打印线程
    threading.Thread(target=log_printer, daemon=True).start()
    
    # 启动服务器
    status = start_server()
    print(status)
    
    # 等待服务器启动
    time.sleep(10)
    
    # 处理工作文件并生成 commands.txt
    commands_file = process_voxel_file()
    if commands_file is None:
        print("命令文件生成失败。")
        return
    
    # 发送 commands.txt 中的所有命令到服务器
    cmd_status = execute_commands_from_file()
    print(cmd_status)
    
    # 确保输出目录存在
    output_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 启动定时打包线程，每隔1分钟打包一次地图
    threading.Thread(target=periodic_zip_world, args=(output_dir,), daemon=True).start()
    
    # 保持主线程等待服务器进程退出
    server_process.wait()

if __name__ == "__main__":
    main()
