import threading
import time
import zipfile
import os

def start_server():
    """
    模拟服务器启动和运行的后台线程。
    如果需要启动真实服务器，可以在此处调用相应的启动代码。
    """
    print("服务器启动中...")
    while True:
        # 模拟服务器持续运行（可以添加服务器心跳或监听代码）
        time.sleep(1)

def load_and_convert():
    """
    读取 working.txt 文件内容，转换后写入 commands.txt。
    此处转换逻辑仅为简单复制，可根据需要修改处理逻辑。
    """
    input_file = "working.txt"
    output_file = "commands.txt"
    
    if not os.path.exists(input_file):
        print(f"未找到文件: {input_file}")
        return False
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 此处可添加额外的转换逻辑
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已将 {input_file} 转换为 {output_file}。")
    return True

def send_commands():
    """
    读取 commands.txt 中的命令并发送给服务器（此处为模拟发送）。
    实际使用时，可将命令通过 socket、API 等方式发送给服务器。
    """
    commands_file = "commands.txt"
    
    if not os.path.exists(commands_file):
        print(f"未找到文件: {commands_file}")
        return
    
    with open(commands_file, 'r', encoding='utf-8') as f:
        commands = f.read().strip().splitlines()
    
    for command in commands:
        # 模拟发送命令（替换为实际发送逻辑）
        print(f"发送命令到服务器: {command}")
        time.sleep(0.1)  # 模拟发送延时
    
    print("所有命令已发送。")

def update_map_and_zip():
    """
    每隔一分钟生成一次服务器地图文件，然后将其压缩保存到 output 目录下。
    """
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    while True:
        # 模拟生成服务器地图内容，这里简单使用当前时间作为示例
        map_content = f"服务器地图更新时间: {time.ctime()}\n"
        map_file = "server_map.txt"
        
        with open(map_file, 'w', encoding='utf-8') as f:
            f.write(map_content)
        
        print("服务器地图已更新。")
        
        # 压缩生成的地图文件
        zip_filename = os.path.join(output_dir, f"server_map_{int(time.time())}.zip")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(map_file)
        
        print(f"已将服务器地图压缩为: {zip_filename}")
        
        # 等待 60 秒后再次更新
        time.sleep(60)

def main():
    # 启动模拟服务器线程
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # 启动地图更新线程
    map_thread = threading.Thread(target=update_map_and_zip, daemon=True)
    map_thread.start()
    
    # 自动加载 working.txt 并转换为 commands.txt
    if load_and_convert():
        # 加载完成后自动发送命令
        send_commands()
    
    # 保持主线程运行（按 Ctrl+C 退出）
    print("程序正在运行，按 Ctrl+C 退出。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序已退出。")

if __name__ == "__main__":
    main()
