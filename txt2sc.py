import os
import nbt
from nbt.nbt import TAG_Short, TAG_Byte_Array, TAG_String, TAG_List, TAG_Compound

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
        
        # 转换行数据
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
        
        # 调整行长度到32
        row = row[:32] + [0]*(32 - len(row)) if len(row) < 32 else row[:32]
        current_layer.append(row)
        
        # 层满32行时保存
        if len(current_layer) == 32:
            layers.append(current_layer)
            current_layer = []
    
    # 处理未完成的层
    if current_layer:
        layers.append(current_layer)
    
    # 调整总层数到32
    layers = layers[:32]
    while len(layers) < 32:
        layers.append([[0]*32 for _ in range(32)])
    
    # 确保每层格式正确
    for y in range(32):
        layer = layers[y][:32]  # 截断行数
        while len(layer) < 32:
            layer.append([0]*32)
        for z in range(32):
            layer[z] = layer[z][:32] + [0]*(32 - len(layer[z])) if len(layer[z]) < 32 else layer[z][:32]
        layers[y] = layer

    return 32, 32, 32, layers

def text_to_schematic(input_file, output_file):
    """转换文本为32x32x32 schematic文件"""
    with open(input_file, 'r', encoding="utf-8") as f:
        text = f.read()

    width, height, length, layers = parse_voxel_text(text)

    nbtfile = nbt.nbt.NBTFile()
    nbtfile.name = "Schematic"
    
    # 添加基础尺寸标签
    nbtfile.tags.extend([
        TAG_Short(name="Width", value=width),
        TAG_Short(name="Height", value=height),
        TAG_Short(name="Length", value=length),
    ])

    # 构建方块数组(YZX顺序)
    blocks = []
    for y in range(32):
        for z in range(32):
            for x in range(32):
                blocks.append(layers[y][z][x])

    # 添加必要NBT标签
    blocks_tag = TAG_Byte_Array(name="Blocks")
    blocks_tag.value = bytearray(blocks)

    data_tag = TAG_Byte_Array(name="Data")
    data_tag.value = bytearray([0] * 32768)  # 32*32*32=32768

    nbtfile.tags.extend([
        blocks_tag,
        data_tag,
        TAG_String(name="Materials", value="Alpha"),
        TAG_List(name="Entities", type=TAG_Compound),
        TAG_List(name="TileEntities", type=TAG_Compound)
    ])

    # 写入文件
    nbtfile.write_file(output_file)
    return os.path.abspath(output_file)

if __name__ == "__main__":
    input_file = "working.txt"
    output_file = "output.schematic"
    
    if os.path.exists(input_file):
        result_path = text_to_schematic(input_file, output_file)
        print(f"成功生成 schematic 文件：{result_path}")
        print(f"文件大小：{os.path.getsize(result_path):,} 字节")
    else:
        print(f"错误：输入文件 {input_file} 不存在")
