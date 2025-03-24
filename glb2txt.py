import os
import trimesh
import numpy as np

def glb_to_voxel_text(glb_file, target_x=11, target_y=11, target_z=11):
    """
    加载 GLB 文件，生成 1:1:1 比例且 x/y/z 方向均为指定层数的体素数据文本。
    """
    try:
        # 加载 GLB 文件
        scene = trimesh.load(glb_file)

        # 合并网格（处理 Scene 对象）
        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_geometry()
        else:
            mesh = scene

        # 获取模型包围盒
        bounds = mesh.bounds
        min_bounds = bounds[0]
        max_bounds = bounds[1]

        # 计算模型在 x/y/z 方向的实际长度
        x_length = max_bounds[0] - min_bounds[0]
        y_length = max_bounds[1] - min_bounds[1]
        z_length = max_bounds[2] - min_bounds[2]

        # 计算各方向所需的体素尺寸
        voxel_size_x = x_length / (target_x - 1)
        voxel_size_y = y_length / (target_y - 1)
        voxel_size_z = z_length / (target_z - 1)

        # 取最大体素尺寸以保证所有方向满足层数要求
        voxel_size = max(voxel_size_x, voxel_size_y, voxel_size_z)

        # 体素化网格
        voxels = mesh.voxelized(pitch=voxel_size)

        # 获取体素网格矩阵（形状为 [x_size, y_size, z_size]）
        voxel_grid = voxels.matrix

        # 如果实际层数小于目标值，填充空白体素
        if voxel_grid.shape[0] < target_x or voxel_grid.shape[1] < target_y or voxel_grid.shape[2] < target_z:
            padded_grid = np.zeros((target_x, target_y, target_z), dtype=bool)
            padded_grid[:voxel_grid.shape[0], :voxel_grid.shape[1], :voxel_grid.shape[2]] = voxel_grid
            voxel_grid = padded_grid

        # 按 y 层生成文本
        output_text = ""
        for y in range(voxel_grid.shape[1]):
            output_text += f"--- Level {y} ---\n"
            for z in range(voxel_grid.shape[2]):
                line = ''
                for x in range(voxel_grid.shape[0]):
                    if voxel_grid[x, y, z]:
                        line += "1 "
                    else:
                        line += "0 "
                output_text += line.strip() + "\n"
            output_text += "\n"
        return output_text

    except FileNotFoundError:
        return f"错误：找不到文件：{glb_file}"
    except Exception as e:
        return f"发生未知错误: {e}"

def save_voxel_text(voxel_text, output_file):
    """
    将体素数据文本保存到文件中。
    """
    try:
        with open(output_file, 'w') as f:
            f.write(voxel_text)
        print(f"体素数据已保存到：{output_file}")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

if __name__ == "__main__":
    # 强制加载当前目录下的 schematic.glb
    glb_file = "schematic.glb"
    output_file = "output.txt"

    # 检查文件是否存在
    if os.path.exists(glb_file):
        voxel_text = glb_to_voxel_text(glb_file, target_x=11, target_y=11, target_z=11)
        save_voxel_text(voxel_text, output_file)
    else:
        print(f"错误：文件不存在：{glb_file}")
