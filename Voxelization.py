import trimesh
import numpy as np
import json
import logging
import os
from skimage.color import rgb2hsv
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载 blockids.json 文件
def load_block_colors(filename):
    """
    从 blockids.json 文件加载 RGB 颜色值和对应的颜色名称。
    """
    block_colors = {}
    if not os.path.exists(filename):
        logger.error(f"文件 {filename} 不存在")
        raise FileNotFoundError(f"Block colors file {filename} not found")
    
    logger.info(f"加载文件: {filename}")
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:  # 使用 utf-8-sig 处理 BOM
            logger.debug("成功打开文件")
            data = json.load(f)
            if not data:
                logger.error(f"{filename} 文件为空")
                raise ValueError(f"Block colors file {filename} is empty")
            logger.debug(f"JSON 文件内容样本: {list(data.items())[:2]}")
            for key, value in data.items():
                if 'color' not in value or 'description' not in value:
                    logger.warning(f"条目 {key} 格式错误: {value}")
                    continue
                color = value['color']
                description = value['description']
                color_name = description.split('-')[-1].strip()
                rgb = tuple(color[:3]) if len(color) == 4 else tuple(color)
                block_colors[rgb] = color_name
            if not block_colors:
                logger.error("未从文件中解析出任何颜色数据")
                raise ValueError("No valid color data parsed from file")
            logger.info(f"成功加载 {len(block_colors)} 个颜色条目")
            logger.debug(f"颜色字典样本: {list(block_colors.items())[:2]}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误: {e}")
        raise
    except Exception as e:
        logger.error(f"加载 {filename} 时出错: {e}")
        raise
    return block_colors

# 获取最近的颜色名称
def get_color_name(rgb, block_colors):
    """
    根据 RGB 值找到 block_colors 中最相近的颜色对应的名称。
    """
    if not block_colors:
        logger.warning("block_colors 字典为空，返回 'unknown'")
        return "unknown"
    min_distance = float('inf')
    closest_name = "unknown"
    for color, name in block_colors.items():
        distance = np.linalg.norm(np.array(rgb) - np.array(color))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    logger.debug(f"RGB {rgb} 的最近颜色名称: {closest_name}, 最小距离: {min_distance}")
    return closest_name

# 提取纹理
def extract_texture(scene):
    logger.info("开始提取纹理")
    try:
        texture = None
        for geometry in scene.geometry.values():
            material = geometry.visual.material
            if isinstance(material, trimesh.visual.material.PBRMaterial):
                base_color_texture = material.baseColorTexture
                if base_color_texture:
                    texture = base_color_texture
                    texture_filename = os.path.join(os.getcwd(), 'extracted_texture.png')
                    texture.save(texture_filename)
                    logger.info(f"材质贴图已保存到: {texture_filename}")
                    return texture
        logger.warning("未找到纹理")
        return texture
    except Exception as e:
        logger.error(f"提取材质时出错: {e}")
        return None

# 改进的体素化函数，包含纹理和颜色校正
def improved_voxelize_with_texture(mesh, texture):
    logger.info("开始体素化处理")
    target_resolution = (32, 32, 32)
    bounds = mesh.bounds
    min_bounds, max_bounds = bounds
    extents = max_bounds - min_bounds
    max_extent = max(extents)
    pitch = max_extent / (max(target_resolution) - 1)

    voxel = mesh.voxelized(pitch=pitch)
    voxel_matrix_full = voxel.matrix

    full_shape = voxel_matrix_full.shape
    voxel_matrix = np.zeros(target_resolution, dtype=bool)
    scale = np.array(target_resolution) / np.array(full_shape)
    for i in range(target_resolution[0]):
        for j in range(target_resolution[1]):
            for k in range(target_resolution[2]):
                x = int(i / scale[0])
                y = int(j / scale[1])
                z = int(k / scale[2])
                if x < full_shape[0] and y < full_shape[1] and z < full_shape[2]:
                    voxel_matrix[i, j, k] = voxel_matrix_full[x, y, z]

    logger.info(f"体素分辨率: {voxel_matrix.shape}, 体素数量: {np.sum(voxel_matrix)}")

    color_matrix = np.zeros((*target_resolution, 3), dtype=np.uint8)

    if texture and hasattr(mesh.visual, 'uv'):
        logger.info("使用纹理生成颜色")
        texture_np = np.array(texture)
        h, w = texture_np.shape[:2]
        logger.debug(f"纹理尺寸: {w}x{h}")

        i, j, k = np.meshgrid(np.arange(32), np.arange(32), np.arange(32), indexing='ij')
        voxel_centers_all = min_bounds + (np.stack([i, j, k], axis=-1) + 0.5) * (max_bounds - min_bounds) / 32

        filled_indices = np.argwhere(voxel_matrix)
        voxel_centers_filled = voxel_centers_all[filled_indices[:,0], filled_indices[:,1], filled_indices[:,2]]

        closest_points, distances, face_indices = mesh.nearest.on_surface(voxel_centers_filled)
        triangles = mesh.triangles[face_indices]
        barycentric = trimesh.triangles.points_to_barycentric(triangles, closest_points)
        face_uvs = mesh.visual.uv[mesh.faces[face_indices]]
        uvs = np.einsum('ij,ijk->ik', barycentric, face_uvs)

        pixel_x = (uvs[:, 0] * (w - 1)).astype(int)
        pixel_y = ((1 - uvs[:, 1]) * (h - 1)).astype(int)
        pixel_x = np.clip(pixel_x, 0, w - 1)
        pixel_y = np.clip(pixel_y, 0, h - 1)

        colors = texture_np[pixel_y, pixel_x]
        logger.debug(f"采样颜色样本: {colors[:5]}")

        for idx, (i, j, k) in enumerate(filled_indices):
            color_matrix[i, j, k] = colors[idx]

        n_colors = 16
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(colors)
        quantized_colors = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
        logger.debug(f"量化颜色样本: {quantized_colors[:5]}")

        for idx, (i, j, k) in enumerate(filled_indices):
            color_matrix[i, j, k] = quantized_colors[idx]

        roof_mask = (filled_indices[:,1] >= 30)
        if np.any(roof_mask):
            roof_colors = colors[roof_mask]
            average_roof_color = np.mean(roof_colors, axis=0).astype(np.uint8)
            logger.info(f"屋顶平均颜色: {average_roof_color}")
            for idx in np.where(roof_mask)[0]:
                i, j, k = filled_indices[idx]
                color_matrix[i, j, k] = average_roof_color
    else:
        logger.warning("无纹理或UV数据，使用默认灰色 [128, 128, 128]")
        color_matrix[voxel_matrix] = [128, 128, 128]

    for i, j, k in filled_indices:
        current_color = color_matrix[i, j, k]
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni, nj, nk = i + di, j + dj, k + dk
                    if (0 <= ni < 32 and 0 <= nj < 32 and 0 <= nk < 32 and voxel_matrix[ni, nj, nk]):
                        neighbors.append(color_matrix[ni, nj, nk])
        if len(neighbors) > 0:
            neighbors_mean = np.mean(neighbors, axis=0)
            color_diff = np.linalg.norm(current_color - neighbors_mean)
            if color_diff > 20:
                logger.debug(f"校正颜色: ({i},{j},{k}) 从 {current_color} 到 {neighbors_mean.astype(np.uint8)}")
                color_matrix[i, j, k] = neighbors_mean.astype(np.uint8)

    logger.debug(f"颜色矩阵样本: {color_matrix[voxel_matrix][:5]}")
    return voxel_matrix, color_matrix

class ModelViewer:
    def __init__(self, block_colors):
        self.block_colors = block_colors
        logger.info(f"初始化 ModelViewer，block_colors 大小: {len(block_colors)}")

    def generate_voxel_text(self, voxel_matrix, color_matrix, output_file):
        logger.info(f"生成体素文本到: {output_file}")
        with open(output_file, 'w') as f:
            for y in range(voxel_matrix.shape[1]):
                f.write(f"--- Y Layer {y:02d} ---\n")
                for z in reversed(range(voxel_matrix.shape[2])):
                    line = []
                    for x in range(voxel_matrix.shape[0]):
                        if voxel_matrix[x, y, z]:
                            rgb = tuple(color_matrix[x, y, z])
                            name = get_color_name(rgb, self.block_colors)
                            line.append(name)
                        else:
                            line.append(".")
                    f.write(" ".join(line) + "\n")
                f.write("\n")
        logger.info("体素文本生成完成")

    def create_voxel_mesh(self, voxel_matrix, color_matrix, name=None):
        logger.info(f"创建体素网格: {name}")
        vertices = []
        faces = []
        colors = []
        cube_vertices = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ])
        cube_faces = np.array([
            [0, 1, 2], [0, 2, 3], [5, 4, 7], [5, 7, 6],
            [4, 0, 3], [4, 3, 7], [1, 5, 6], [1, 6, 2],
            [3, 2, 6], [3, 6, 7], [4, 5, 1], [4, 1, 0]
        ])
        for x, y, z in np.argwhere(voxel_matrix):
            vertex_offset = cube_vertices + np.array([x, y, z])
            face_offset = cube_faces + len(vertices)
            vertices.extend(vertex_offset)
            faces.extend(face_offset)
            color = color_matrix[x, y, z] / 255.0
            colors.extend([color] * 8)

        if vertices:
            voxel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
            logger.info(f"生成的体素网格 {name}: 顶点数={len(vertices)}, 面数={len(faces)}")
            return voxel_mesh
        logger.warning(f"警告: 体素网格 {name} 未生成")
        return None

    def view_model(self, glb_path, output_dir):
        logger.info(f"处理模型: {glb_path}")
        scene = trimesh.load(glb_path)
        meshes = [mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
        if not meshes:
            logger.error("模型中未找到网格")
            raise ValueError("No meshes found in the model")
        combined_mesh = trimesh.util.concatenate(meshes)
        logger.info(f"原始模型: 顶点数={len(combined_mesh.vertices)}, 面数={len(combined_mesh.faces)}")

        texture = extract_texture(scene)
        voxel_matrix, color_matrix = improved_voxelize_with_texture(combined_mesh, texture)

        voxel_mesh = self.create_voxel_mesh(voxel_matrix, color_matrix, "full_model")
        if voxel_mesh is None:
            logger.error("未能创建体素网格")
            raise ValueError("Failed to create voxel mesh")
        os.makedirs(output_dir, exist_ok=True)
        ply_path = os.path.join(output_dir, "voxel_model_32x32x32.ply")
        voxel_mesh.export(ply_path)
        logger.info(f"完整 32x32x32 体素模型已保存到: {ply_path}")

        voxel_text_path = os.path.join(output_dir, "voxel_model.txt")
        self.generate_voxel_text(voxel_matrix, color_matrix, voxel_text_path)
        logger.info(f"体素文本已保存到: {voxel_text_path}")

# 主逻辑
if __name__ == "__main__":
    # 指定 blockids.json 的路径（可根据实际情况修改）
    json_path = 'blockids.json'
    block_colors = load_block_colors(json_path)
    logger.info(f"已加载 {len(block_colors)} 个颜色条目。")
    if block_colors:
        logger.debug(f"样本条目: {list(block_colors.items())[:2]}")

    viewer = ModelViewer(block_colors)
    viewer.view_model('model.glb', 'output_directory')







    
