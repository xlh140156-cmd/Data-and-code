import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

# 1. 定义路径和参数
npy_file = r'F:\new_predicted_water_vapor.npy'  # .npy 文件
lat_lon_file = r'\JFBLLLL\tcwv\lat_lon.npy'  # 经纬度文件
output_dir = r'F:/tiff_output/'  # TIFF 输出目录
height = 180  # 网格高度（纬度）
width = 360  # 网格宽度（经度）
selected_time_steps = [412, 413, 414]  # 时间步索引，对应 month_425, month_426, month_427

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 加载 .npy 文件
data = np.load(npy_file)
print("加载 .npy 文件，形状：", data.shape)

# 3. 加载 lat_lon.npy 获取地理信息
lat_lon = np.load(lat_lon_file)
lats = lat_lon[:, 0]  # 纬度
lons = lat_lon[:, 1]  # 经度
west = lons[0]  # 0.375
north = lats[0]  # 89.625
x_res = 1.0  # 经度分辨率
y_res = -1.0  # 纬度分辨率（北到南）

# 创建仿射变换矩阵
transform = from_origin(west, north, x_res, y_res)

# 坐标参考系
crs = 'EPSG:4326'

# 4. 生成选定时间步的 GeoTIFF
for t in selected_time_steps:
    month = t + 13  # 转换为 month_XXX（t=412 -> month_425）
    data_t = data[t, :]  # 提取时间步数据
    if np.all(np.isnan(data_t)):
        print(f"时间步 {month}: 全为 NaN，跳过")
        continue

    # 重塑为一维网格点数据为二维数组，并翻转纬度方向
    data_2d = np.flipud(data_t.reshape(height, width))

    # 定义 GeoTIFF 元数据
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,  # 单波段
        'dtype': data_2d.dtype,  # float64
        'crs': crs,
        'transform': transform,
        'nodata': np.nan  # NaN 为无效值
    }

    # 保存为 TIFF 文件
    output_file = os.path.join(output_dir, f'predicted_water_vapor_month_{month}.tif')
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(data_2d, 1)  # 写入波段 1
        dst.update_tags(time=f"Month {month} (1980-2023)")

    print(f"已保存 TIFF 文件: {output_file}")
    print(f"时间步 {month} 数据范围: {np.nanmin(data_2d)} {np.nanmax(data_2d)}")
    print(f"TIFF 元数据: {profile}")

print("\n生成完成！请检查 F:/tiff_output/ 的 TIFF 文件效果。")