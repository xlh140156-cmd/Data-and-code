import xarray as xr
import rasterio
import numpy as np
from rasterio.transform import from_origin
import os


def nc_to_tiff(nc_file, output_dir, var_name='tcwv'):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取NetCDF文件
    ds = xr.open_dataset(nc_file)

    # 提取变量数据
    data = ds[var_name]

    # 获取经纬度信息
    lons = ds['longitude'].values
    lats = ds['latitude'].values

    # 计算栅格分辨率
    lon_res = abs(lons[1] - lons[0])
    lat_res = abs(lats[1] - lats[0])

    # 获取时间维度
    times = ds['valid_time'].values

    # 遍历每个时间步
    for i, time in enumerate(times):
        # 提取当前时间步的数据
        data_slice = data[i].values

        # 处理缺失值（如果有）
        data_slice = np.where(np.isnan(data_slice), -9999, data_slice)

        # 构造输出文件名（使用时间戳，避免非法字符）
        time_str = str(time).replace(':', '-').replace(' ', '_')
        output_file = os.path.join(output_dir, f'tcwv_{time_str}.tiff')

        # 定义GeoTIFF的元数据
        transform = from_origin(lons.min(), lats.max(), lon_res, lat_res)
        metadata = {
            'driver': 'GTiff',
            'height': data_slice.shape[0],
            'width': data_slice.shape[1],
            'count': 1,
            'dtype': data_slice.dtype,
            'crs': 'EPSG:4326',
            'transform': transform,
            'nodata': -9999
        }

        # 保存为TIFF文件
        with rasterio.open(output_file, 'w', **metadata) as dst:
            dst.write(data_slice, 1)

        print(f'Saved {output_file}')

    # 关闭数据集
    ds.close()


if __name__ == '__main__':
    # 输入文件和输出目录
    nc_file = r'C:\Users\Administrator\Desktop\20201.nc'  # 替换为你的NetCDF文件路径
    output_dir = r'F:\111output_tiffs'  # 替换为你想保存TIFF的目录
    nc_to_tiff(nc_file, output_dir)