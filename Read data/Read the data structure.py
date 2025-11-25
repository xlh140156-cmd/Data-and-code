import numpy as np
import rasterio
import glob
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def sen_slope_pixelwise_row(args):
    """
    计算单行像素的Sen's slope
    参数：
        args: (row_index, data_row, time_points)
            - row_index: 行索引
            - data_row: 2D数组 (time, width)
            - time_points: 时间点数组
    返回：
        (row_index, slopes_row): 行索引和斜率数组
    """
    row_index, data_row, time_points = args
    width = data_row.shape[1]
    slopes_row = np.full(width, np.nan, dtype=np.float32)

    for j in range(width):
        pixel_series = data_row[:, j]
        if np.any(np.isnan(pixel_series)) or np.all(pixel_series == pixel_series[0]):
            continue
        n = len(time_points)
        pixel_slopes = []
        for t1 in range(n):
            for t2 in range(t1 + 1, n):
                if time_points[t2] != time_points[t1]:
                    slope = (pixel_series[t2] - pixel_series[t1]) / (time_points[t2] - time_points[t1])
                    pixel_slopes.append(slope)
        if pixel_slopes:
            slopes_row[j] = np.median(pixel_slopes)

    return row_index, slopes_row


def process_tiff_stack_blockwise(tiff_dir, output_path, time_points, block_size=10, num_processes=4):
    """
    分块处理GeoTIFF文件栈，计算全球Sen's slope
    参数：
        tiff_dir: GeoTIFF文件目录
        output_path: 输出GeoTIFF路径
        time_points: 时间点数组
        block_size: 每次处理的行数
        num_processes: 并行进程数
    """
    # 获取所有GeoTIFF文件
    tiff_files = sorted(glob.glob(os.path.join(tiff_dir, 'tcwv.nc*.tif')))
    if not tiff_files:
        raise ValueError("未找到GeoTIFF文件")
    if len(tiff_files) != 780:
        raise ValueError(f"预期780个文件，实际找到 {len(tiff_files)} 个")
    if len(tiff_files) != len(time_points):
        raise ValueError(f"文件数量 ({len(tiff_files)}) 与时间点数量 ({len(time_points)}) 不匹配")
    print(f"处理 {len(tiff_files)} 个文件: {tiff_files[0]} 到 {tiff_files[-1]}")

    # 读取元数据
    with rasterio.open(tiff_files[0]) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width

    # 更新输出元数据
    meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    # 检查输出文件是否可写
    try:
        if os.path.exists(output_path):
            print(f"检测到现有输出文件 {output_path}，尝试删除...")
            os.remove(output_path)
    except PermissionError:
        raise PermissionError(f"无法删除 {output_path}，请确保文件未被其他程序占用或具有写入权限")

    # 初始化输出文件
    with rasterio.open(output_path, 'w', **meta) as dst:
        # 分块处理
        for start_row in tqdm(range(0, height, block_size), desc="处理行块"):
            end_row = min(start_row + block_size, height)
            block_height = end_row - start_row

            # 读取当前块
            data_block = np.zeros((len(tiff_files), block_height, width), dtype=np.float32)
            for i, file in enumerate(tiff_files):
                with rasterio.open(file) as src:
                    window = rasterio.windows.Window(0, start_row, width, block_height)
                    data = src.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)
                    data_block[i] = data

            # 并行计算Sen's slope
            slopes_block = np.full((block_height, width), np.nan, dtype=np.float32)
            with Pool(processes=num_processes) as pool:
                tasks = [(i, data_block[:, i, :], time_points) for i in range(block_height)]
                results = list(tqdm(pool.imap(sen_slope_pixelwise_row, tasks), total=block_height, desc="计算行"))
                for row_index, slopes_row in results:
                    slopes_block[row_index] = slopes_row

            # 写入当前块
            window = rasterio.windows.Window(0, start_row, width, block_height)
            dst.write(slopes_block, 1, window=window)

    # 输出统计
    with rasterio.open(output_path) as src:
        slopes = src.read(1, masked=True).filled(np.nan)
        print(f"Sen's slope计算完成！")
        print(f"  输出文件: {output_path}")
        print(f"  像素尺寸: {width} x {height}")
        print(
            f"  斜率范围: min={np.nanmin(slopes):.6f}, max={np.nanmax(slopes):.6f}, mean={np.nanmean(slopes):.6f} kg/m² per year")
        print(f"  非NaN像素数: {np.sum(~np.isnan(slopes))} / {height * width}")


if __name__ == '__main__':
    # 设置路径和时间点
    tiff_dir = r'E:\qudong\TCWV\22'
    output_path = r'E:\qudong\TCWV\22\sen_slope_result_1959_2023.tif'
    # 生成1959年1月至2023年12月的月度时间点
    dates = pd.date_range(start='1959-01-01', end='2023-12-31', freq='ME')
    time_points = np.array([1959 + i / 12.0 for i in range(len(dates))])
    print(f"时间点范围: {time_points[0]:.4f} 到 {time_points[-1]:.4f} (共 {len(time_points)} 点)")

    # 设置并行进程数（可根据性能调整）
    num_processes = min(8, cpu_count())  # 尝试8核，16核机器可支持
    print(f"使用 {num_processes} 个CPU核心")

    # 运行分块处理
    process_tiff_stack_blockwise(tiff_dir, output_path, time_points, block_size=10, num_processes=num_processes)