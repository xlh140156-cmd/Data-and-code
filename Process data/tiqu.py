import netCDF4 as nc
import numpy as np
from tqdm import tqdm

# 文件路径
input_file = r'C:\Users\Administrator\Downloads\tcwv.nc'

# 打开 NetCDF 文件
dataset = nc.Dataset(input_file, 'r')

# 获取 t2m 数据变量信息
t2m_var = dataset.variables['tcwv']
time_steps = t2m_var.shape[0]  # 时间维度大小

# 循环保存每个时间步的数据
for i in tqdm(range(time_steps), desc="Processing time steps"):
    # 提取当前时间步的 t2m 数据并展平
    t2m_slice = t2m_var[i, :, :].flatten()

    # 检查是否是 MaskedArray，如果是则填充
    if isinstance(t2m_slice, np.ma.MaskedArray):
        t2m_slice = t2m_slice.filled()

    # 保存为单独的 .npy 文件
    np.save(f't2m_time_step_{i}.npy', t2m_slice)

# 关闭 Dataset
dataset.close()
