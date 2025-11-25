import numpy as np
import pandas as pd
import os
import logging
import warnings
import gc

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 文件路径
OUTPUT_DIR_TEST = r'F:\空间预测\程序\分解预测\D3\CESHSHUJUI11'
OUTPUT_DIR_TRAIN = r'F:\空间预测\程序\分解预测\D3\XUNLIANSHUJU22'
OUTPUT_NPY_PATH = r'F:\空间预测\predicted_DD3333_coefficients_1980_2023.npy'

# 参数
expected_time_steps = 516
test_rows = 104
train_rows = expected_time_steps - test_rows  # 412
grid_points_expected = 8153  # 预期网格点

# 1. 检查目录
if not os.path.exists(OUTPUT_DIR_TEST):
    logger.error(f"测试集目录不存在: {OUTPUT_DIR_TEST}")
    raise FileNotFoundError(f"测试集目录不存在: {OUTPUT_DIR_TEST}")
if not os.path.exists(OUTPUT_DIR_TRAIN):
    logger.error(f"训练集目录不存在: {OUTPUT_DIR_TRAIN}")
    raise FileNotFoundError(f"训练集目录不存在: {OUTPUT_DIR_TRAIN}")

# 2. 获取网格点文件
test_files = [f for f in os.listdir(OUTPUT_DIR_TEST) if f.endswith('.xlsx') and f.startswith('grid_d3_')]
if not test_files:
    logger.error(f"测试集目录 {OUTPUT_DIR_TEST} 中没有找到以 'grid_d3_' 开头的 Excel 文件")
    raise FileNotFoundError("没有找到测试集 Excel 文件")
grid_points = len(test_files)
if grid_points != grid_points_expected:
    logger.warning(f"找到 {grid_points} 个网格点文件，少于预期 {grid_points_expected}")
logger.info(f"找到 {grid_points} 个网格点的测试文件")

# 3. 初始化预测数据数组
predicted_data = np.zeros((expected_time_steps, grid_points))
logger.info(f"初始化 predicted_data 形状: {predicted_data.shape}")

# 4. 分批处理网格点
batch_size = 200
for batch_start in range(0, grid_points, batch_size):
    batch_end = min(batch_start + batch_size, grid_points)
    logger.info(f"处理网格点批次 {batch_start + 1} 到 {batch_end}/{grid_points}")

    for grid_idx in range(batch_start, batch_end):
        grid_num = grid_idx + 1
        test_file = f'grid_d3_{grid_num}_test_results.xlsx'
        train_file = f'grid_d3_{grid_num}_train_results.xlsx'
        test_path = os.path.join(OUTPUT_DIR_TEST, test_file)
        train_path = os.path.join(OUTPUT_DIR_TRAIN, train_file)

        try:
            # 检查文件
            if not os.path.exists(test_path) or not os.path.exists(train_path):
                logger.warning(f"网格点 {grid_num} 的测试或训练文件缺失，跳过")
                continue

            # 读取数据
            train_df = pd.read_excel(train_path)
            test_df = pd.read_excel(test_path)

            if 'Predicted' not in train_df.columns or 'Predicted' not in test_df.columns:
                logger.warning(f"网格点 {grid_num} 的文件缺少 'Predicted' 列，跳过")
                continue

            train_pred = train_df['Predicted'].values
            test_pred = test_df['Predicted'].values

            # 验证行数
            if len(test_pred) != test_rows:
                logger.warning(f"网格点 {grid_num} 测试集行数 {len(test_pred)} 不等于预期 {test_rows}，跳过")
                continue
            if len(train_pred) != train_rows:
                logger.info(f"网格点 {grid_num} 训练集行数 {len(train_pred)} 不等于预期 {train_rows}，调整")
                if len(train_pred) > train_rows:
                    train_pred = train_pred[:train_rows]
                else:
                    train_pred = np.pad(train_pred, (0, train_rows - len(train_pred)), mode='constant')

            # 拼接
            combined_pred = np.concatenate([train_pred, test_pred])

            # 调整长度
            if len(combined_pred) > expected_time_steps:
                combined_pred = combined_pred[:expected_time_steps]
                logger.info(f"网格点 {grid_num} 预测值截断到 {expected_time_steps} 行")
            elif len(combined_pred) < expected_time_steps:
                combined_pred = np.pad(combined_pred, (0, expected_time_steps - len(combined_pred)), mode='constant')
                logger.info(f"网格点 {grid_num} 预测值填充到 {expected_time_steps} 行")

            predicted_data[:, grid_idx] = combined_pred
            logger.info(f"网格点 {grid_num} 已处理，预测值长度: {len(combined_pred)}")
            logger.info(f"处理进度: {(grid_idx + 1) / grid_points * 100:.2f}%")

        except Exception as e:
            logger.error(f"处理网格点 {grid_num} 失败: {str(e)}")
            continue

    # 清理内存
    gc.collect()

# 5. 保存为 .npy
np.save(OUTPUT_NPY_PATH, predicted_data)
logger.info(f"预测数据已保存为: {OUTPUT_NPY_PATH}")

# 6. 验证
saved_data = np.load(OUTPUT_NPY_PATH)
logger.info(f"验证 .npy 文件形状: {saved_data.shape}")
if saved_data.shape != (expected_time_steps, grid_points):
    logger.error(f"保存的 .npy 文件形状 {saved_data.shape} 与预期 {(expected_time_steps, grid_points)} 不匹配！")
else:
    logger.info("保存的 .npy 文件形状正确")