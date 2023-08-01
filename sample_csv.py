import pandas as pd
import random

def random_sample_csv(input_file, output_file, sample_percent=1):
    """
    随机采样CSV文件的一部分，并将采样结果保存到新的CSV文件中。
    
    参数：
    input_file (str): 输入CSV文件的路径和文件名。
    output_file (str): 输出采样结果的CSV文件路径和文件名。
    sample_percent (float): 采样的百分比，默认为1（1%）。
    """
    # 使用pandas库读取输入CSV文件
    df = pd.read_csv(input_file)

    # 计算要采样的行数
    total_rows = df.shape[0]
    sample_size = int(total_rows * sample_percent / 100)

    # 使用random.sample函数随机选取要采样的行的索引
    sample_indexes = random.sample(range(total_rows), sample_size)

    # 使用iloc函数根据索引获取采样的行
    sampled_df = df.iloc[sample_indexes]

    # 将采样结果保存到新的CSV文件中
    sampled_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # 输入文件名和输出文件名
    input_csv_file = "sr3d_train.csv"
    output_csv_file = "sr3d_train_sampled.csv"

    # 调用函数进行1%的随机采样
    random_sample_csv(input_csv_file, output_csv_file, sample_percent=0.5)

    print("随机采样完成并已保存到文件：", output_csv_file)
