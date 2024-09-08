import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import argparse



def merge_csv_files(directory):
    # 获取目录中的所有CSV文件
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # 创建一个空的DataFrame来存储合并后的数据
    merged_df = pd.DataFrame()
    
    # 遍历每个CSV文件
    for file in files:
        # 读取CSV文件
        df = pd.read_csv(os.path.join(directory, file))
        
        # 获取Value列的数值，并添加到新表格中
        value_column = df['Value']
        # 使用文件名区分新表格中每一列数据
        column_name = file.replace('.csv', '')
        merged_df[column_name] = value_column
    
    # 分别绘制每列数据的折线图
    # for column in merged_df.columns:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(merged_df[column], label=column)
    #     plt.title(f'{column}')
    #     plt.xlabel('index')
    #     plt.ylabel('Value')
    #     plt.legend()
    #     plt.show()
    
    # 返回合并后的DataFrame
    return merged_df

def save_csv(df, csv_path):
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加命令行参数
    parser.add_argument('--dir', type=str, help='directory')
    parser.add_argument('--csv_path', type=str, default='./merged.csv', help='output file')

    arg = parser.parse_args()

    merged_df = merge_csv_files(arg.dir)
    save_csv(merged_df, arg.csv_path)
    print("merged finished!")
