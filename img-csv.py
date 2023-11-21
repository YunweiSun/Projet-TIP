import os
import csv

def list_files_in_folder(folder_path):
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            # filename = './isic-2020-resized/test-resized/ISIC_1933796.jpg'
            # filename = filename.split("/")[2].split(".")[0]
            filename = filename.split(".")[0]
            files.append(filename)
    return files

def save_to_csv(file_list, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File Name'])  # 写入表头
        for file in file_list:
            csv_writer.writerow([file])

# 指定文件夹路径
folder_path = "./isic-2020-resized/test-resized/"

# 获取文件列表
files_list = list_files_in_folder(folder_path)
print

# 将文件名保存到 CSV 文件中
csv_filename = "./isic-2020-resized/test.csv"
save_to_csv(files_list, csv_filename)

print(f"File names saved to {csv_filename}")
