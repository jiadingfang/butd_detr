# encoding:utf-8
import numpy as np
import csv


def read_csv_with_index(file_path):
    data = {}  # 用字典来保存索引后的数据
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取第一行作为header
        for index, row in enumerate(reader, start=1):  # 从第二行开始遍历内容，同时记录行索引
            data[index] = dict(zip(headers, row))  # 使用字典的方式，将header和对应行的内容组合成键值对

    return data

def gen_GPT_prompt_sr3d(sr3d_data_index,scannet_data_root):
    # 读入sr3d数据集
    print("sr3d_data_index:",sr3d_data_index)
    csv_path="sr3d_train.csv"
    data=read_csv_with_index(csv_path)[sr3d_data_index]
    
    # 读入scan_id
    scan_id=data["scan_id"]
    print("scan_id:",scan_id)

    # 读入refered class and object ids
    target_class=data["reference_type"]
    target_id=data["target_id"]
    distractor_ids=eval(data["distractor_ids"])

    # 读入anchor classes and ids
    anchor_classes=data["anchors_types"]
    anchor_ids=eval(data["anchor_ids"])

    # 读入utterance
    utterance=data["utterance"]

    
    # 读入事先准备好的物体信息，即npy文件
    npy_path=scannet_data_root+"/objects_info/objects_info_"+scan_id+".npy"
    objects_info=np.load(npy_path,allow_pickle=True)

    # 整合所有物体信息
    objects_related=[]
    objects_related.append(objects_info[int(target_id)])
    for id in distractor_ids:
        objects_related.append(objects_info[int(id)])
    for id in anchor_ids:
        objects_related.append(objects_info[int(id)])
    print("object_related:",objects_related)

    # 生成prompt
    prompt=scan_id + " has Cartesian coordinate system with x-y-z axes. There are several objects in the scene, and I will tell you the center position and size in x-y-z direction of the objects:\n"
    for obj in objects_related:
        line="A %s with id %s, its center position is %s, and its size in x,y,z direction is %s.\n" %(obj["label"],obj["id"],str(obj["quan_info"][0:3]),str(obj["quan_info"][3:]) )
        prompt=prompt+line
    line="Plese find the referred object and its id in the following sentence:\n"
    prompt=prompt+line+utterance+"."

    print("--------------------------------------------")
    print("Generated prompt:\n"+prompt)
    print("--------------------------------------------")
    print("Right answer:",target_id)
    return prompt


gen_GPT_prompt_sr3d(57575,"/share/data/ripl/scannet_raw/train/") #第一个index是在excel中查看到的行数-1
