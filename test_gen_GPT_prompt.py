# encoding:utf-8
import numpy as np
import csv
from gpt_dialogue import Dialogue
from datetime import datetime
import openai
import time

SCANNET_DATA_ROOT="/share/data/ripl/scannet_raw/train/"

def read_csv_with_index(file_path):
    data = {}  # 用字典来保存索引后的数据
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取第一行作为header
        for index, row in enumerate(reader, start=1):  # 从第二行开始遍历内容，同时记录行索引
            data[index] = dict(zip(headers, row))  # 使用字典的方式，将header和对应行的内容组合成键值对
    # print(len(data))
    return data

def gen_GPT_prompt_sr3d(sr3d_data_index,scannet_data_root,to_print=True):
    """
    对于sr3d中的制定数据，返回prompt以及其他相关信息
    """
    
    # 读入sr3d数据集
    if to_print:
        print("sr3d_data_index:",sr3d_data_index)
    csv_path="sr3d_train_sampled.csv"
    data=read_csv_with_index(csv_path)[sr3d_data_index]
    
    # 读入scan_id
    scan_id=data["scan_id"]
    if to_print:
        print("scan_id:",scan_id)

    # 读入refered class and object ids
    target_class=data["instance_type"]
    target_id=data["target_id"]
    distractor_ids=eval(data["distractor_ids"])

    # 读入anchor classes and ids
    anchor_classes=data["anchors_types"]
    anchor_ids=eval(data["anchor_ids"])

    # 读入utterance
    utterance=data["utterance"]

    # 读入reference type
    reference_type=data["coarse_reference_type"]

    
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
    # if to_print:
        # print("object_related:",objects_related)

    # 生成prompt
    prompt=scan_id + " has objects in it and I'll give you some quantitative descriptions. " +\
    "All descriptions are in right-handed Cartesian coordinate system with x-y-z axes, " + \
    "where x represents left-right, y represents forward-backward, and z represents up-down. " + \
    "Objects are:\n"
    for obj in objects_related:
        line="A %s with id %s, its center position is %s, and its size in x,y,z direction is %s.\n" %(obj["label"],obj["id"],str(obj["quan_info"][0:3]),str(obj["quan_info"][3:]) )
        prompt=prompt+line
    line="Find the referred object in the following sentence and display its id only:\n"
    prompt=prompt+line+utterance+ '.' #".\nDo not display anything expect the id!"

    if to_print:
        print("--------------------------------------------")
        print("Generated prompt:\n"+prompt)
        print("--------------------------------------------")
        print("Right answer:",target_id)
    info=(scan_id,target_id,reference_type,utterance)
    return prompt,info


def dialogue_with_GPT(scannet_data_root=SCANNET_DATA_ROOT):
    # 创建dialogue实例
    config = {
        'model': 'gpt-4',
        # 'model': 'gpt-3.5-turbo',
        'temperature': 0,
        'top_p': 0.1,
        'max_tokens': 'inf',
        'system_message': '',
        # 'load_path': 'chats/dialogue_an apple.json',
        'save_path': 'chats',
        'debug': False
    }
    dialogue = Dialogue(**config)

    # 告知GPT背景信息
    background_prompt=\
    "I wiil describe some scenes and some objects in the scene, and I want you to analyse the spatial relationship of the objects in the scene and answer my questions." + \
    " All descriptions are in right-handed Cartesian coordinate system with x-y-z axes, " + \
    "where x represents left-right, y represents forward-backward, and z represents up-down. " + \
    "In each scene, I will tell you the center position and size in x-y-z direction of the objects."
    dialogue.call_openai(background_prompt)

    while True:
        # 生成sr3d中指定问题的prompt
        sr3d_line_number=input("Line number in sr3d_train.csv:")
        if sr3d_line_number == 'exit':
            break
        prompt,info=gen_GPT_prompt_sr3d(int(sr3d_line_number)-1, scannet_data_root)

        response=dialogue.call_openai(prompt)
        print("*******************************************")
        print("Response from GPT:")
        print(response['content'])
        print("*******************************************\n")

def evaluate_on_GPT(sr3d_line_numbers):

    assert np.max(sr3d_line_numbers)<=65845,"line number %s > 65845!"%str(np.max(sr3d_line_numbers))
    assert np.min(sr3d_line_numbers)>=2,"line number %s < 2!"%str(np.max(sr3d_line_numbers))
    
    # 创建结果表格，格式如下
    # sr3d_line_number # scan_id # reference_type # target_id # answer_id # is_correct #
    sr3d_len=len(sr3d_line_numbers)
    results_table=np.zeros([sr3d_len,6],dtype='<U21')

    # 创建dialogue实例
    config = {
        'model': 'gpt-4',
        # 'model': 'gpt-3.5-turbo',
        'temperature': 0,
        'top_p': 0.1,
        'max_tokens': 'inf',
        'system_message': '',
        # 'load_path': 'chats/dialogue_an apple.json',
        'save_path': 'chats',
        'debug': False
    }
    # dialogue = Dialogue(**config)
    # # 告知GPT背景信息
    # background_prompt=\
    # "I wiil describe some scenes and some objects in the scene, and I want you to analyse the spatial relationship of the objects in the scene and answer my questions." + \
    # " All descriptions are in right-handed Cartesian coordinate system with x-y-z axes, " + \
    # "where x represents left-right, y represents forward-backward, and z represents up-down. " + \
    # "In each scene, I will tell you the center position and size in x-y-z direction of the objects."
    # dialogue.call_openai(background_prompt)
    # print("--------------------------------------------")
    # print("background_prompot:\n"+background_prompt)
    # print("--------------------------------------------")

    # 遍历给定的数据集部分
    for idx,line_number in enumerate(sr3d_line_numbers):
        print("Processing sr3d line %d, %d/%d."%(line_number,idx+1,sr3d_len))
        
        # 生成prompt
        prompt,info=gen_GPT_prompt_sr3d(line_number-1, SCANNET_DATA_ROOT,to_print=False)
        scan_id,target_id,reference_type,utterance=info

        # 获取GPT回复结果
        dialogue = Dialogue(**config)
        while True:
            try:
                answer_id=dialogue.call_openai(prompt)["content"]
                break
            except openai.error.RateLimitError as r:
                print("OpenAI RateLimitError!")
                print(r)
                time.sleep(1)
            except openai.error.ServiceUnavailableError as r:
                print("OpenAI ServiceUnavailableError!")
                print(r)
                time.sleep(1)
            except Exception as r:
                print("Something Unkown was wrong!")
                print(r)
                time.sleep(1)

        # 在表格中记录相关信息
        results_table[idx][0]=str(line_number)
        results_table[idx][1]=str(scan_id)
        results_table[idx][2]=str(reference_type)
        results_table[idx][3]=str(target_id)
        results_table[idx][4]=str(answer_id)
        if str(answer_id)==str(target_id):
            print("answer correct.")
            results_table[idx][5]=str(True)
        else:
            print("answer wrong!")
            results_table[idx][5]=str(False)
            print("Error info:\nutterance: %s\ntarget_id:%s\nanswer_id:%s"%(utterance,str(target_id),str(answer_id)))
    
    # 保存结果表格
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    save_path="./eval_results/%s.npy"%formatted_time
    np.save(save_path, results_table)
    print("results saved to: %s"%save_path)
    return save_path

def analyse_result(result_path):
    """
    分析保存好的结果npy文件
    # sr3d_line_number # scan_id # reference_type # target_id # answer_id # is_correct #
    """
    result=np.load(result_path,allow_pickle=True)
    # 统计数据
    accuracy_count={
        "count_total":0,"correct_count_total":0,
        "count_horizontal":0,"correct_count_horizontal":0,
        "count_vertical":0,"correct_count_vertical":0,
        "count_support":0,"correct_count_support":0,
        "count_between":0,"correct_count_between":0,
        "count_allocentric":0,"correct_count_allocentric":0
    }
    for result_line in result:
        reference_type=result_line[2]
        accuracy_count["count_total"]+=1
        accuracy_count["count_"+reference_type]+=1
        if result_line[5]=="True":
            accuracy_count["correct_count_total"]+=1
            accuracy_count["correct_count_"+reference_type]+=1

    #分析正确率 
    for name in ["total","horizontal","vertical","support","between","allocentric"]:
        print(name+" accuracy:")
        correct=accuracy_count["correct_count_"+name]
        total=accuracy_count["count_"+name]
        percentage = "-" if total==0 else correct/total*100
        print(str(percentage)+"%% (%d/%d)"%(correct,total))
    




# gen_GPT_prompt_sr3d(57575,"/share/data/ripl/scannet_raw/train/") #第一个index是在excel中查看到的行数-1

# dialogue_with_GPT("/share/data/ripl/scannet_raw/train/")

lines=np.arange(10000,10050) #sr3d中要测试的行数
lines=np.arange(2,330)
result_path=evaluate_on_GPT(lines)
analyse_result(result_path)

