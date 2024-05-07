import argparse
import json
import logging
import sys
sys.path.append('llms/')
from remote.__init__ import RemoteLLMs
from remote.ChatGPT import ChatGPTLLM
from remote.ChatGLM2 import ChatGLM2_6BWrapper
import pandas as pd
import time
import re
import os
def set_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于from utils import set_logger写入日志文件
    # 增加对logger.info中文乱码的处理
    # python3.9以上才可以设置encoding
    if sys.version_info.major == 3 and sys.version_info.minor >= 9:
        file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    else:
        file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger
def is_json_path(path):  
    return path.endswith('.json') 
class ScoringAgent:
    def __init__(self, logger, llm_model: RemoteLLMs, task_name: str, language,background_material,
                 src_term, tgt_term, in_context_examples=[], more_guidance=[], more_task_definition=[]):
        self.logger = logger
        self.llm_model = llm_model
        self.prompt_pattern = """
[任务描述] 
这是一个根据给定的背景材料生成问题和答案的任务。所有的[输入]都是{{语言}}的。
{{更多任务定义}}
你需要根据提供的背景生成问题和相应的答案。
你的输出应该遵循[输出格式]。

[指导]
你应该严格遵循我的指导：
1. 每个答案都应该相关且信息丰富。
2. 你应该严格遵循给定的输出格式，不能输出其他信息。
{{更多指导}}
如果你违反我的指导，将会受到惩罚。

[输出格式]
你的输出应严格遵循这个格式，并且可以直接由 Python 解码：
'''
{{输出}}
'''
示例：
{
    '小明发现了什么？': '小明在阁楼的旧箱子里发现了一块古老的怀表。',
}
[输入]
'''
{
    背景材料: {{SRC_VALUE}}
}
'''
 
"""

        if len(in_context_examples) > 0:
            more_guidance.append('为了帮助你的判断，在[示例]中提供了一些示例。')
            in_context_prompt = ["[示例]", "'''"]
            in_context_prompt.append(json.dumps(in_context_examples, ensure_ascii=False, indent=4))
            in_context_prompt.append("'''")
            in_context_prompt = '\n'.join(in_context_prompt)
        else:
            in_context_prompt = ""

        tmp = []
        for idx, guidance in enumerate(more_guidance):
            tmp.append('%s. %s' % (idx + 4, guidance))
        more_guidance = '\n'.join(tmp)

        # 给定输入的模板格式
        input_format = json.dumps({"%SRC%": '%SRC_VALUE%'}, ensure_ascii=False, indent=4)

        self.meta_dict = {
            "{{语言}}": language,
            "{{输出}}": "{'问题1': '答案1', '问题2': '答案2'}",
            "{{输入}}": input_format,
            "{{任务名称}}": task_name,
            "{{更多指导}}": more_guidance,
            "{{更多任务定义}}": '\n'.join(more_task_definition),
            "{{在上下文中的示例}}": in_context_prompt,
            "{{SRC_VALUE}}":background_material
        }

    def generate_questions_answers(self, background_material):
        llm_model = self.llm_model
        repeat_times = -1

        while True:
            repeat_times += 1
            if repeat_times >= llm_model.max_retries:
                break

            # 构造提示，包括背景材料
            prompt = llm_model.fit_case(pattern=self.prompt_pattern, data=background_material, meta_dict=self.meta_dict)
            contexts = llm_model.create_prompt(prompt)

            # 请求问题-答案对
            results = llm_model.request_llm(contexts, repeat_times=repeat_times)

            if results is not None and results[-1]['role'] == 'assistant':
                generated_questions_answers = self.extract_questions_answers(results[-1]['content'])
                if generated_questions_answers is not None and self.validate_questions_answers(generated_questions_answers):
                    return prompt, generated_questions_answers

        return None

    def extract_questions_answers(self, last_response: str):
        try:
            return last_response
        except Exception as e:
            raise e
            return None

    def validate_questions_answers(self, generated_questions_answers):
        try:
            # 检查问答对的格式是否正确
            if isinstance(generated_questions_answers, str):
                # 检查问答对的内容是否符合预期
                if re.search(r"\'(.*?)\'\s*:\s*\'(.*?)\'", generated_questions_answers):
                    return True
            return False
        except Exception as e:
            # 如果发生异常，也返回 False
            return False
import threading
def make_qa(input_lujing):
    print(input_lujing)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path1", type=str, default="llms/remote/configs/wsx_gpt35.json")
    parser.add_argument("--config_path2", type=str, default="llms/remote/configs/wyh_gpt35.json")
    args = parser.parse_args()
    logger = set_logger("tmp.log")
    chat_gpt1 = ChatGPTLLM(args.config_path1)
    chat_gpt2 = ChatGPTLLM(args.config_path2)
    task_name = "基于背景材料的问答"
    src_term = "背景材料"
    tgt_term = "生成的问题和答案"
    language = "中文"
    more_guidance = ['每个问题都应该基于给定的背景材料进行表述。']
    # 提示用户输入背景材料
    text = pd.read_json(input_lujing)
    with open("llms/applications/duandian.txt","r",encoding="utf-8") as file:
        lines = file.readlines()
    m = int(lines[0])
    question1_list = []
    answer1_list = []
    def xiancheng1(j,chat_gpt666):
        if not xiancheng1.executed:
            for i in range(j,j+3):
                print(i)
                time.sleep(5)
                with open("llms/applications/duandian.txt","w",encoding="utf-8") as file:
                    file.write(str(i))
                background_material = text["text"][i]
                score_agent = ScoringAgent(logger, chat_gpt666, task_name, language=language,background_material=background_material, src_term=src_term, tgt_term=tgt_term,
                                        more_guidance=more_guidance)
                
                prompt, generated_questions_answers = score_agent.generate_questions_answers({"background_material": background_material})
                tt = str(i+1)
                a = generated_questions_answers
                a = a.replace("'","\"")
                a = a.replace("`","")
                pattern = r'"(?![^{]*})'  
                a = re.sub(pattern, '', a) 
                a = a.replace("'",'"')
                json_pattern = r'{.*?}'
                a = re.search(json_pattern, a, re.DOTALL)
                a = a.group()
                try:
                    questions_and_answers = json.loads(a)
                except json.JSONDecodeError:
                    with open('error.txt', 'a', encoding='utf-8') as file:
                            file.write("第"+ tt +"个文章无法处理：\n") 
                    continue
                for question, answer in questions_and_answers.items():  
                    if question not in question1_list:
                        question1_list.append(question)
                        answer1_list.append(answer)
            xiancheng1.executed = True
        return question1_list,answer1_list
    xiancheng1.executed = False
    question2_list = []
    answer2_list = []
    def xiancheng2(j,chat_gpt666):
        if not xiancheng2.executed:
            for i in range(j,j+3):
                time.sleep(5)
                with open("llms/applications/duandian.txt","w",encoding="utf-8") as file:
                    file.write(str(i))
                background_material = text["text"][i]
                score_agent = ScoringAgent(logger, chat_gpt666, task_name, language=language,background_material=background_material, src_term=src_term, tgt_term=tgt_term,
                                        more_guidance=more_guidance)
                
                prompt, generated_questions_answers = score_agent.generate_questions_answers({"background_material": background_material})
                tt = str(i+1)
                a = generated_questions_answers
                a = a.replace("'","\"")
                a = a.replace("`","")
                pattern = r'"(?![^{]*})'  
                a = re.sub(pattern, '', a) 
                a = a.replace("'",'"')
                json_pattern = r'{.*?}'
                a = re.search(json_pattern, a, re.DOTALL)
                a = a.group()
                try:
                    questions_and_answers = json.loads(a)
                except json.JSONDecodeError:
                    with open('error.txt', 'a', encoding='utf-8') as file:
                            file.write("第"+ tt +"个文章无法处理：\n") 
                    continue
                print(questions_and_answers)
                for question, answer in questions_and_answers.items():  
                    if question not in question2_list:
                        question2_list.append(question)
                        answer2_list.append(answer)
                xiancheng2.executed = True
        return question2_list,answer2_list
    xiancheng2.executed = False
    thread1 = threading.Thread(target=xiancheng1,args=(0,chat_gpt1))
    thread2 = threading.Thread(target=xiancheng2,args=(5,chat_gpt2))
    thread1.start()
    thread2.start()
    thread1.join()
    question_list1, answer_list1 = xiancheng1(0,chat_gpt1)
    thread2.join()
    question_list2, answer_list2 = xiancheng2(5,chat_gpt2)
    question_list = question_list1 + question_list2
    answer_list  =answer_list1 + answer_list2
    return question_list,answer_list
def make_now(input_text):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="llms/remote/configs/wsx_gpt35.json")
    args = parser.parse_args()
    logger = set_logger("tmp.log")
    chat_gpt = ChatGPTLLM(args.config_path)
    task_name = "基于背景材料的问答"
    src_term = "背景材料"
    tgt_term = "生成的问题和答案"
    language = "中文"
    more_guidance = ['每个问题都应该基于给定的背景材料进行表述。']
    # 提示用户输入背景材料

    question_list = []
    answer_list = []
    background_material = input_text
    score_agent = ScoringAgent(logger, chat_gpt, task_name, language=language,background_material=background_material, src_term=src_term, tgt_term=tgt_term,
                            more_guidance=more_guidance)
    time.sleep(8)
    prompt, generated_questions_answers = score_agent.generate_questions_answers({"background_material": background_material})
    a = generated_questions_answers
    a = a.replace("'","\"")
    a = a.replace("`","")
    pattern = r'"(?![^{]*})'  
    a = re.sub(pattern, '', a) 
    a = a.replace("'",'"')
    json_pattern = r'{.*?}'
    a = re.search(json_pattern, a, re.DOTALL)
    a = a.group()
    questions_and_answers = json.loads(a)
    for question, answer in questions_and_answers.items():  
        question_list.append(question)
        answer_list.append(answer)
    return question_list,answer_list

def make_api(input_text):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="llms/remote/configs/wsx_gpt35.json")
    args = parser.parse_args()
    logger = set_logger("tmp.log")
    chat_gpt = ChatGPTLLM(args.config_path)
    task_name = "基于背景材料的问答"
    src_term = "背景材料"
    tgt_term = "生成的问题和答案"
    language = "中文"
    more_guidance = ['每个问题都应该基于给定的背景材料进行表述。']
    # 提示用户输入背景材料

    question_list = []
    answer_list = []
    background_material = input_text
    score_agent = ScoringAgent(logger, chat_gpt, task_name, language=language,background_material=background_material, src_term=src_term, tgt_term=tgt_term,
                            more_guidance=more_guidance)
    
    prompt, generated_questions_answers = score_agent.generate_questions_answers({"background_material": background_material})
    a = generated_questions_answers
    a = a.replace("'","\"")
    a = a.replace("`","")
    pattern = r'"(?![^{]*})'  
    a = re.sub(pattern, '', a) 
    a = a.replace("'",'"')
    json_pattern = r'{.*?}'
    a = re.search(json_pattern, a, re.DOTALL)
    a = a.group()
    try:
        questions_and_answers = json.loads(a)
    except json.JSONDecodeError:
        # 如果 JSON 解析错误，则执行 continue
        print(666)
    for question, answer in questions_and_answers.items():  
        question_list.append(question)
        answer_list.append(answer)
    return question_list,answer_list

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("您提供的参数数量不对哦，请分别依次提供输入路径，输出路径，是否选择API或者本地大模型（0代表调用API，1代表调用本地部署大模型，和期待使用线程数）")
        sys.exit(1)
    # input_path = sys.argv[1]
    # out_path = sys.argv[2]
    # select = sys.argv[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="输入路径")
    parser.add_argument("out_path", type=str, help="输出路径")
    parser.add_argument("select", type=int, help="选择（0代表调用API，1代表调用本地部署大模型）")
    parser.add_argument("--config_path", type=str, default="llms/remote/configs/wsx_gpt35.json")
    parser.add_argument("num_thread", type=int, help="指定您期待的线程数（现行版本仅支持1或2）")
    args = parser.parse_args()
    if args.num_thread == 2 and args.select == 1:
        print("大模型暂不支持多线程")
        sys.exit(1)
    if is_json_path(args.input_path) is True:

        logger = set_logger("tmp.log")
        if args.select=="0":
            chat_gpt = ChatGPTLLM(args.config_path)
        else :
            chat_gpt = ChatGLM2_6BWrapper(args.config_path)
        task_name = "基于背景材料的问答"
        src_term = "背景材料"
        tgt_term = "生成的问题和答案"
        language = "中文"

        more_guidance = ['每个问题都应该基于给定的背景材料进行表述。']

        # 提示用户输入背景材料

        text = pd.read_json(args.input_path)
        with open("llms/applications/duandian.txt","r",encoding="utf-8") as file:
            lines = file.readlines()
        print(lines[0])
        j = int(lines[0])
        chat_gpt = ChatGPTLLM(args.config_path)
        for i in range(j,15):
            dui = []
            question_list = []
            answer_list = []
            time.sleep(10)
            with open("llms/applications/duandian.txt","w",encoding="utf-8") as file:
                file.write(str(i))
            background_material = text["text"][i]
            score_agent = ScoringAgent(logger, chat_gpt, task_name, language=language, src_term=src_term, tgt_term=tgt_term,
                                    more_guidance=more_guidance)
            
            prompt, generated_questions_answers = score_agent.generate_questions_answers({"background_material": background_material})
            # print("生成的提示:", prompt)
            # print("生成的问题和答案:", generated_questions_answers)
            if i==0:
                with open(args.out_path+'questions_and_answers.txt', 'w', encoding='utf-8') as file:  
                    tt = str(i+1)
                    file.write("以下是第"+ tt +"个文章的问答对：\n")
                    a = generated_questions_answers
                    a = a.replace("'","\"")
                    a = a.replace("`","")
                    pattern = r'"(?![^{]*})'  
                    a = re.sub(pattern, '', a) 
                    a = a.replace("'",'"')
                    json_pattern = r'{.*?}'
                    a = re.search(json_pattern, a, re.DOTALL)
                    a = a.group()
                    try:
                        questions_and_answers = json.loads(a)
                    except json.JSONDecodeError:
                        with open('error.txt', 'a', encoding='utf-8') as file:
                            file.write("第"+ tt +"个文章无法处理：\n") 
                        continue
                    for question, answer in questions_and_answers.items():  
                        question_list.append(question)
                        answer_list.append(answer)
                        file.write(f"{question}\n{answer}\n\n")
            else:
                with open(args.out_path+'questions_and_answers.txt', 'a', encoding='utf-8') as file: 
                    tt = str(i+1)
                    file.write("以下是第"+ tt +"个文章的问答对：\n") 
                    a = generated_questions_answers
                    a = a.replace("'","\"")
                    a = a.replace("`","")
                    pattern = r'"(?![^{]*})'  
                    a = re.sub(pattern, '', a) 
                    a = a.replace("'",'"')
                    json_pattern = r'{.*?}'
                    a = re.search(json_pattern, a, re.DOTALL)
                    a = a.group()
                    questions_and_answers = json.loads(a)
                    for question, answer in questions_and_answers.items():  
                        question_list.append(question)
                        answer_list.append(answer)
                        file.write(f"{question}\n{answer}\n\n") 
            for k in range(len(question_list)):
                new_dict = {"id":i+1,"question":question_list[k],"answer":answer_list[k]}
                dui.append(new_dict)
            if i == 0:    
                with open(args.out_path+"data.json", "w") as json_file:
                    json.dump(dui, json_file, indent=4, separators=(",", ": "), ensure_ascii=False)
            else:
                with open(args.out_path+"data.json", "a") as json_file:
                    json.dump(dui, json_file, indent=4, separators=(",", ": "), ensure_ascii=False)
    if is_json_path(args.input_path) is False:
        txt = []
        for filename in os.listdir(args.input_path):  
            # 检查文件是否为txt文件  
            if filename.endswith('.txt'):  
                # 构建文件的完整路径  
                file_path = os.path.join(args.input_path, filename)  
                
                # 打开文件并读取内容  
                with open(file_path, 'r', encoding='utf-8') as file:  # 使用utf-8编码读取文件，可以根据需要更改  
                    content = file.read()  # 读取文件全部内容，你也可以使用file.readlines()按行读取  
                    txt.append(content) 
        
    

        logger = set_logger("tmp.log")

        if args.select=="0":
            chat_gpt = ChatGPTLLM(args.config_path)
        else :
            chat_gpt = ChatGPTLLM(args.config_path)
            kkkk = ChatGLM2_6BWrapper(args.config_path)

        task_name = "基于背景材料的问答"
        src_term = "背景材料"
        tgt_term = "生成的问题和答案"
        language = "中文"

        more_guidance = ['每个问题都应该基于给定的背景材料进行表述。']

        # 提示用户输入背景材料
        print("请输入背景材料：")
        
        # text = pd.read_json("llms/applications/text.json")
        text = pd.read_json(args.input_path)
        with open("llms/applications/duandian.txt","r",encoding="utf-8") as file:
            lines = file.readlines()
        print(lines[0])
        j = int(lines[0])
        for i in range(j,15):
            dui = []
            question_list = []
            answer_list = []
            time.sleep(10)
            with open("llms/applications/duandian.txt","w",encoding="utf-8") as file:
                file.write(str(i))
            background_material = txt[i]
            score_agent = ScoringAgent(logger, chat_gpt, task_name, language=language, src_term=src_term, tgt_term=tgt_term,
                                    more_guidance=more_guidance)
            
            prompt, generated_questions_answers = score_agent.generate_questions_answers({"background_material": background_material})
            # print("生成的提示:", prompt)
            # print("生成的问题和答案:", generated_questions_answers)
            if i==0:
                with open(args.out_path+'questions_and_answers.txt', 'w', encoding='utf-8') as file:  
                    tt = str(i+1)
                    file.write("以下是第"+ tt +"个文章的问答对：\n")
                    a = generated_questions_answers
                    a = a.replace("'","\"")
                    a = a.replace("`","")
                    pattern = r'"(?![^{]*})'  
                    a = re.sub(pattern, '', a) 
                    a = a.replace("'",'"')
                    json_pattern = r'{.*?}'
                    a = re.search(json_pattern, a, re.DOTALL)
                    a = a.group()
                    questions_and_answers = json.loads(a)
                    for question, answer in questions_and_answers.items():  
                        question_list.append(question)
                        answer_list.append(answer)
                        file.write(f"{question}\n{answer}\n\n")
            else:
                with open(args.out_path+'questions_and_answers.txt', 'a', encoding='utf-8') as file: 
                    tt = str(i+1)
                    file.write("以下是第"+ tt +"个文章的问答对：\n") 
                    a = generated_questions_answers
                    a = a.replace("'","\"")
                    a = a.replace("`","")
                    pattern = r'"(?![^{]*})'  
                    a = re.sub(pattern, '', a) 
                    a = a.replace("'",'"')
                    json_pattern = r'{.*?}'
                    a = re.search(json_pattern, a, re.DOTALL)
                    a = a.group()
                    questions_and_answers = json.loads(a)
                    for question, answer in questions_and_answers.items():  
                        question_list.append(question)
                        answer_list.append(answer)
                        file.write(f"{question}\n{answer}\n\n") 
            for k in range(len(question_list)):
                new_dict = {"id":i+1,"question":question_list[k],"answer":answer_list[k]}
                dui.append(new_dict)
            if i == 0:    
                with open(args.out_path+"data.json", "w") as json_file:
                    json.dump(dui, json_file, indent=4, separators=(",", ": "), ensure_ascii=False)
            else:
                with open(args.out_path+"data.json", "a") as json_file:
                    json.dump(dui, json_file, indent=4, separators=(",", ": "), ensure_ascii=False)
