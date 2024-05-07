import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
import argparse


from remote import RemoteLLMs

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="llms/remote/configs/wsx_gpt35.json")
    args = parser.parse_args()
    return args


class ChatGLM2_6BWrapper(RemoteLLMs):
    def init_local_client(self):
        """
        初始化本地的 ChatGLM2-6B 模型
        :return: 模型实例
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
            model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
            # model.eval()

            # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True)
            # model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True).float()

            model = model.eval()
            return (tokenizer, model)
        except Exception as e:
            print(f"初始化模型失败：{e}")
            return None

    def create_prompt(self, current_query):
        history = []
        """
        根据历史对话和当前查询创建输入提示
        :param history: 对话历史列表，包含 (query, response) 元组
        :param current_query: 当前用户输入
        :return: 创建的提示信息字符串
        """
        prompt = "ChatGLM2-6B 对话:"
        for query, response in history:
            prompt += f"\n用户：{query}\nChatGLM2-6B：{response}"
        prompt += f"\n用户：{current_query}"
        return prompt

    def request_llm(self, context, seed=1234, sleep_time=1, repeat_times=0):
        """
        向语言模型发送请求并获取回答
        :param context: 输入的上下文或提示
        :param seed: 随机种子，用于确保结果的可重复性
        :param sleep_time: 请求之间的休眠时间（秒）
        :param repeat_times: 重复请求次数（用于处理失败的请求）
        :return: 模型的回答
        """
        tokenizer, model = self.client
        input_ids = tokenizer.encode(context, return_tensors='pt').to('cuda')
        # input_ids = tokenizer.encode(context, return_tensors='pt').to('cpu')
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=1024, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        context = [{
                    'role': "assistant",
                    'content': response
                }]
        return context

if __name__ == "__main__":
    # 示例：创建实例并使用
    # config_path = "path_to_your_config.json"
    args = read_args()
    chat_glm_wrapper = ChatGLM2_6BWrapper(args.config_path)
    while True:
        user_input = input("请输入你的查询：")
        if user_input.lower() == 'quit':
            break
        prompt = chat_glm_wrapper.create_prompt([], user_input)
        response = chat_glm_wrapper.request_llm(prompt)
        print(f"ChatGLM2-6B：{response}")
