import argparse
import json
import sys
from tmp_utils import set_logger
sys.path.append('llms/')
from remote.__init__ import RemoteLLMs
from remote.ChatGPT import ChatGPTLLM


class ScoringAgent:
    def __init__(self, logger, llm_model: RemoteLLMs, task_name: str, language,
                 src_term, tgt_term, in_context_examples=[], more_guidance=[], more_task_definition=[]):
        self.logger = logger
        self.llm_model = llm_model
        self.prompt_pattern = """
[Task Description] 
Here is a task of generating questions and answers based on given background material. All [Input] are in {{Language}}.
{{MORE_TASK_DEFINITION}}
Your are required to generate questions and corresponding answers based on the provided %SRC%.
If you can not understand my background material,Please automatically generate the background and the questions and answers for that background.
Your output should follow the [Output Format].

[Guidance]
You should strictly follow my guidance:
1. If you don't understand the background material I've given, please automatically generate a new background material with questions and answers for that background (that's the premise of everything)
2. Each question should be formulated based on the given background material.
3. Each answer should be relevant and informative.
4. You should strictly follow the given output format and can't output other information.
{{MORE_GUIDANCE}}
If you break my guidance, you will be penalized.

[Output Format]
Your output should strictly follow this format and can be directly decoded by Python:
'''
{{Output}}
'''

[Input]
'''
{
    background_material: {{SRC_VALUE}}
}
'''
 
"""

        if len(in_context_examples) > 0:
            more_guidance.append('To help your judgment, some examples are provided in [Examples].')
            in_context_prompt = ["[Examples]", "'''"]
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
            "{{Language}}": language,
            "{{Output}}": "{'Question1': 'Answer1', 'Question2': 'Answer2'}",
            "{{Input}}": input_format,
            "{{TASK_NAME}}": task_name,
            "{{MORE_GUIDANCE}}": more_guidance,
            "{{MORE_TASK_DEFINITION}}": '\n'.join(more_task_definition),
            "{{In-Context Examples}}": in_context_prompt,
            "{{SRC_VALUE}}":background_material
        }

    def generate_questions_answers(self, background_material):
        llm_model = self.llm_model
        repeat_times = -1

        while True:
            repeat_times += 1
            if repeat_times >= llm_model.max_retries:
                break

            # 构造 prompt，包括背景材料
            prompt = llm_model.fit_case(pattern=self.prompt_pattern, data=background_material, meta_dict=self.meta_dict)
            contexts = llm_model.create_prompt(prompt)

            # 请求问题-答案对
            results = llm_model.request_llm(contexts, repeat_times=repeat_times)

            if results is not None and results[-1]['role'] == 'assistant':
                generated_questions_answers = self.extract_questions_answers(results[-1]['content'])
                if generated_questions_answers is not None:
                    return prompt, generated_questions_answers

        return None

    def extract_questions_answers(self, last_response: str):
        try:
            return last_response
        except Exception as e:
            raise e
            return None





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="llms/remote/configs/wsx_gpt35.json")
    args = parser.parse_args()

    logger = set_logger("tmp.log")

    chat_gpt = ChatGPTLLM(args.config_path)

    task_name = "Question Answering Based on Background Material"
    src_term = "Background Material"
    tgt_term = "Generated Questions and Answers"
    language = "Chinese"

    more_guidance = ['Each question should be formulated based on the given background material.']

    # 提示用户输入背景材料
    print("Please provide the background material:")
    background_material = input()

    score_agent = ScoringAgent(logger, chat_gpt, task_name, language=language, src_term=src_term, tgt_term=tgt_term,
                               more_guidance=more_guidance)
    
    prompt, generated_questions_answers = score_agent.generate_questions_answers({"background_material": background_material})
    print("Generated Prompt:", prompt)
    print("Generated Questions and Answers:", generated_questions_answers)
