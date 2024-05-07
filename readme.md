# 基于ChatGPT API与本地ChatGLM-6B的问答对生成

## 1 环境配置

### 1.1 yml包导入

```bash
conda env create -f LLM.yml
```

### 1.2 原生环境部署指南
本文默认所使用的处理器为GPU，所使用机器已装备好conda环境。

首先，创立新环境并配置相匹配的pytorch环境：
```bash
# nvidia-smi  可查询CUDA Version
# 随后前往pytorch官网 https://pytorch.org/ 查询对应的 pytorch版本下载命令

conda create -n QA_env
conda active QA_env
conda install pip3

# 请确保所使用命令对应版本与机器适配
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

接下来，切换到本项目根目录并下载所需环境
```bash
# 切换到本项目根目录
cd path/to/LLM_QA

# 下载所需库，其中主要是
pip install -r ./set-up/requirements.txt
```

## 2 项目使用

### 2.1 输入输出格式规定


总体而言，本项目的输入基本单元是文本，输出基本单元是对于每一段文本所提取出的问答对。

项目输入适配两种格式，项目输出则固定采用json.

**输入格式**

+ `json`格式，每个`json`文件表示一个数据集，提交`json`
+ `txt`格式，每个`txt`文件表示一条数据，提交含`txt`文件的文件夹路径

**输出格式**

+ `json`格式，示例：
```json
[
    {
        "id": 1,
        "question": "小明发现了什么？",
        "answer": "小明在阁楼的旧箱子里发现了一块古老的怀表。"
    },
    {
        "id": 1,
        "question": "怀表具有什么特殊之处？",
        "answer": "这块怀表看似普通，但却散发着神秘的光芒。"
    },
    {
        "id": 2,
        "question": "小莉是否经常探险寻找秘密?",
        "answer": "是的,小莉有一个特殊的爱好,那就是喜欢在村庄周围探险,寻找未知的秘密。"
    },
    {
        "id": 2,
        "question": "小莉是如何发现魔法花园的?",
        "answer": "小莉无意间发现了一个隐藏在密林深处的古老花园,花园的入口被一堵爬满藤蔓的高墙遮挡,仿佛是在守护着某个秘密。"
    }
]
```

### 2.2 命令行交互指南
 
> usage: python path/to/QA_maker.py path/to/input path/to/output mode
> 
> mode="0" -> GPT API; 
> 
> mode="1" -> local GLM

例如：

```
python .\llms\applications\QA_maker.py .\input\text.json .\output\ 0

python .\llms\applications\QA_maker.py .\input\story\ .\output\ 1
```


## 大预言模型（LLM）与提示词工程

### LLM的基本概念与发展

大语言模型(LLM)是当今人工智能领域的一项重要创新，其在自然语言处理任务中表现出色，成为了各领域研究和应用的热门方向。LLM基于深度学习技术，拥有数十亿甚至上百亿的参数量，使得其在理解和生成自然语言文本方面取得了显著突破。通过大规模的预训练和微调过程，LLM能够从大量的文本数据中学习语法、语义、情感等信息，从而实现对复杂自然语言任务的高水平处理。

LLM在自然语言处理领域引起了巨大关注，其应用领域涵盖了文本生成、文本理解、情感分析、机器翻译等多个领域。CHATGPT作为其中的杰出代表，通过深层的变换和自注意力机制，能够生成流畅、连贯且有逻辑性的文本，使得其在各种语言任务中都展现出令人印象深刻的性能。

### 提示词工程与应用

提示词工程作为一种创新的方法，致力于利用LLM的强大能力，通过适当的提示词语来引导模型生成特定类型的输出。提示词工程结合了领域知识、语境设定和问题引导，以确保LLM能够按照期望的方式生成内容。

通过对话历史的分析和挖掘，我们可以更好地理解提示词工程的优势和应用。提示词工程能够根据用户需求，构建合适的提示词语，从而引导LLM生成与口语学习相关的内容。这种方式可以通过灵活调整提示词的构成原理，使得LLM输出更贴合不同场景的需求。

### 提示词工程设计

我们的任务背景是 “Question-Answer及给定一段背景材料，要求模型从中发现问题以及答案（即生成问题、答案对）再搭配任务场景时，我们对每个场景的提示词进行了详细设计与构建。提示词的组成原理包括了Task Description、role、Guidance、In-Context Examples、Limit。

| 元素                  | 具体含义                                       |
|-----------------------|------------------------------------------------|
| 任务描述(Task Description) | 任务描述部分详细描述了特定的问答对抽取场景，使模型尽可能的抽取背景材料中的问答对。有助于为大模型提供具体的任务场景信息。 |
| 角色(Role)                | 角色部分告知大语言模型在特定场景下扮演的角色，一个专业的语言学者。有助于模型在生成回复时更好的理解自己的身份和定位。       |
| 指导(Guidance)            | 指导部分明确了模型完成任务时必须遵守的要求，如流畅的语言表述，问答对必须基于背景材料等，有助于引导模型生成更适合于背景材料的问答对。 |
| 示例(In-Context Examples) | 实力部分提供了特定材料下抽取的问答对范例，使的模型能够从少量的示例中推断出适合于的顶背景材料下的问答对。            |
| 限制(Limit)               | 限制部分对模型的输入输出格式进行了限制，这使得模型的输出更符合我们的预期效果。                          |

在具体构建工程中我们发现如果将输出格式受限的太死板会影响大模型的输出灵活性，所以并未对输出格式做具体限制。我们构建了中英文的prompt，具体如下

**【英文】**

```
[Task Description] 
This is a task where you are required to generate as many question-answer pairs as possible based on the given background information. All inputs are in {{Language}}.
You need to act as a professional native-speaker human annotator to create  question-answer pairs from the provided {{BACKGROUND}} in [input]  .
Your generated pairs should follow the [Guidance].

[Guidance]
Please strictly adhere to the following guidelines:
1. Each generated question-answer pair should be relevant to the given background and summarize background material whenever possible.
2. The questions should be clear and answerable based on the provided information.
3. Make sure that both the question and answer are grammatically correct and fluent .
4. If you do not understand the given background material please generate your own background material and Q&A pair and let me know you generated it yoursel

To help your judgment, some examples are provided in [Examples].
Failure to comply with the guidelines may result in penalties.

{{In-Context Examples}}

[input]
'''
{
    "{{BACKGROUND}}": {{BACKGROUND_value}}
}
'''
```

**【中文】**

```
[任务描述]
这是一个根据给定的背景材料生成问题和答案的任务。所有的[输入]都是中文的。

你需要根据提供的%SRC%生成问题和相应的答案。
你的输出应该遵循[输出格式]。

[指导]
你应该严格遵循我的指导：
1.每个问题都应该基于给定的背景材料进行表述。
2.每个答案都应该相关且信息丰富


3.你应该严格遵循给定的输出格式，不能输出其他信息。
4.每个问题都应该基于给定的背景材料进行表述。
如果你违反我的指导，将会受到惩罚。

[输出格式]
你的输出应严格遵循这个格式，并且可以直接由Python解码：
'''
{'问题1':'答案1','问题2':'答案2'}
'''

[输入]
'''
{
背景材料:根據網站Icedeal的報導，新規定要求運算能力超過70TeraFLOPS的組件及電腦，在出口到名單內的國家前均需要事先取得授權。今次的更新對GeForceRTX4090D顯
示卡和NVIDIAH20(Hopper)數據中心加速器帶來影響，因為兩者的運算能力分別為73.5TFLOPS和74TFLOPS。新規則還對一些限制作出修訂，例如對中國、澳門和其他D5組別國家
採取預設否決政策，美國亦會對所有出口人工智能晶片到中國的要求作出審查。
}
'''
```

### 问题与改进

#### 示例部分的探索

在测试过程中，我们观察到了几个关键问题。当给定一段不易理解的背景材料时，模型在生成问答对时可能会受到示例中特定人物的影响。例如，如果示例中提到了人物“小明”，但模型未能理解背景材料中的上下文，它可能会不加区分地将“小明”加入生成的问答对中。这引发了我们对示例数据质量和代表性的思考。

我们发现模型在处理简单语境时可能会出现拆分句子并以主语和谓语作为问题的现象。例如，在给定背景材料“今天星期四，明天星期五”时，我们期望模型能够正确提取问题“后天星期几？”并生成答案“后天星期六”，但实际上模型生成了问题“今天星期几？”和答案“今天星期四”。这表明模型在处理时间关系时可能存在一定的局限性，需要进一步改进。

所以我们对prompt中的示例做出了修改，背景材料保持不变均为“一位名为小明的学生参加了一场数学竞赛，题目难度很高，但他取得了优异的成绩”

| 原有示例   | 现有示例        |
|------------|-----------------|
| 问题       | 小明参加了什么比赛？ |
| 答案       | 数学竞赛        |
| 新问题     | 小明会感觉到开心吗？ |
