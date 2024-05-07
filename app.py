from flask import Flask, request, jsonify ,render_template,send_file
from llms.applications.QA_maker import make_qa,make_now,make_api
import requests
app = Flask(__name__)  
# 初始化要发送的消息
message = ""


data = {
    "prompt": message,
    "history": "",
    "max_length": 100,
    "top_p": 0.7,
    "temperature": 0.95
}
qq_list = []
aa_list = []
qqq_list = []
aaa_list = []
@app.route('/')
def index():
    return render_template('index.html')
    
def generate_qapairs_with_model(text):   
    global qq_list,aa_list
    qq_list,aa_list = make_now(text)
    response = requests.post("http://localhost:8097/", json=data)
    pairs = []
    for _ in range(len(qq_list)):
        pairs.append(qq_list[_])
        pairs.append(aa_list[_])  
        print(response)
    return pairs  
   
def generate_qapairs_with_api(text):  
    global qqq_list,aaa_list
    qqq_list,aaa_list = make_api(text)
    pairs = []
    for _ in range(len(qqq_list)):
        pairs.append(qqq_list[_])
        pairs.append(aaa_list[_])  
    return pairs  
  
# 大模型生成问答对的路由  
@app.route('/generate-qapairs-model', methods=['POST'])  
def generate_qapairs_model_route():  
    data = request.get_json()  
    text = data['text']  
    pairs = generate_qapairs_with_model(text)  
    return jsonify({'pairs': pairs})  
  
# API生成问答对的路由  
@app.route('/generate-qapairs-api', methods=['POST'])  
def generate_qapairs_api_route():  
    data = request.get_json()  
    text = data['text']  
    pairs = generate_qapairs_with_api(text)  
    return jsonify({'pairs': pairs})  
q_list = []
a_list = []

@app.route('/upload-file', methods=['POST'])
def upload_file():
    global q_list,a_list
    file = request.files['file']
    name = str(file.filename)
    file = "input/" + name
    q_list,a_list = make_qa(file)
    print(99999999999999999999)
    print(q_list)
    print(444444444444444444)
    # print(a_list)
    # file.save(file.filename)
    return '文件上传成功'

@app.route('/output-file')
def output_file():
    global q_list,a_list
    dui = []
    i = 0
    filename = 'output.json'  
    import json
    for k in range(len(q_list)):
        new_dict = {"id":i+1,"question":q_list[k],"answer":a_list[k]}
        dui.append(new_dict)
        with open("output.json", "a") as json_file:
            json.dump(dui, json_file, indent=4, separators=(",", ": "), ensure_ascii=False)
        dui = []
        i += 1
    return send_file(filename, as_attachment=True)
  
if __name__ == '__main__':  
    app.run(debug=True)