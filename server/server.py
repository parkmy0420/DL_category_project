from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
from tqdm import tqdm
import model

app = Flask(__name__)

@app.route('/file_upload', methods=['GET', 'POST']) # 함수 이름이 아닌 해당 코드로 html과 연결됨 
def file_upload():
    save_f = []     # save file name list
    if request.method == 'POST':
        up_f = request.files.getlist("file[]")  # request: html에서 보낸 매개변수를 받아서 저장, html의 name="file[]"과 이름이 같아야함
        for f in up_f:
            f_name = './static/IMG/' + secure_filename(f.filename)
            f.save(f_name)
            save_f.append(f_name)
        label = inference(save_f)
        return render_template('file_upload.html', name=save_f, label = label)    # 지정해준 파일로 가서 열린다. 
        
    return render_template('file_upload.html')

def inference(filename):

    # 상세 분류
    label =  model.get_best_class(filename)
    #print(label)
    return label
    

if __name__ == '__main__':
    app.run('0.0.0.0', port=8765, debug=True)