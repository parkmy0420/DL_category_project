from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
from tqdm import tqdm

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
        return render_template('file_upload.html', name=save_f, label=label)    # 지정해준 파일로 가서 열린다. 
        
    return render_template('file_upload.html')

def inference(filename):

    # 상세 분류
    category_dict = {
        0 : 'JumpSuit', 1 : 'Blouse', 2 : 'Tshirt', 3 : 'KnitWear', 4 : 'Shirt', 5 : 'Cardigan',
        6 : 'Hoodie', 7 : 'Jeans', 8 : 'Pants', 9 : 'Skirt', 10 : 'Dress', 11 : 'JoggerPants',
        12 : 'Coat', 13 : 'Jacket', 14 : 'Jumper', 15 : 'PaddedJacket', 16 : 'Vest'}
    
    length_dict = {
        0: 'Sleeveless shirt', 1: 'Short sleeve', 2: 'Cap', 3: 'Three-quarter sleeve', 4: 'Long sleeve'}
    
    material_dict = {0: 'padding', 1: 'fur', 2: 'Mustang', 3: 'suede', 4: 'corduroy', 5: 'Sequin/Glitter',
        6: 'Denim', 7: 'jersey', 8: 'tweed', 9: 'velvet', 10: 'vinyl/PVC', 11: 'wool/cashmere', 12: 'hair knit',
        13: 'knit', 14: 'lace', 15: 'linen', 16: 'messi', 17: 'fleece', 18: 'neoprene', 19: 'silk',
        20: 'spandex', 21: 'jacquard', 22: 'leather', 23: 'chiffon', 24: 'woven'}

    category_model = YOLO('./category_best.pt')
    length_model = YOLO('./length_best.pt')
    material_model = YOLO('./material_best.pt')

    label = []
    
    # 대분류: 상/하의, 원피스, 점프수트, 아우터 구분
    # Dress = 10
    # jumpsuit = 0
    top = [1, 2, 3, 4, 6]
    bottom = [7, 8, 9, 11]
    outer = [12, 13, 14, 15, 5, 16]

    # 모델 추론
    for img in tqdm(filename):
        category_results = category_model(img, verbose=False, conf=0.35) # 모델마다 conf 조절 필요
        material_results = material_model(img, verbose=False)
        # length_results = length_model(img, verbose=False)

        total = len(category_results)
        ca_boxes = category_results[0].boxes
        ca_cls = ca_boxes.cls.tolist()

        mate_boxes = material_results[0].boxes
        mate_cls = mate_boxes.cls.tolist()

        print('file_name:', img)
        print('ca_cls: ', ca_cls)
        print('mate_cls: ', mate_cls)
        
        # MY BRANCH TEST
    return label
    

if __name__ == '__main__':
    app.run('0.0.0.0', port=8765, debug=True)