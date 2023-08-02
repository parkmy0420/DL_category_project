import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2, torch
import pandas as pd
import numpy as np

# 박스 위치 확인(이미지 확인)
def visualize_yolo(image_path, x, y, w, h):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    # Draw the bounding box
    xmin = int((x - w / 2) ) 
    ymin = int((y - h / 2) )
    xmax = int((x + w / 2) )
    ymax = int((y + h / 2) )
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    #cv2.putText(img, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# df 생성 함수
def output_df(model, image):
    result = model.predict(image)
    names_list = model.names

    for each_result in result:
        cls_list = []
    
        # conf_confidence df 추가
        conf_list = each_result.boxes.conf.tolist()
    
        # class df 추가
        for each_class in each_result.boxes.cls:
            cls_list.append(names_list[int(each_class)])
   
        # x, y, w, h
        length_xywh_list = result[0].boxes.xywh.tolist()
        xywh_list = []
        for each_xywh in length_xywh_list:
            
            # x,y,w,h df 추가
            each_xywh = [int(i) for i in each_xywh]
            xywh_list.append(each_xywh)
        
            # 시각화
            #print(each_xywh)
            #xmin, ymin, xmax, ymax = each_xywh
            #visualize_yolo(image, xmin, ymin, xmax, ymax)

        
    predicted_data = pd.DataFrame(zip(cls_list, conf_list, xywh_list), columns=['class', 'conf_confidence', 'xywh']).sort_values(by=['conf_confidence'], ascending=False)
    return predicted_data


# 카테고리 df 생성
def get_category_df(category_model, test_image_list):
    result_df = pd.DataFrame()
    for each_image in test_image_list:
        category_df =  output_df(category_model, each_image)
        for index, row in category_df.iterrows():
            if row['class'] in ['JumpSuit', 'Dress']:
                category_df.loc[index, 'first_class'] = 'Onepiece'
            if row['class'] in ['Blouse', 'Tshirt', 'KnitWear', 'Shirt', 'Cardigan', 'Hoodie']:
                category_df.loc[index, 'first_class'] = 'Top'
            if row['class'] in ['Jeans', 'Pants', 'Skirt', 'JoggerPants']:
                category_df.loc[index, 'first_class'] = 'Bottom'
            if row['class'] in ['Coat', 'Jacket', 'Jumper', 'PaddedJacket', 'Vest']:
                category_df.loc[index, 'first_class'] = 'Outer'
            
        result_df = pd.concat([result_df, category_df], axis=0)
    return result_df


# 소재 df 생성
def get_material_df(material_model, test_image_list):
    result_df = pd.DataFrame()
    for each_image in test_image_list:
        material_df =  output_df(material_model, each_image)
        result_df = pd.concat([result_df, material_df], axis=0)
    return result_df

# 소매기장 df 생성
def get_length_df(length_model, test_image_list):
    result_df = pd.DataFrame()
    for each_image in test_image_list:
        length_df =  output_df(length_model, each_image)
        result_df = pd.concat([result_df, length_df], axis=0)
    return result_df

# 모델별 결과 매칭 함수
# 카테고리 df row의 좌표별로 소재 및 소매기장 df row의 좌표를 각각 비교하여 가장 작은(가까운) 거리끼리 매칭

def get_match_df(image_list):
    
    # 모델 불러오기
    category_model = YOLO('./category_best.pt')
    length_model = YOLO('./length_best.pt')
    material_model = YOLO('./material_best.pt')
    
    # 카테고리별 df 
    category_df = get_category_df(category_model, image_list)
    material_df = get_material_df(material_model, image_list)
    length_df = get_length_df(length_model, image_list)
  
    # 결과 df
    result_df = category_df.copy()
    result_df.rename(columns={'class' : 'c_class', 'conf_confidence':'c_conf_confidence',
                            'xywh' : 'c_xywh', 'first_class':'c_first_class'}, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    

    for re_index, result_row in result_df.iterrows():
        category_temp = result_row['c_xywh']

        c_x_center = category_temp[0] + category_temp[2] /2
        c_y_center = category_temp[1] + category_temp[3] /2
        
        
        # 카테고리와 소재 매칭
        temp_x_center1 = 0
        temp_y_center1 = 0
        
        # 소재 df row가 두개 이상일 경우
        # 카테고리 df row의 좌표별로 소재 df row의 좌표를 각각 비교하여 가장 작은 거리끼리 매칭
        if len(material_df) > 1:
            
            for idx, m_row in material_df.iterrows():
                
                material_temp = m_row['xywh']
                m_x_center = material_temp[0] + material_temp[2] /2
                m_y_center = material_temp[1] + material_temp[3] /2
                
                diff_x_center = abs(c_x_center - m_x_center)
                diff_y_center = abs(c_y_center - m_y_center)
                
                    
                if temp_x_center1 == 0 and temp_y_center1 == 0:
                    temp1 = m_row
                    temp_x_center1 = diff_x_center
                    temp_y_center1 = diff_y_center
               
                elif temp_x_center1 >= diff_x_center:
                    if temp_y_center1 >= diff_y_center:
                        result_df.loc[re_index, 'm_class'] = m_row['class']
                        result_df.loc[re_index, 'm_conf_confidence'] = m_row['conf_confidence']
                        result_df.loc[re_index, 'm_xywh'] = str(m_row['xywh'])
                        
                else:
                    result_df.loc[re_index, 'm_class'] = temp1['class']
                    result_df.loc[re_index, 'm_conf_confidence'] = temp1['conf_confidence']
                    result_df.loc[re_index, 'm_xywh'] = str(temp1['xywh'])
                    
        # 소재 df row가 한개일 경우(없을 경우는 고려하지 않음)에각 모든 카테고리 row에 추가
        # 없는 경우에는 NaN 값이 됨
        else:
            for idx, m_row in material_df.iterrows():
                result_df.loc[re_index, 'm_class'] = m_row['class']
                result_df.loc[re_index, 'm_conf_confidence'] = m_row['conf_confidence']
                result_df.loc[re_index, 'm_xywh'] = str(m_row['xywh'])
                
        #==========================================================================================       
        # 소매기장 매칭
        # 1차 카테고리가 'Onepiece', 'Top', 'Outer'  중에 해당될 경우
        if result_row['c_first_class'] in ['Onepiece', 'Top', 'Outer']:
            
            temp_x_center2 = 0
            temp_y_center2 = 0
            
            # 소매기장 df row가 두개 이상일 경우
            # 카테고리 df row의 좌표별로 소매길이 df row의 좌표를 각각 비교하여 가장 작은 거리끼리 매칭
            if len(result_df[result_df['c_first_class'].isin(['Onepiece', 'Top', 'Outer'])]) > 1: 
                    
                for idx, l_row in length_df.iterrows():
                    length_temp2 = l_row['xywh']
                    l_x_center = length_temp2[0] + length_temp2[2] /2
                    l_y_center = length_temp2[1] + length_temp2[3] /2
                
                    diff_x_center = abs(c_x_center - l_x_center)
                    diff_y_center = abs(c_y_center - l_y_center)
                    
                    
                    if temp_x_center2 == 0 and temp_y_center2 == 0:
                        temp2 = l_row
                        temp_x_center2 = diff_x_center
                        temp_y_center2 = diff_y_center
                    
                    elif temp_x_center2 >= diff_x_center:
                        if temp_y_center2 >= diff_y_center:
                            result_df.loc[re_index, 'l_class'] = l_row['class']
                            result_df.loc[re_index, 'l_conf_confidence'] = l_row['conf_confidence']
                            result_df.loc[re_index, 'l_xywh'] = str(l_row['xywh'])
                        
                    else:
                        result_df.loc[re_index, 'l_class'] = temp2['class']
                        result_df.loc[re_index, 'l_conf_confidence'] = temp2['conf_confidence']
                        result_df.loc[re_index, 'l_xywh'] = str(temp2['xywh'])
                        
            # 소매길이 df row가 한개일 경우(없을 경우는 고려하지 않음)에각 모든 카테고리 row에 추가
            # 없는 경우에는 NaN 값이 됨
            else:  
                for idx, l_row in length_df.iterrows():
                    #print(l_row)
                    result_df.loc[re_index, 'l_class'] = l_row['class']
                    result_df.loc[re_index, 'l_conf_confidence'] = l_row['conf_confidence']
                    result_df.loc[re_index, 'l_xywh'] = str(l_row['xywh'])
                    
    return result_df
                        

# 클래스 한글화 함수
def get_korean_class(first_category, second_category, material, length=None):
    
    # 첫번째 카테고리
    korean_first_category_dict = {'Onepiece' : '원피스', 'Top' : '상의', 'Outer' : '아우터', 'Bottom':'하의'}
    for key, value in korean_first_category_dict.items():
        if first_category == key:
            k_first_category = value
            
            
    # 두번째 카테고리
    korean_second_category_dict = {'JumpSuit' : '점프수트', 'Blouse' : '블라우스', 'Tshirt' : '티셔츠', 
                                'KnitWear':'니트웨어', 'Shirt':'셔츠', 'Cardigan':'가디건', 'Hoodie':'후드티', 'Jeans':'청바지',
                                'Pants':'팬츠', 'Skirt':'스커트', 'Dress':'드레스', 'JoggerPants':'조거팬츠',
                                'Coat':'코트','Jacket':'재킷', 'Jumper':'점퍼', 'PaddedJacket':'패딩', 'Vest':'베스트'}
    for key, value in korean_second_category_dict.items():
        if second_category == key:
            k_second_category = value
            
    
    # 소재
    korean_material_dict = {'padding' : '패딩', 'Mustang' : '무스탕', 'suede':'스웨이드', 'corduroy':'코듀로이', 'Sequin/Glitter' : '스팽글/글리터',
                           'Denim' : '데님', 'jersey' : '저지', 'tweed' : '트위드', 'velvet' : '벨벳', 'vinyl/PVC' : '비닐/PVC',
                           'wool/cashmere' : '울/캐시미어', 'hair knit' : '헤어니트', 'knit' : '니트', 'lace' : '레이스',
                           'linen' : '린넨', 'messi' : '메시', 'fleece' : '플리스', 'neoprene' : '네오프렌', 'silk' : '실크',
                            'spandex' : '스판덱스', 'jacquard' : '자카드', 'leather' : '가죽', 'chiffon' : '시폰', 'woven' : '우븐'}
    for key, value in korean_material_dict.items():
        if material == key:
            k_material = value
            
    
    # 소매기장
    korean_length_dict = {'Sleeveless shirt' : '민소매', 'Short sleeve' : '반팔', 'Cap':'캡', 'Three-quarter sleeve' : '7부소매', 'Long sleeve' : '긴팔'}
    for key, value in korean_length_dict.items():
        if length == key:
            k_length = value
            
    
    if length is None:
        return k_first_category, k_second_category, k_material
    else:
        return k_first_category, k_second_category, k_material, k_length
      
        
# 분류 결과값 출력 함수
def get_output_class(category_first_class, category_second_class, material_class, length_class=None):
    if length_class is None:
        print('===========================================================================================================')
        print(f'카테고리 : {category_first_class} > {category_second_class} > {material_class}')
        print('===========================================================================================================')
        print(f'1차 카테고리 : {category_first_class} | 2차 카테고리 : {category_second_class} | 소재 : {material_class}')
        output1 = f'카테고리 : {category_first_class} > {category_second_class} > {material_class}'
        output2 = f'( 1차 카테고리 : {category_first_class} | 2차 카테고리 : {category_second_class} | 소재 : {material_class}'
        return [output1, output2]
    else:
        print('===========================================================================================================')
        print(f'카테고리 : {category_first_class} > {length_class} {category_second_class} > {material_class}')
        print('===========================================================================================================')
        print (f'1차 카테고리 : {category_first_class} | 2차 카테고리 : {category_second_class} | 소재 : {material_class} | 소매 기장 : {length_class}')
        output1 = f'카테고리 : {category_first_class} > {length_class} {category_second_class} > {material_class}'
        output2 = f'( 1차 카테고리 : {category_first_class} | 2차 카테고리 : {category_second_class} | 소재 : {material_class} | 소매 기장 : {length_class} )'
        return [output1, output2]


# 점수 높은 class 구하는 함수
# 점수가 높은 class 구하기

def get_best_class(test_image_list):
    
    # 모델별 결과 매칭 가져오기 
    r_df = get_match_df(test_image_list)
    
    # 1차 카테고리 class 구하기
    c1_max_df = r_df.loc[r_df.groupby(['c_first_class'])['c_conf_confidence'].idxmax(skipna=False)]

    
    # 소재 class 구하기
    m_max_df = c1_max_df.loc[c1_max_df.groupby(['c_first_class'])['m_conf_confidence'].idxmax(skipna=False)]

    
    # 소매기장 class 구하기
    # 소매기장 컬럼이 존재하는 경우
    if 'l_class' in m_max_df.columns:
        
        # 하나라도 NaN 아닌 값이 존재할 경우
        if m_max_df['l_class'].isnull().values.any() == True:
            
            # NaN 값 때문에 group by가 안되는 것 방지
            l_temp_df = m_max_df.copy()
            l_temp_df[['l_class','l_conf_confidence']] = l_temp_df[['l_class', 'l_conf_confidence']].fillna(0)
            
            result_df = m_max_df.loc[l_temp_df.groupby(['c_first_class'])['l_conf_confidence'].idxmax(skipna=False)]
            
        else:
            result_df = m_max_df
    else:
        result_df = m_max_df
    
    # 결과 list로 출력하기 위해 생성
    result = []
    
    # 한글화
    for index, row in result_df.iterrows():
        
        
        # 소매기장 열이 존재할 경우
        if 'l_class' in result_df.columns:
            # 소매기장 값 NaN일 경우
            if pd.isna(row['l_class']):
                category_first_class, category_second_class, material_class = \
                get_korean_class(row['c_first_class'], row['c_class'], row['m_class'])
                result.append(get_output_class(category_first_class, category_second_class, material_class))
                
                
            # 소매기장 값 존재할 경우
            else: 
                category_first_class, category_second_class, material_class, length_class = \
                get_korean_class(row['c_first_class'], row['c_class'], row['m_class'], row['l_class'])
                result.append(get_output_class(category_first_class, category_second_class, material_class, length_class))
                
        # 소매기장 열이 존재하지 않을 경우
        else:
            category_first_class, category_second_class, material_class = \
            get_korean_class(row['c_first_class'], row['c_class'], row['m_class'])
            result.append(get_output_class(category_first_class, category_second_class, material_class))
    
    return result