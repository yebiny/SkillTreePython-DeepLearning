import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import IPython

# 모델을 생성합니다. 
# prediction 하는 레이어의 이름을 반환합니다. 
def set_model(weight_file, cfg_file):  
  model = cv2.dnn.readNet(weight_file, cfg_file) # net
  predict_layer_names = [model.getLayerNames()[i[0] - 1] for i in model.getUnconnectedOutLayers()]
  return model, predict_layer_names

# 클래스 이름과 색상을 지정해 줍니다.
# 클래스 색상 리스트 생성시 정수 변환을 오히려 안해야 오류가 안뜨니 참고해 주세요
def set_label(name_file):
  with open(name_file, 'r') as f:
      class_names = [line.strip() for line in f.readlines()]
  
  class_colors = np.random.uniform(0, 255, size=(len(class_names), 3))
  return class_names, class_colors

# 모델, 클래스 정보 등 세팅
######################################################### 
# 실제 이미지 추론 

def get_preds(img, model, predict_layer_names, min_confidence=0.5):
  img_h, img_w, img_c = img.shape
  blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
  model.setInput(blob)
  preds_each_layers = model.forward(predict_layer_names)

  boxes = []
  confidences=[]
  class_ids = []
  for preds in preds_each_layers:
    for pred in preds:
      box, confidence, class_id = pred[:4], pred[4], np.argmax(pred[5:])
      if confidence > min_confidence:
        x_center, y_center, w, h = box
        x_center, w = int(x_center*img_w), int(w*img_w)
        y_center, h = int(y_center*img_h), int(h*img_h)
        x, y = x_center-int(w/2), y_center-int(h/2)
        
        boxes.append([x, y, w, h])
        confidences.append(float(confidence)) # float 처리를 해야 NMSBoxes 함수 사용 가능
        class_ids.append(class_id)
  return boxes, confidences, class_ids

def draw_result(img, 
                boxes, confidences, class_ids,
                class_names, class_colors,
                min_confidence=.5,
                font_size=.6):
  selected_box_idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  for bi, (x, y, w, h) in enumerate(boxes):
    if bi in selected_box_idx:
      class_id = class_ids[bi]
      color = class_colors[class_id]
      class_name = class_names[class_id]

      cv2.rectangle(img, (x, y), (x+w, y+h), color , 2)
      cv2.putText(img, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2 )
  IPython.display.clear_output(wait=True) # 출력결과 매번 지워줌
  cv2_imshow(img)

def img2detect(img, model, predict_layer_names, 
               class_names, class_colors,
               min_confidence=.5,
               font_size=.5):
  boxes, confidences, class_ids = get_preds(img, model, predict_layer_names, min_confidence=0.5)
  draw_result(img, 
              boxes, confidences, class_ids,
              class_names, class_colors,
              min_confidence=min_confidence,
              font_size=font_size)
