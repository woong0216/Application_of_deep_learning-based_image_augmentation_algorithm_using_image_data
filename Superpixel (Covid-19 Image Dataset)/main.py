import cv2
from img_aug_3 import Img_aug #같은 위치에서 만들었던 모델 img_aug.py를 불러옴 (img_aug_1, img_aug_2, img_aug3)
import os

root_dir = './train/Covid' # 기존 폐 이미지 위치
 
img_path_list = []
possible_img_extension = ['.jpg', '.jpeg', '.png'] # 기존 데이터 확장파일
 
for (root, dirs, files) in os.walk(root_dir):
    if len(files) > 0:
        for file_name in files:
            if os.path.splitext(file_name)[1] in possible_img_extension:
                img_path = root + '/' + file_name
                
                # 파일 확장명 리스트화
                img_path = img_path.replace('./train/Covid/','')
                img_path_list.append(img_path)

aug = Img_aug()		# data augmentation py 선언
augment_num = 3	# 각 기존 그림 파일에 대한 증강 개수 선언

save_path = './augment/bright_color_flip_affine/Covid/' # 저장 위치
dir = "./train/Covid/" # 파일 확장명과 같이 위치 확인

for image in img_path_list:
    img=cv2.imread(dir+image) # 이미지 읽기
    images_aug = aug.seq.augment_images([img for i in range(augment_num)]) # augment_num 개수만큼 증강
    
    for num,aug_img in enumerate(images_aug) :
        cv2.imwrite(save_path+'{}_{}'.format(num,image),aug_img) # 저장
        
print('Complete augmenting images')
