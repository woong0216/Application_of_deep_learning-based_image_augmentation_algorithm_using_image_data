#-- filename : img_aug_1.py --
# 뒤집기, 회전, 크기 변형

import numpy as np
import imgaug as ia
import imgaug.augmenters as iag

# numpy를 통해 각 그림 파일에도 random화
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8) # image type

# https://imgaug.readthedocs.io/en/latest/source/api.html의 API code를 참고하여 생성함

class Img_aug :
    def __init__(self) :
        # sometimes를 통해 Random하게 augmentation을 수행하게 해줌
        self.sometimes = lambda aug: iag.Sometimes(0.5, aug) # 랜덤화 함수

        # Sequential를 통해 다양한 기법을 연달아 적용할 수 있음
        # 또한, random_order 매개 변수를 통해 하위 항목을 임의 순서대로 제공 가능하게 해줌
        # 팀프로젝트에서 제안한 것 처럼 뒤집기, 회전, 색상, 밝기 변환 등을 조합 및 응용을 구현하고자 함
        # 3가지 실험을 하고자 함 (모두 랜덤화하게 나타나도록 random_order=True)
        # 흑백 사진인 폐가 과연 색상 밝기를 조절하는 것이 괜찮은 방법인지 의구시을 가짐
        # 1.뒤집기, 회전, 크기 변형만 한 경우
        # 1.1 뒤집기 (상하, 좌우)할
        # 1.2 회전
        # 1.3 크기 변형
        # 2.색상, 밝기만 한 경우
        # 2.1 segmentation 200개로 분할
        # 2.2 blur 색상 변형
        # 2.3 convolutional 변형
        # 2.4 dropout 변형
        # 2.5 밝기 변형
        # 2.6 색깔 변형
        # 3.뒤집기, 회전, 색상, 밝기 등 모두 한 경우
        
        self.seq = iag.Sequential(
            [# 1.1 뒤집기 (상하, 좌우)
             iag.Fliplr(0.5), # 이미지를 좌우 대칭으로 바꾸는 것 (뒤집기1)
             iag.Flipud(0.5), # 이미지를 상하 대칭으로 바꾸는 것 (뒤집기2)
             
             # 1.2 회전
             # range 범위내에 random하게 실행하도록 self.sometimes 실행
             self.sometimes(iag.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, # 크기를 50%~150%까지 축소 및 확대
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # x, y축 독립적으로 -20% to 20% 축소 및 확대
                        rotate=(-45, 45), # 회전 각도 -45~45도
                        shear=(-16, 16), # 그림 기울기 -16~16도
                        order=[0, 1], # neighbour or bilinear interpolation 이용
                        cval=(0, 255), # cval 0~255 사용
                        mode=ia.ALL)),
             
             # 1.3 크기 변형
             # range 범위내에 random하게 실행하도록 self.sometimes 실행
             self.sometimes(iag.CropAndPad(percent=(-0.1, 0.1), # height 및 width -10~10%정도 자르거나 붙이기
                            pad_mode=ia.ALL, # imagug 모두 이행
                            pad_cval=(0, 255))), # 0에서 255구간
             
            ],
            random_order=True # random하게 순서를 바꾸면서 돌아갈 수 있게 하고 싶을 때 true로 설정 
        )