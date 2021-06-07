#-- filename : img_aug_2.py --
# 색상, 밝기 변형

import numpy as np
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
            [# 2.1 segmentation 200개로 분할
             # replace(1.0)이 모든 segment가 평균 색상으로 대체되도록 해줌
             # 각 image당 100에서 200사이로 superpixel 분할함
             self.sometimes(iag.Superpixels(p_replace=(0, 1.0), n_segments=(100,200))), 
             
             # 2.2 blur 색상 변형 처리
             # 5가지 기법이 있기 때문에 random하게 blur 처리할 수 있도록 OneOf 사용
             iag.OneOf(
                 [iag.GaussianBlur(sigma=(0, 3.0)), # sigma를 0에서 3.0사이를 주는 Gaussian 기법 (보통 0이 no blur, 3이 strong blur)
                  iag.AverageBlur(k=(3, 7)), # 보통 kernel을 3에서 7 사이로 주고 샘플링 하는 average 기법
                  iag.MedianBlur(k=(3, 7)), # 보통 kernel을 3에서 7 사이로 주고 샘플링 하는 median 기법
                  iag.BilateralBlur(d=(3,10), sigma_color=(10, 250), sigma_space=(10, 250)), # bilateral filter를 통해 이미지를 blur하게 만듬 
                  iag.MotionBlur(k=7, angle=[-45,45]) # kernel size는 7로 설정하고 angle을 주는 motion blur를 사용함
                  ]),
             
             # 2.3 convolutional 변형
             # 4가지 기법이 있기 때문에 random하게 blend 처리할 수 있도록 OneOf 사용
              iag.OneOf(
                  [iag.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # 빛 변화를 0.75~1.5로 변형
                   iag.Emboss(alpha=(0, 1.0), strength=(0.5, 1.5)), # interval [0.5, 1.5] 및 alpha-blend factor를 0%에서 100%로 변형함
                   iag.EdgeDetect(alpha=(0.5, 1.0)), # alpha-blend factor를 0%에서 100%로 변형하고 edgedetect을 통해 활용함
                   iag.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0)) # alpha-blend factor를 0%에서 100%로 변형하고 방향성을 둔 edgedetect를 활용함
                   ]),
              
              # 2.4 dropout 변형
              # 3가지 기법이 있기 때문에 random하게 dropout 처리할 수 있도록 OneOf 사용
              iag.OneOf([
                  iag.Dropout((0.01, 0.1), per_channel=0.5), # random하게 10%씩 pixel을 제거함
                  iag.CoarseDropout((0.01, 0.1), size_percent=(0.02, 0.05), per_channel=0.2), # pixel을 랜덤하게 10%씩 제거하는데 직사각형 크기로 제거함
                  iag.TotalDropout(0.1)
                  ]),
              
              # 2.5 밝기 변형
              iag.OneOf([
                  iag.AddToBrightness((-30, 30)), # 기본 빛에서 -30에서 30으로 밝기 조절
                  iag.Add((-10, 10), per_channel=0.5), # 기본 값에서 -10에서 10으로 밝기 조절
                  ]),
              
              # 2.6 색깔 변형
              iag.OneOf([
                  iag.Invert(0.05, per_channel=True), # 색깔 channel 변형
                  iag.UniformColorQuantization(n_colors=(2,16)) # discrete interval [4, 16]으로 random하게 color 변경
                                    
                  ]),
           
              
                  ],
                  random_order=True # random하게 순서를 바꾸면서 돌아갈 수 있게 하고 싶을 때 true로 설정 
        )
      