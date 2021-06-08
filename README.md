# Application_of_deep_learning-based_image_augmentation_algorithm_using_image_data

## Superpixel (Covid-19 Image Dataset)

### Python file
> img_aug_1.py
- 기본적인 augmentation 기법: 뒤집기(Filp), 회전(Rotate), 크기(Scale), 자르기(Crop), 붙이기(Pad) 변형
> img_aug_2.py
- Superpixel을 적용한 기법: 랜덤하게 흐리게(Blur), 합성곱(Convolutional), 희석(Dropout), 밝기(Brightness), 색깔(Color) 변형
> img_aug_3.py
- 기본적인 augmentation + Superpixel을 적용한 기법
> main.py
- img_aug_1.py, img_aug_2.py, img_aug_3.py를 실행한 후 main.py를 통해 train data를 augmentation을 해주는 프로그램 (augment 폴더에 기입)
> Performance Evaluation.py
- resNet50를 통해 image classification 수행
- augment 폴더의 augment train dataset과 test dataset을 통해 loss 및 accuracy 측정
- 
### Dataset
> train folder
- 기존 train dataset
> test folder
- test dataset
> augment folder
- flip_affine folder: img_aug_1.py와 main.py를 통해 형성
- brigt_color folder: img_aug_2.py와 main.py를 통해 형성
- flip_affine folder: img_aug_3.py와 main.py를 통해 형성
- 
### Output
> output folder
- Performance Evaluation.py를 통해 기존 train, aug1(method1), aug2(method2), aug3(method3)의 결과물 도출


## DCGAN (Skin Canner Dataset)

### Dataset
cvd_19: COVID-19 X-ray 이미지 데이터셋 폴더. <br>
skincancer: 피부암 조직 이미지 데이터셋 폴더.
* 각 데이터셋은 데이터셋 이름 - train or test 폴더 - label 명 폴더로 이루어져 있어야 정상적으로 데이터 로드가 가능합니다.
* 깃허브에는 일부 데이터만 포함되어 있으며, 전체 데이터셋은 구글 드라이브 링크를 통해 다운받을 수 있습니다.

### Python scripts
- DCGAN.ipynb: 이미지 생성을 위한 DCGAN 노트북 파일.
- CNN_classifier.ipynb: 성능 평가를 위한 CNN 분류기 노트북 파일.
- utils.ipynb: 결과 분석을 위한 노트북 파일(평균과 표준편차 계산, learning curve 그리기).
- skin_dataset.py: 피부암 조직 이미지 데이터를 불러오기 위한 custom dataset 파일.
- covid_dataset.py:  COVID-19 X-ray 이미지 데이터를 불러오기 위한 custom dataset 파일.

* pytorch 버전: 1.8.1+cu111
