# Application_of_deep_learning-based_image_augmentation_algorithm_using_image_data
## Superpixel (Covid-19 Image Dataset)
### python file
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
### dataset
> train folder
- 기존 train dataset
> test folder
- test dataset
> augment folder
- flip_affine folder: img_aug_1.py와 main.py를 통해 형성
- brigt_color folder: img_aug_2.py와 main.py를 통해 형성
- flip_affine folder: img_aug_3.py와 main.py를 통해 형성
### output
> output folder
- Performance Evaluation.py를 통해 기존 train, aug1(method1), aug2(method2), aug3(method3)의 결과물 도출
