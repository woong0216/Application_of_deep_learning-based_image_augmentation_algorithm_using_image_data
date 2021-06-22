# Application of deep learning-based image augmentation algorithm using image data

## Background
- 딥러닝 모델은 학습 과정에서 훈련 데이터에만 모델이 최적화되고 실제 데이터에는 잘 작동하지 않는 과적합(overfitting) 현상이 발생할 수 있다.
- 충분한 수의 데이터를 확보하는 것도 중요하다.
- 의료 데이터는 개인 정보 문제 등으로 인해 다량의 데이터를 수집하기가 어렵다.
- 레이블을 만들기 위해 의료인의 노동이 필요하기 때문에 많은 비용이 든다는 어려움이 있다.

## Propose
- 데이터 증강(data augmentation)은 이렇게 데이터를 구하기 어려운 경우 데이터의 다양성을 학습시켜 딥러닝의 성능을 높이고 싶을 때 사용하는 기법이다.
- 데이터의 본질을 변형시키지 않는 선상에서 노이즈를 이용해 데이터의 개수를 늘리는 방법이다.
- 이 보고서에서는, 2종류의 진단용 의료 데이터에 대해 Superpixel과 DCGAN 두 종류의 데이터 증강 기법을 적용해보고 성능 향상이 이루어지는지 확인한다.
- epoch 등을 바꾸어 보며 어떠한 조건에서 성능 향상이 더 잘 이루어지는지도 확인해보고자 한다.

## Dataset
### Dataset 1 (Covie-19 Image)

![Covid-19 Dataset](https://user-images.githubusercontent.com/63955072/122932221-84bedf00-d3a8-11eb-9890-88ee5496642f.png)

- train folder
> 기존 train dataset <br/>

- test folder
> test dataset <br/>

- augment folder
> flip_affine folder: img_aug_1.py와 main.py를 통해 형성 <br/>
> brigt_color folder: img_aug_2.py와 main.py를 통해 형성 <br/>
> flip_affine folder: img_aug_3.py와 main.py를 통해 형성 <br/>

### Dataset 2 (Skin Cancer Image)

![Skin Cancer Dataset](https://user-images.githubusercontent.com/63955072/122932291-93a59180-d3a8-11eb-9e2c-f865657a75c3.png)

- covid_19: COVID-19 X-ray 이미지 데이터셋 폴더. <br>
- skincancer: 피부암 조직 이미지 데이터셋 폴더. <br>
> 각 데이터셋은 데이터셋 이름 - train or test 폴더 - label 명 폴더로 이루어져 있어야 정상적으로 데이터 로드가 가능합니다. <br>
> 깃허브에는 일부 데이터만 포함되어 있으며, 전체 데이터셋은 구글 드라이브 링크를 통해 다운받을 수 있습니다. <br>

## Method
### Superpixel (Covid-19 Image Dataset)
#### Python file
- img_aug_1.py
> 기본적인 augmentation 기법: 뒤집기(Filp), 회전(Rotate), 크기(Scale), 자르기(Crop), 붙이기(Pad) 변형 <br/>

- img_aug_2.py
> Superpixel을 적용한 기법: 랜덤하게 흐리게(Blur), 합성곱(Convolutional), 희석(Dropout), 밝기(Brightness), 색깔(Color) 변형 <br/>

- img_aug_3.py
> 기본적인 augmentation + Superpixel을 적용한 기법 <br/>

- main.py
> img_aug_1.py, img_aug_2.py, img_aug_3.py를 실행한 후 main.py를 통해 train data를 augmentation을 해주는 프로그램 (augment 폴더에 기입) <br/>

- Performance Evaluation.py
> resNet50를 통해 image classification 수행 <br/>
> augment 폴더의 augment train dataset과 test dataset을 통해 loss 및 accuracy 측정 <br/>

### DCGAN (Skin Canner Dataset)
#### Python scripts
- DCGAN.ipynb: 이미지 생성을 위한 DCGAN 노트북 파일.
- CNN_classifier.ipynb: 성능 평가를 위한 CNN 분류기 노트북 파일.
- utils.ipynb: 결과 분석을 위한 노트북 파일(평균과 표준편차 계산, learning curve 그리기).
- skin_dataset.py: 피부암 조직 이미지 데이터를 불러오기 위한 custom dataset 파일.
- covid_dataset.py:  COVID-19 X-ray 이미지 데이터를 불러오기 위한 custom dataset 파일.

## Output
- output folder
> Performance Evaluation.py를 통해 기존 train, aug1(method1), aug2(method2), aug3(method3)의 결과물 도출 <br/>

- Covid-19 Image Dataset
> Data augmentation results using superpixel <br/>
>> Sample <br/>

![Data augmentation results using superpixel](https://user-images.githubusercontent.com/63955072/122930753-365d1080-d3a7-11eb-82be-624249722ae9.PNG)

- Skin Canner Dataset <br/>
> Data augmenation results using superpixel and DCGAN <br/>
>> Sample <br/>

![Data augmenation results using superpixel and DCGAN](https://user-images.githubusercontent.com/63955072/122931445-dca91600-d3a7-11eb-8f00-5d6b61d76bf3.PNG)

## Analysis
- Covid-19 Image Dataset
> Data accuracy results using superpixel <br/>

![Covid-19 Dataset Result](https://user-images.githubusercontent.com/63955072/122932821-057ddb00-d3a9-11eb-8d0e-b77b0f9601e2.png)

- Skin Canner Dataset <br/>
> Data accuracy results using superpixel and DCGAN <br/>

![Skin Cancer Dataset Result](https://user-images.githubusercontent.com/63955072/122933094-48d84980-d3a9-11eb-807f-1e38dd38f3b0.png)

## Conclusion
- Covid-19 X-ray 데이터셋에서는 Superpixel에서 random 변형 기법(Aug 3) 세팅을 적용했을 때에 성능과 recall 향상율이 가장 높았다. 
- 하지만, DCGAN을 사용할 경우 이미지의 해상도가 크게 감소하기 때문에 DCGAN은 고해상도 이미지에 활용하기 어려울 것으로 보인다.
- 피부암 조직 데이터셋에서는, 모든 세팅에서 정확도 향상은 큰 폭으로 이루어지지 않았지만 recall 값의 변화는 10%의 GAN 데이터를 사용했을 때와, 20%의 Superpixel 증강 데이터를 활용했을 때에 상당한 향상이 있었다.
