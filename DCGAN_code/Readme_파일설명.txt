
<DCGAN 파트>

cvd_19: COVID-19 X-ray 이미지 데이터셋 폴더.
skincancer: 피부암 조직 이미지 데이터셋 폴더.
* 각 데이터셋은 데이터셋 이름 - train or test 폴더 - label 명 폴더로 이루어져 있어야 정상적으로 데이터 로드가 가능합니다.

- DCGAN.ipynb: 이미지 생성을 위한 DCGAN 노트북 파일.
- CNN_classifier.ipynb: 성능 평가를 위한 CNN 분류기 노트북 파일.
- utils.ipynb: 결과 분석을 위한 노트북 파일(평균과 표준편차 계산, learning curve 그리기).
- skin_dataset.py: 피부암 조직 이미지 데이터를 불러오기 위한 custom dataset 파일.
- covid_dataset.py:  COVID-19 X-ray 이미지 데이터를 불러오기 위한 custom dataset 파일.

* pytorch 버전: 1.8.1+cu111
