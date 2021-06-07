# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Tesorflow에서 GPU Setting 생성

# train set 및 test set 불러오기
# 기존: "./train"
# method 1: "./augment/flip_affine"
# method 1: "./augment/bright_color"
# method 3: "./augment/bright_color_flip_affine"
train_data_dir = "./augment/bright_color_flip_affine" # 바꾸면서 성능 평가
test_data_dir = "./test"

# 각 라벨링 된 image 불러오기
covid_images = [os.path.join(train_data_dir, 'Covid', path) for path in os.listdir(train_data_dir + '/Covid')]
normal_images = [os.path.join(train_data_dir, 'Normal', path) for path in os.listdir(train_data_dir + '/Normal')]
viral_pneumonia_images = [os.path.join(train_data_dir, 'Viral Pneumonia', path) for path in os.listdir(train_data_dir + '/Viral Pneumonia')]

# 이미지 데이터 정규화
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 크기 및 배치 사이즈 규정
img_width, img_height = 224, 224
batch_size = 8

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

# 각 trainset 개수 확인
print("The individual class count in train set is ", Counter(train_generator.classes))
print("The individual class count in test set is ", Counter(test_generator.classes))

# 기본 모델 생성 (ResNet50을 사용한 ImageNet 클래스 분류)
base_model = tf.keras.applications.resnet.ResNet50(weights='imagenet', include_top=False) # ResNet50 이미지 분류 모델 사용
global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model.output) # GAP 사용 (사이즈에 대한 구애를 받지 않기 위해)
output = tf.keras.layers.Dense(3, activation='softmax')(global_avg_pooling) # 출력 뉴런 3개 및 활성화 함수로 softmax 구현
model = tf.keras.Model(inputs=base_model.input, outputs=output)
optimizer = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0.01) # learning rate=0.001, momentum = 0.9 decaying factor = 0.01
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

# 모델 epoch 설정 후 성능 측정
history = model.fit(train_generator, epochs=50, validation_data=test_generator, verbose=1) # epoch 50으로 지정

# plot 그리기
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(50), history.history['loss'], label='train loss')
plt.plot(range(50), history.history['val_loss'], label='val loss')
plt.title('loss change with epoch change ')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(50), history.history['accuracy'], label='train accuracy')
plt.plot(range(50), history.history['val_accuracy'], label='val accuracy')
plt.title('accuracy change with epoch change')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# 정확도 및 F1 score를 측정하기 위해 다양한 지표 설정
all_batches = []
count = 1
for batch in tqdm(test_generator):
    all_batches.append(batch)
    count = count + 1
    if count == 33:
        break

all_y_hats = []
all_y = []
for X, y in all_batches:
    y_hat = model.predict(X)
    y_hat = np.argmax(y_hat, 1)
    all_y_hats.extend(list(y_hat))
    all_y.extend(list(np.argmax(y, 1)))
target_names=['Covid','Normal','Viral Pneumonia']
    
print(classification_report(all_y, all_y_hats, target_names=target_names))
    
# heatmap을 통해 시각화
sns.heatmap(confusion_matrix(all_y, all_y_hats), annot=True, cmap='mako')
plt.xticks(np.arange(0.5, len(target_names), 1), target_names)
plt.yticks(np.arange(0.5, len(target_names), 1), target_names)
plt.title('Confusion Matrix on Test Set')
plt.show()