

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import cv2
import imageio
import os
from tqdm import tqdm
import gc
import random
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


datapath = 'data'

directories = []
for directory in os.listdir(datapath):
    directories.append(directory)
print('Classes Present : ',list(directories))
# all_benign,all_early,all_pre,all_pro,brain_glioma,brain_menin,brain_tumor,breast_benign,breast_malignant,cervix_dyk,cervix_koc,cervix_mep,cervix_pab,cervix_sfi,colon_aca,colon_bnt,kidney_normal,kidney_tumor,lung_aca,lung_bnt,lung_scc,lymph_cll,lymph_fl,lymph_mcl,oral_normal,oral_scc

all_benign_files=[]
all_early_files=[]
all_pre_files=[]
all_pro_files=[]
brain_glioma_files=[]
brain_menin_files=[]
brain_tumor_files=[]
breast_benign_files=[]
breast_malignant_files=[]
cervix_dyk_files=[]
cervix_koc_files=[]
cervix_mep_files=[]
cervix_pab_files=[]
cervix_sfi_files=[]
colon_aca_files=[]
colon_bnt_files=[]
kidney_normal_files=[]
kidney_tumor_files=[]
lung_aca_files=[]
lung_bnt_files=[]
lung_scc_files=[]
lymph_cll_files=[]
lymph_fl_files=[]
lymph_mcl_files=[]
oral_normal_files=[]
oral_scc_files=[]

for directory in directories:
    for files in os.listdir(os.path.join(datapath,directory)):
        if directory == 'all_benign':
            all_benign_files.append(os.path.join(datapath,'all_benign',files))
        elif directory == 'all_early':
            all_early_files.append(os.path.join(datapath,'all_early',files))
        elif directory == 'all_pre':
            all_pre_files.append(os.path.join(datapath,'all_pre',files))
        elif directory == 'all_pro':
            all_pro_files.append(os.path.join(datapath,'all_pro',files))
        elif directory == 'brain_glioma':
            brain_glioma_files.append(os.path.join(datapath,'brain_glioma',files))
        elif directory == 'brain_menin':
            brain_menin_files.append(os.path.join(datapath,'brain_menin',files))
        elif directory == 'brain_tumor':
            brain_tumor_files.append(os.path.join(datapath,'brain_tumor',files))
        elif directory == 'breast_benign':
            breast_benign_files.append(os.path.join(datapath,'breast_benign',files))
        elif directory == 'breast_malignant':
            breast_malignant_files.append(os.path.join(datapath,'breast_malignant',files))
        elif directory == 'cervix_dyk':
            cervix_dyk_files.append(os.path.join(datapath,'cervix_dyk',files))
        elif directory == 'cervix_koc':
            cervix_koc_files.append(os.path.join(datapath,'cervix_koc',files))
        elif directory == 'cervix_mep':
            cervix_mep_files.append(os.path.join(datapath,'cervix_mep',files))
        elif directory == 'cervix_pab':
            cervix_pab_files.append(os.path.join(datapath,'cervix_pab',files))
        elif directory == 'cervix_sfi':
            cervix_sfi_files.append(os.path.join(datapath,'cervix_sfi',files))
        elif directory == 'colon_aca':
            colon_aca_files.append(os.path.join(datapath,'colon_aca',files))
        elif directory == 'colon_bnt':
            colon_bnt_files.append(os.path.join(datapath,'colon_bnt',files))
        elif directory == 'kidney_normal':
            kidney_normal_files.append(os.path.join(datapath,'kidney_normal',files))
        elif directory == 'kidney_tumor':
            kidney_tumor_files.append(os.path.join(datapath,'kidney_tumor',files))
        elif directory == 'lung_aca':
            lung_aca_files.append(os.path.join(datapath,'lung_aca',files))
        elif directory == 'lung_bnt':
            lung_bnt_files.append(os.path.join(datapath,'lung_bnt',files))
        elif directory == 'lung_scc':
            lung_scc_files.append(os.path.join(datapath,'lung_scc',files))
        elif directory == 'lymph_cll':
            lymph_cll_files.append(os.path.join(datapath,'lymph_cll',files))
        elif directory == 'lymph_fl':
            lymph_fl_files.append(os.path.join(datapath,'lymph_fl',files))
        elif directory == 'lymph_mcl':
            lymph_mcl_files.append(os.path.join(datapath,'lymph_mcl',files))
        elif directory == 'oral_normal':
            oral_normal_files.append(os.path.join(datapath,'oral_normal',files))
        elif directory == 'oral_scc':
            oral_scc_files.append(os.path.join(datapath,'oral_scc',files))

print('Total all_benign :', len( all_benign_files))
print('Total all_early :', len( all_early_files))
print('Total all_pre :', len( all_pre_files))
print('Total all_pro :', len( all_pro_files))
print('Total brain_glioma :', len( brain_glioma_files))
print('Total brain_menin :', len( brain_menin_files))
print('Total brain_tumor :', len( brain_tumor_files))
print('Total breast_benign :', len( breast_benign_files))
print('Total breast_malignant :', len( breast_malignant_files))
print('Total cervix_dyk :', len( cervix_dyk_files))
print('Total cervix_koc :', len( cervix_koc_files))
print('Total cervix_mep :', len( cervix_mep_files))
print('Total cervix_pab :', len( cervix_pab_files))
print('Total cervix_sfi :', len( cervix_sfi_files))
print('Total colon_aca :', len( colon_aca_files))
print('Total colon_bnt :', len( colon_bnt_files))
print('Total kidney_normal :', len( kidney_normal_files))
print('Total kidney_tumor :', len( kidney_tumor_files))
print('Total lung_aca :', len( lung_aca_files))
print('Total lung_bnt :', len( lung_bnt_files))
print('Total lung_scc :', len( lung_scc_files))
print('Total lymph_cll :', len( lymph_cll_files))
print('Total lymph_fl :', len( lymph_fl_files))
print('Total lymph_mcl :', len( lymph_mcl_files))
print('Total oral_normal :', len( oral_normal_files))
print('Total oral_scc :', len( oral_scc_files))


random_num=random.randint(0,len(all_pre_files))
all_benign_pic= all_benign_files[random_num]
all_early_pic= all_early_files[random_num]
all_pre_pic= all_pre_files[random_num]
all_pro_pic= all_pro_files[random_num]
brain_glioma_pic= brain_glioma_files[random_num]
brain_menin_pic= brain_menin_files[random_num]
brain_tumor_pic= brain_tumor_files[random_num]
breast_benign_pic= breast_benign_files[random_num]
breast_malignant_pic= breast_malignant_files[random_num]
cervix_dyk_pic= cervix_dyk_files[random_num]
cervix_koc_pic= cervix_koc_files[random_num]
cervix_mep_pic= cervix_mep_files[random_num]
cervix_pab_pic= cervix_pab_files[random_num]
cervix_sfi_pic= cervix_sfi_files[random_num]
colon_aca_pic= colon_aca_files[random_num]
colon_bnt_pic= colon_bnt_files[random_num]
kidney_normal_pic= kidney_normal_files[random_num]
kidney_tumor_pic= kidney_tumor_files[random_num]
lung_aca_pic= lung_aca_files[random_num]
lung_bnt_pic= lung_bnt_files[random_num]
lung_scc_pic= lung_scc_files[random_num]
lymph_cll_pic= lymph_cll_files[random_num]
lymph_fl_pic= lymph_fl_files[random_num]
lymph_mcl_pic= lymph_mcl_files[random_num]
oral_normal_pic= oral_normal_files[random_num]
oral_scc_pic= oral_scc_files[random_num]
all_benign_data=imageio.imread( all_benign_pic)
all_early_data=imageio.imread( all_early_pic)
all_pre_data=imageio.imread( all_pre_pic)
all_pro_data=imageio.imread( all_pro_pic)
brain_glioma_data=imageio.imread( brain_glioma_pic)
brain_menin_data=imageio.imread( brain_menin_pic)
brain_tumor_data=imageio.imread( brain_tumor_pic)
breast_benign_data=imageio.imread( breast_benign_pic)
breast_malignant_data=imageio.imread( breast_malignant_pic)
cervix_dyk_data=imageio.imread( cervix_dyk_pic)
cervix_koc_data=imageio.imread( cervix_koc_pic)
cervix_mep_data=imageio.imread( cervix_mep_pic)
cervix_pab_data=imageio.imread( cervix_pab_pic)
cervix_sfi_data=imageio.imread( cervix_sfi_pic)
colon_aca_data=imageio.imread( colon_aca_pic)
colon_bnt_data=imageio.imread( colon_bnt_pic)
kidney_normal_data=imageio.imread( kidney_normal_pic)
kidney_tumor_data=imageio.imread( kidney_tumor_pic)
lung_aca_data=imageio.imread( lung_aca_pic)
lung_bnt_data=imageio.imread( lung_bnt_pic)
lung_scc_data=imageio.imread( lung_scc_pic)
lymph_cll_data=imageio.imread( lymph_cll_pic)
lymph_fl_data=imageio.imread( lymph_fl_pic)
lymph_mcl_data=imageio.imread( lymph_mcl_pic)
oral_normal_data=imageio.imread( oral_normal_pic)
oral_scc_data=imageio.imread( oral_scc_pic)

fig,axs=plt.subplots(6,5)
axs[0,0].imshow(all_benign_data)
axs[0,1].imshow(all_early_data)
axs[0,2].imshow(all_pre_data)
axs[0,3].imshow(all_pro_data)
axs[0,4].imshow(brain_glioma_data)
axs[1,0].imshow(brain_menin_data)
axs[1,1].imshow(brain_tumor_data)
axs[1,2].imshow(breast_benign_data)
axs[1,3].imshow(breast_malignant_data)
axs[1,4].imshow(cervix_dyk_data)
axs[2,0].imshow(cervix_koc_data)
axs[2,1].imshow(cervix_mep_data)
axs[2,2].imshow(cervix_pab_data)
axs[2,3].imshow(cervix_sfi_data)
axs[2,4].imshow(colon_aca_data)
axs[3,0].imshow(colon_bnt_data)
axs[3,1].imshow(kidney_normal_data)
axs[3,2].imshow(kidney_tumor_data)
axs[3,3].imshow(lung_aca_data)
axs[3,4].imshow(lung_bnt_data)
axs[4,0].imshow(lung_scc_data)
axs[4,1].imshow(lymph_cll_data)
axs[4,2].imshow(lymph_fl_data)
axs[4,3].imshow(lymph_mcl_data)
axs[4,4].imshow(oral_normal_data)
axs[5,0].imshow(oral_scc_data)

axs[0,0].set_xlabel(' all_benign')
axs[0,1].set_xlabel(' all_early')
axs[0,2].set_xlabel(' all_pre')
axs[0,3].set_xlabel(' all_pro')
axs[0,4].set_xlabel(' brain_glioma')
axs[1,0].set_xlabel(' brain_menin')
axs[1,1].set_xlabel(' brain_tumor')
axs[1,2].set_xlabel(' breast_benign')
axs[1,3].set_xlabel(' breast_malignant')
axs[1,4].set_xlabel(' cervix_dyk')
axs[2,0].set_xlabel(' cervix_koc')
axs[2,1].set_xlabel(' cervix_mep')
axs[2,2].set_xlabel(' cervix_pab')
axs[2,3].set_xlabel(' cervix_sfi')
axs[2,4].set_xlabel(' colon_aca')
axs[3,0].set_xlabel(' colon_bnt')
axs[3,1].set_xlabel(' kidney_normal')
axs[3,2].set_xlabel(' kidney_tumor')
axs[3,3].set_xlabel(' lung_aca')
axs[3,4].set_xlabel(' lung_bnt')
axs[4,0].set_xlabel(' lung_scc')
axs[4,1].set_xlabel(' lymph_cll')
axs[4,2].set_xlabel(' lymph_fl')
axs[4,3].set_xlabel(' lymph_mcl')
axs[4,4].set_xlabel(' oral_normal')
axs[5,0].set_xlabel(' oral_scc')

plt.savefig('Images.png')
# plt.show()

gc.collect()
generator = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)
train_ds = generator.flow_from_directory(
    'data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_ds = generator.flow_from_directory(
    'data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
checkpoint_filepath = 'checkpoints_all'
callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=True,monitor='val_accuracy',mode='max',save_best_only=True)
]
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model_resnet.layers:
    layer.trainable = False
x = base_model_resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(26, activation='softmax')(x)
model = Model(inputs=base_model_resnet.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds,verbose = 1,epochs = 10,batch_size = 32,validation_data = val_ds,callbacks = callback)
model.save('total_classify.h5')

