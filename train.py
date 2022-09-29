import numpy as np
import json
import cv2
import os
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog  #注册Metadata
from detectron2.data import DatasetCatalog   #注册资料集
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode  #标记方式
from matplotlib import pyplot as plt

def get_balloon_dicts(img_dir):
    json_file=os.path.join(img_dir,'via_region_data.json')
    with open(json_file) as f:
        imgs_anns=json.load(f)
    dataset_dicts=[]
    for idx,v in enumerate(imgs_anns.values()):
        record={}  #标准字典档

        filename=os.path.join(img_dir,v['filename'])
        height,width=cv2.imread(filename).shape[:2]  #获取尺寸

        record['file_name']=filename
        
        record['image_id']=idx
        record['height']=height
        record['width']=width

        annos=v['regions']  #范围

        objs=[]
        for _,anno in annos.items():
            assert not anno['region_attributes']
            anno=anno['shape_attributes']
            px=anno['all_points_x']
            py=anno['all_points_y']
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)] #标记框框
            poly=[p for x in poly for p in x]
            obj={
                'bbox':[np.min(px),np.min(py),np.max(px),np.max(py)], #左上角坐标和右下角坐标
                'bbox_mode':BoxMode.XYXY_ABS,
                'segmentation':[poly],
                'category_id':0, #类别id
                'iscrowd':0    #只有一个类别
            }
            objs.append(obj)
        record['annotations']=objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ['train','val']:  #注册数据集
    DatasetCatalog.register('balloon_'+d,lambda d=d: get_balloon_dicts('./balloon/'+d))
    MetadataCatalog.get('balloon_'+d).set(thing_classes=['balloon'])

cfg=get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #预设档，参数
cfg.DATASETS.TRAIN=('balloon_train',)  #训练集
cfg.DATASETS.TEST=('balloon_val',)  #测试集
cfg.DATALOADER.NUM_WORKERS=2   #执行序，0是cpu
cfg.SOLVER.IMS_PER_BATCH=2  #每批次改变的大小
cfg.SOLVER.BASE_LR=0.01  #学习率
cfg.SOLVER.STEPS=(4000,)
cfg.SOLVER.MAX_ITER=6000  #最大迭代次数
cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  #迁移基础
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=128  #default:512 批次大小
cfg.MODEL.ROI_HEADS.NUM_CLASSES=1  #一类
# cfg.MODEL.DEVICE='cpu'  #注释掉此项，系统默认使用NVidia的显卡
cfg.OUTPUT_DIR = './temp_model'

os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
trainer=DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("finished !!!")