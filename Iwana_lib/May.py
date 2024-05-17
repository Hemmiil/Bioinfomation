def polygon2dataset(root_dir, label_dict, json_files_dir = None, train_rate = 0.7):
    #root_dir -> YOLOdatasetを作成したいディレクトリパスを入力
    #label_dict -> ラベルのクラス名と番号の辞書
        #e.g. {"EZ_skin" : "0", "NK_skin" : "1"}
    #json_files_dir -> jsonファイルを格納しているディレクトリ 指定なしだとroot_dirから探す
    #split_rate -> 訓練に充てるデータの割合 float で入力 None だと全て訓練
    import os
    import random
    from sklearn.model_selection import train_test_split
    import glob
    import numpy as np
    import base64
    import json
    from labelme import utils
    import matplotlib.pyplot as plt
    import yaml
    import shutil

    #root_dir = "../data/segmentation"
    #保存先のディレクトリを生成

    if os.path.exists(f"{root_dir}/YOLOdataset"):
        shutil.rmtree(f"{root_dir}/YOLOdataset")
    os.mkdir(f"{root_dir}/YOLOdataset")
    for ctg in ["images", "labels"]:
        os.mkdir(f"{root_dir}/YOLOdataset/{ctg}")
        os.mkdir(f"{root_dir}/YOLOdataset/{ctg}/train")
        os.mkdir(f"{root_dir}/YOLOdataset/{ctg}/val")
    
    #jsonファイルのパスの取得　root_dirに格納されている前提
    if not json_files_dir:
        json_files = glob.glob(f"{root_dir}/*.json")
    else:
        josn_files = glob.glob(f"{json_files_dir}/*.json")

    #訓練、評価用ファイルを分割
    if train_rate == None:
        train_files = json_files
        val_files = []
    else:
        train_files, val_files = train_test_split(json_files, train_size=train_rate[0])
    for tv, tv_files in zip (["train", "val"], [train_files, val_files]):
        for path in tv_files:
            bs = os.path.basename(path).split(".")[0]
            # JSONファイルを読み込む
            with open(path, "rb") as f:
                data = json.load(f)
            
            #画像ファイルの保存
            img_b64 = data["imageData"]
            img_data = base64.b64decode(img_b64)
            img_pil = utils.img_data_to_pil(img_data)
            img_pil.save(f"{root_dir}/YOLOdataset/images/{tv}/{bs}.jpg")
            
            #Labelテキストファイルの作成
            res = []
            points = data["shapes"][0]["points"]
            label = data["shapes"][0]["label"]
            label_id = label_dict[label]
            w, h = data["imageWidth"], data["imageHeight"]
            res.append(str(label_id))

            for i in range(len(points)):
                x,y = list(map(lambda x: round(x, 4), [points[i][0]/w, points[i][1]/h]))
                x_y = " ".join(list(map(str, [x,y])))
                res.append(x_y)

            res = " ".join(res)
            with open(f"{root_dir}/YOLOdataset/labels/{tv}/{bs}.txt", "w") as f:
                f.write(res)
    #yamlファイルの作成
    yml = {}
    yml["path"] = root_dir
    yml['train'] = f"{root_dir}/segmentation/YOLOdataset/images/train"
    yml["val"] = f"{root_dir}/segmentation/YOLOdataset/images/val"

    yml["names"] = {
        0 : "EZ_mask"
    }

    #yamlファイルの書き出し
    with open(f'{root_dir}/YOLOdataset/yaml.yaml', 'wb') as f:
        yaml.dump(yml, f, encoding='utf-8', allow_unicode=True)


def YOLO_train(data, epochs, project, name, model_name = "n"):
    # data -> データセットのyamlファイルのパス
    # epochs -> 学習回数
    # project -> 結果を出力するディレクトリのパス
    # name -> project の中に生成される、結果をまとめたディレクトリの名前
    #model_name -> str モデルの種類を入力 n,s,m,l,x の中から択一
    from ultralytics import YOLO
    # Load a model
    model = YOLO(f'yolov8{model_name}-seg.yaml')  # build a new model from YAML
    model = YOLO(f'yolov8{model_name}-seg.pt')  # load a pretrained model (recommended for training)
    model = YOLO(f'yolov8{model_name}-seg.yaml').load(f'yolov8{model_name}.pt')  # build from YAML and transfer weights
    # Train the model
    results = model.train(data=data, epochs=epochs, imgsz=640,
                        project=project, name=name)