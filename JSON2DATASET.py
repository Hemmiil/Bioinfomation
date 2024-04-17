class json2img():
    def __init__(self, json_path, rate=0.15, save_fig=False):
        self.json_path = json_path
        self.json_data = ""
        self.shape_dict = {}
        self.params = {}
        self.rate = rate
        self.save_fig = save_fig
    def input(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        import pandas as pd
        import glob
        import json
        import base64
        import io
        from labelme import utils
        from pandas import DataFrame
        json_path_list = glob.glob(self.json_path)
        json_data = json.load(open(self.json_path, "r"))
        self.json_data = json_data

    def json2png(self):
        import numpy as np
        import base64
        import json
        from labelme import utils
        self.input()
        img_b64 = self.json_data["imageData"]
        img_data = base64.b64decode(img_b64)
        img_pil = utils.img_data_to_pil(img_data)
        self.img = np.array(img_pil)

    def json2polygon(self):
        import numpy as np
        self.input()
        shape_dict = dict()
        for info in self.json_data["shapes"]:
            shape_dict[info["group_id"]] = np.array(info["points"]).astype(np.int32)
        self.params["shape"] = shape_dict

        #最も右のキーポイントと最も左のキーポイントの間の20％を正方形の長さとする
        rate = self.rate
        length = abs(int((shape_dict[4][0,0] - shape_dict[1][0,0])*rate))
        self.params["length"] = length

        #４本の線で区切られた３つの領域から長方形を切り出す
        self.params["rectangles"] = dict()
        for i in range(1, 4, 1):
            line1 = self.params["shape"][i]
            line2 = self.params["shape"][i+1]
            rectangle = np.array([
                [min(line1[0,0], line2[0,0]), max(line1[0,1], line2[0,1])], #左上の座標(x, y)
                [max(line1[1,0], line2[1,0]), min(line1[1,1], line2[1,1])]  #右下の座標(x, y)
            ])
            self.params["rectangles"][f"rectangle{i}"] = rectangle


    def crop(self):
        import numpy as np
        self.input()
        self.json2polygon()
        #それぞれのrectangleから４隅の正方形を切り出す
        L = self.params["length"]
        self.params["square"] = dict()
        for rectangle in [ "rectangle1", "rectangle2", "rectangle3"]:
            square = self.params["rectangles"][rectangle]
            [[x0, y0],[x1, y1]] = square
            square_a = np.array([
                [x0,y0],[x0+L,y0+L]
            ])
            square_b = np.array([
                [x1-L,y0],[x1,y0+L]
            ])
            square_c = np.array([
                [x0,y1-L],[x0+L,y1]
            ])
            square_d = np.array([
                [x1-L,y1-L],[x1,y1]
            ])

            self.params["square"][rectangle] = {
                "square_a" : square_a,
                "square_b" : square_b,
                "square_c" : square_c,
                "square_d" : square_d
            }
            
                
    def visualize2(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import os
        import glob
        

        #ファイルの名称を取得
        file_name = self.json_path.split("/")[-1].split(".")[0]


        self.input()
        self.json2png()
        self.json2polygon()
        self.crop()
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.imshow(self.img)
        ax.set_title(f"{file_name} all")
        ax.axis("off")

        fig_rectangles = plt.figure(figsize=(10,3))
        fig_rectangles.suptitle(f"{file_name} rectangles")
        for i, r in enumerate(self.params["rectangles"].values()):
            width = abs( r[0,0]-r[1,0] )
            height = abs(r[0,1]-r[1,1])
            rec = patches.Rectangle(r[0], width=width, height=height, fill=True, ec="blue", alpha=0.5)
            ax.add_patch(rec)
            crop = self.img[r[0,1]:r[0,1]+height, r[0,0]:r[0,0]+width, :]
            rec_ax = fig_rectangles.add_subplot(1,3,i+1)
            rec_ax.imshow(crop)
            rec_ax.set_title(f"rectangle_{i}")
            rec_ax.axis("off")
            

        keys1 = ["rectangle1", "rectangle2", "rectangle3", ]
        keys2 = ["square_a", "square_b", "square_c", "square_d" ]
        L = self.params["length"]
        fig_square = plt.figure(figsize=(10,10))
        fig_square.suptitle(f"{file_name} squares")
        i=1
        for key1 in keys1:
            for key2 in keys2:
                s = self.params["square"][key1][key2]
                sq = patches.Rectangle(s[0], L, L, fill=False, ec="red")
                ax.add_patch(sq)
                sq_ax = fig_square.add_subplot(3,4,i)
                crop = self.img[s[0,1]:s[0,1]+L, s[0,0]:s[0,0]+L, :]
                sq_ax.imshow(crop)
                sq_ax.set_title(f"{key1} {key2}")
                sq_ax.axis("off")
                i += 1
        if not self.save_fig==False:

            if glob.glob(f"{self.save_fig}/{file_name}") == []:
                os.mkdir(f"{self.save_fig}/{file_name}")
            fig.savefig(f"{self.save_fig}/{file_name}/all.png")
            fig_rectangles.savefig(f"{self.save_fig}/{file_name}/rectangle.png")
            fig_square.savefig(f"{self.save_fig}/{file_name}/square.png")

                



            


        
