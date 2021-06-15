import cv2
import os
import json
import copy
import numpy as np
import shutil

# 定义灰色，用于信息显示的背景和未定义物体框的显示
COLOR_GRAY = (192, 192, 192)
BAR_HEIGHT = 16
FPS = 24

get_bbox_name = '{}.txt'.format

SUPPOTED_FORMATS = ['jpg', 'jpeg', 'png']

#缩放系数
Zoom_Factor = 0.6

#固定开始帧
DET_ZEN = 0

def walk_path(root, lines, keyword):
    for p0 in os.listdir(root):
        P0 = os.path.join(root, p0)
        if os.path.isdir(P0):
            walk_path(P0, lines, keyword)
        else:
            if keyword != '':
                if p0.endswith(keyword):
                    lines.append(P0)
            else:
                lines.append(P0)
    return lines


class ShowRemark:
    def __init__(self, base_dir, label_thing):
        self.BASE_DIR = base_dir
        self.window_name = "show_box"
        self._cur_label = label_thing
        # 当前图像对应的所有已标注框
        self._bboxes = []
        self._point = []
        self._BoxInfo = []
        # self._BoxInfobox = {"box":[],"points":[],"label": {"标注对象": self._cur_label,self._cur_label: len(self._BoxInfo)+1}}
        self._BoxInfobox = {"box": [], "points": [],
                            "label": {"label": self._cur_label, "label_id": len(self._BoxInfo) + 1, "object":"人体"}}
        # 鼠标点位
        self._pt0 = None
        self._pt1 = None
        # 表明当前是否正在画框的状态标记
        self._drawing = False
        # 当前标注物体的名称
        self.imagefiles = [x for x in os.listdir(self.BASE_DIR) if x[x.rfind('.') + 1:].lower() in SUPPOTED_FORMATS]
        self.labeled = [x for x in self.imagefiles if os.path.exists(os.path.join(self.BASE_DIR,x.split('.')[0] + '.txt' ))]
        self.to_be_labeled = [x for x in self.imagefiles if x not in self.labeled]
        # # 每次打开一个文件夹，都自动从还未标注的第一张开始
        self._filelist = self.labeled + self.to_be_labeled
        self._index = 0
        if self._index > len(self._filelist) - 1:
            self._index = len(self._filelist) - 1
        self._detect_mode = False
        self.ST = True

    # 鼠标回调函数
    def _mouse_ops(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._pt0 = (x, y)

        # 左键抬起，表明当前框画完了，坐标记为右下角，并保存，同时改变drawing标记为False
        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._pt1 = (x, y)
            # self._bboxes.append((self._cur_label, (self._pt0, self._pt1)))
            self._BoxInfobox["box"] = [self._pt0[0], self._pt0[1], self._pt1[0]-self._pt0[0], self._pt1[1]-self._pt0[1]]
            self._BoxInfobox["points"] = self.box_to_point([self._pt0[0], self._pt0[1], self._pt1[0]-self._pt0[0], self._pt1[1]-self._pt0[1]])
            self._BoxInfobox["label"]["label_id"] = len(self._BoxInfo) + 1
            self._BoxInfo.append(self._BoxInfobox)
            print(self._BoxInfo)
            self.clear_state()


        # 实时更新右下角坐标方便画框
        elif event == cv2.EVENT_MOUSEMOVE:
            self._pt1 = (x, y)

        # 鼠标右键删除最近画好的框
        elif event == cv2.EVENT_RBUTTONUP:
            if self._BoxInfo:
                self._BoxInfo.pop(-1)


    def clear_state(self):
        self._bboxes = []
        self._point = []
        self._BoxInfobox = {"box": [], "points": [],
                            "label": {"label": self._cur_label, "label_id": len(self._BoxInfo) + 1, "object":"人体"}}

    def show_Boxinfo(self, canvas):
        """显示Boxinfo中所有的point"""
        if self._BoxInfo == []:
            return canvas
        for idx, box in enumerate(self._BoxInfo):
            showbox = [[(i[0], i[1]), (y[0], y[1])] for i in box["points"] for y in box["points"]]
            for pointbox in showbox:
                cv2.line(canvas, pointbox[0], pointbox[1], (255,111,111), thickness=1, lineType=1)
                cv2.putText(canvas, "id:" + str(box["label"]["label_id"]), (box["points"][0][0]+3,box["points"][0][1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 0), 2)
        return canvas

    # 检查模式
    def detect_show(self, canvas):
        # 检测出上一个label标签
        img_file = [(IDX, x) for IDX, x in enumerate(self._filelist[0:self._index])]
        IMG_label = [x[0] for x in img_file if os.path.exists(os.path.join(self.BASE_DIR, x[1].split('.')[0] + '.txt'))]
        last_zen = IMG_label[-1]
        all_txt_file_list = [os.path.join(self.BASE_DIR, x.split('.')[0] + '.txt') for x in self._filelist]
        with open(all_txt_file_list[last_zen], 'r', encoding='utf-8') as f:
            base_data = json.loads(f.read())
            base_info = list(base_data.values())[0]
        for idx, box in enumerate(base_info["BoxInfo"]):
            showbox = [[(i[0], i[1]), (y[0], y[1])] for i in box["points"] for y in box["points"]]
            for pointbox in showbox:
                cv2.line(canvas, pointbox[0], pointbox[1], (125, 155, 125), thickness=1, lineType=1)
                cv2.putText(canvas, str(idx), (int((box["points"][0][0] + box["points"][1][0])/2),box["points"][0][1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (125, 155, 125), 2)
                cv2.putText(canvas, str(box["label"]["label"]), (box["points"][0][0] + 18, box["points"][0][1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 155, 125), 1)
        return canvas


    # 标签显示
    def _draw_bbox(self, canvas):
        # 正在标注的物体信息，如果鼠标左键已经按下，则显示两个点坐标，否则显示当前待标注物体的名称
        label_msg = '{}: {}'.format(self._cur_label, self._pt0) \
             if self._drawing \
            else 'Current label: {}'.format(self._cur_label)
        # 显示当前文件名，文件个数信息
        msg = '{}/{}: {} | {} A:next D:last J:replace L:remove c:change_label p:get_medi_label H:show_before B:close_show_before'.format(self._index + 1, len(self._filelist), self._filelist[self._index], label_msg)

        cv2.putText(canvas, msg, (1, 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)
        #1、bbox中点的实时显示
        for box_point in self._bboxes:
            cv2.putText(canvas, '.', (box_point[0], box_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0,255,0), 2)
        # 2、point圈的可视化，并将框输入到self._BoxInfo = []
        canvas = ShowRemark.show_Boxinfo(self, canvas)
        if self._detect_mode:
            try:
                canvas = self.detect_show(canvas)
            except Exception as e:
                print(e)
                print(e.__traceback__.tb_lineno)
        return canvas

    # 微调模式，识别出相近框，并自动合并相近框
    def auto_remark(self):
        if self.ST == False:
            print("退出微调")
            return 0
        if len(self._BoxInfo)>=2:
            new_box = self._BoxInfo[-1]["box"]
            for idx, bigbox in enumerate(self._BoxInfo[:-1]):
                box = bigbox["box"]
                if ((box[0]<new_box[0]<box[0]+box[2]/2) and (box[1]<new_box[1]<box[1]+box[3]/2)) or ((new_box[0]<box[0]<new_box[0]+new_box[2]/2) and (new_box[1]<box[1]<new_box[1]+new_box[3]/2)):
                    self._BoxInfo[int(idx)]["box"] = self._BoxInfo[-1]["box"]
                    self._BoxInfo[int(idx)]["points"] = self.box_to_point(self._BoxInfo[-1]["box"])
                    self._BoxInfo.pop(-1)
                    print("正在微调")

    @staticmethod
    def load_bbox(filepath):
        bboxes = []
        with open(filepath, 'r') as f:
            line = f.readline().rstrip()
            while line:
                bboxes.append(eval(line))
                line = f.readline().rstrip()
        return bboxes

    # 读取图像文件和对应标注框信息（如果有的话）
    @staticmethod
    def load_sample(filepath):
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        bbox_filepath = get_bbox_name(filepath)
        bboxes = []
        if os.path.exists(bbox_filepath):
            bboxes = ShowRemark.load_bbox(bbox_filepath)
        return img, bboxes

    def save_txt(self):
        if self._BoxInfo == []:
            return 0
        newdict = dict()
        newdict[self.imagefiles[self._index]] = {"BoxNum":len(self._BoxInfo),"BoxInfo":self._BoxInfo}
        with open(os.path.join(self.BASE_DIR,self.imagefiles[self._index]).split(".")[0] + '.txt','w',encoding="utf-8") as f:
            f.write(json.dumps(newdict, indent=4, ensure_ascii=False))
        self._BoxInfo = []

    def read_txt(self):
        print(os.path.join(self.BASE_DIR,self.imagefiles[self._index]).split(".")[0] + '.txt')
        if not os.path.exists(os.path.join(self.BASE_DIR,self.imagefiles[self._index]).split(".")[0] + '.txt'):
            self._BoxInfo = []
            return 0
        with open(os.path.join(self.BASE_DIR,self.imagefiles[self._index]).split(".")[0] + '.txt','r',encoding="utf-8") as f:
            data = json.loads(f.read())
        self._BoxInfo = data[self.imagefiles[self._index]]["BoxInfo"]
        return 0

    def add_kong_label(self):
        print(self.to_be_labeled)
        for img in self.to_be_labeled:
            imgname, _ = os.path.splitext(img)
            imgpath = os.path.join(self.BASE_DIR, imgname + '.txt')
            print(imgpath)
            newdict = dict()
            newdict[img] = {"BoxNum": 0, "BoxInfo": []}
            with open(imgpath, 'w', encoding='utf-8') as f:
                f.write(json.dumps(newdict, indent=4, ensure_ascii=False))



    def remove_BoxInfo_form_id(self, _id):
        for idx, box in enumerate(self._BoxInfo):
            if idx == int(_id):
                self._BoxInfo.remove(box)

    @staticmethod
    def box_to_point(box):  # [408,870,151,89] -> point #[1039,296],[1371,250],[1108,766],[1400,700]
        x, y, w, h = box[0], box[1], box[2], box[3]
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    def remove_now_label(self):
        if os.path.exists(os.path.join(self.BASE_DIR, self.imagefiles[self._index].split('.')[0] + '.txt')):
            os.remove(os.path.join(self.BASE_DIR, self.imagefiles[self._index].split('.')[0] + '.txt'))

    def have_label_infomation_to_accurate_chazen(self):
        begin_zen = input("请输入开始帧：")
        DET_ZEN = int(begin_zen) - 1
        end_zen = self._index
        all_txt_file_list = [os.path.join(self.BASE_DIR, x.split('.')[0] + '.txt') for x in self._filelist]
        with open(all_txt_file_list[DET_ZEN], 'r', encoding='utf-8') as f:
            base_data = json.loads(f.read())
        with open(all_txt_file_list[end_zen], 'r', encoding="utf-8") as fb:
            end_data = json.loads(fb.read())
        start_point = list(base_data.values())[0]["BoxInfo"]
        # end_point = self._BoxInfo
        end_point = list(end_data.values())[0]["BoxInfo"]
        SAVE_DICT = dict() #存储生成的点
        for idx, points in enumerate(start_point):# 循环第几个boxinfo元素（相当于第几个框）
            for idy, s_point in enumerate(points["points"]):# 循环points中第几个点（相当于框的第几个点）
                e_point = end_point[idx]["points"][idy]
                addx = (e_point[0] - s_point[0])/(end_zen - DET_ZEN)
                addy = (e_point[1] - s_point[1]) / (end_zen - DET_ZEN)
                x_y_list = [[int(round(e_point[0] - addx*num, 0)),int(round(e_point[1] - addy*num, 0))] for num in range(1,(end_zen - DET_ZEN))]
                key = str(idx) + '_' + str(idy)
                SAVE_DICT[key] = x_y_list
        # 开始将生成的点生成中间帧,倒叙生成点
        #模板字典
        normaldict = copy.deepcopy(end_point)
        for idz, num in enumerate(range((end_zen - DET_ZEN -1),0,-1)):
            now_zen = DET_ZEN + num
            now_label_path = all_txt_file_list[now_zen]
            save_info = normaldict
            real_dict = dict()
            #生成中间帧label标签格式
            for idx, kuang in enumerate(save_info):
                for idy, point in enumerate(kuang["points"]):
                    key = str(idx) + '_' + str(idy)
                    save_info[idx]["points"][idy] = copy.deepcopy(SAVE_DICT[key][idz])
                save_info[idx]["box"] = ShowRemark.point_to_box(save_info[idx]["points"])
            real_dict[self._filelist[now_zen]] = {"BoxNum": len(save_info), "BoxInfo": save_info}
            with open(now_label_path, 'w', encoding="utf-8") as f:
                f.write(json.dumps(real_dict, indent=4, ensure_ascii=False))

    def accurate_chazen(self):
        img_file = [(IDX, x) for IDX, x in enumerate(self._filelist[0:self._index])]
        IMG_label = [x[0] for x in img_file if os.path.exists(os.path.join(self.BASE_DIR, x[1].split('.')[0] + '.txt'))]
        begin_zen = IMG_label[-1]
        DET_ZEN = begin_zen
        end_zen = self._index
        if len(self._BoxInfo) == 0:
            pass
        all_txt_file_list = [os.path.join(self.BASE_DIR, x.split('.')[0] + '.txt') for x in self._filelist]
        with open(all_txt_file_list[DET_ZEN], 'r', encoding='utf-8') as f:
            base_data = json.loads(f.read())
        with open(all_txt_file_list[end_zen], 'r', encoding="utf-8") as fb:
            end_data = json.loads(fb.read())
        start_point = list(base_data.values())[0]["BoxInfo"]
        end_point = list(end_data.values())[0]["BoxInfo"]
        # end_point = self._BoxInfo
        SAVE_DICT = dict() #存储生成的点
        for idx, points in enumerate(start_point):# 循环第几个boxinfo元素（相当于第几个框）
            for idy, s_point in enumerate(points["points"]):# 循环points中第几个点（相当于框的第几个点）
                e_point = end_point[idx]["points"][idy]
                addx = (e_point[0] - s_point[0])/(end_zen - DET_ZEN)
                addy = (e_point[1] - s_point[1]) / (end_zen - DET_ZEN)
                x_y_list = [[int(round(e_point[0] - addx*num, 0)),int(round(e_point[1] - addy*num, 0))] for num in range(1,(end_zen - DET_ZEN))]
                key = str(idx) + '_' + str(idy)
                SAVE_DICT[key] = x_y_list
        # 开始将生成的点生成中间帧,倒叙生成点
        #模板字典
        normaldict = copy.deepcopy(end_point)
        for idz, num in enumerate(range((end_zen - DET_ZEN -1),0,-1)):
            now_zen = DET_ZEN + num
            now_label_path = all_txt_file_list[now_zen]
            save_info = normaldict
            real_dict = dict()
            #生成中间帧label标签格式
            for idx, kuang in enumerate(save_info):
                for idy, point in enumerate(kuang["points"]):
                    key = str(idx) + '_' + str(idy)
                    save_info[idx]["points"][idy] = copy.deepcopy(SAVE_DICT[key][idz])
                save_info[idx]["box"] = ShowRemark.point_to_box(save_info[idx]["points"])
                # save_info[idx]["label"]["label_id"] =
            real_dict[self._filelist[now_zen]] = {"BoxNum": len(save_info), "BoxInfo": save_info}
            with open(now_label_path, 'w', encoding="utf-8") as f:
                f.write(json.dumps(real_dict, indent=4, ensure_ascii=False))

    @staticmethod
    def point_to_box(point):  # [408,870,151,89] -> point #[1039,296],[1371,250],[1108,766],[1400,700]
        x, y = point[0][0], point[0][1]
        w, h = point[1][0] - point[0][0], point[2][1] - point[0][1]
        return [x, y, w, h]


    def change_id(self):
        try:
            nowid = int(input("请输入要更换图片的id:"))
            need_id = int(input("请输入要更改为的id:"))
            for bigbox in self._BoxInfo:
                if bigbox["label"]["label_id"] == nowid:
                    bigbox["label"]["label_id"] = need_id
        except Exception as e:
            print(e,"您输入的格式不规范，请重新输入")

    def change_one_label(self):
        try:
            nowid = int(input("请输入要更换图片的id:"))
            need_label = str(input("请输入要更改为的标签:"))
            line1 = walk_path(self.BASE_DIR, [], 'txt')
            # for bigbox in self._BoxInfo:
            #     if bigbox["label"]["label_id"] == nowid:
            #         bigbox["label"]["label"] = need_label
            line1 = walk_path(self.BASE_DIR, [], 'txt')
            for fileObj in line1:
                imgname = fileObj.split('\\')[-1].replace('.txt', '.jpg')
                with open(fileObj, 'r+', encoding='ANSI') as file:
                    strfile = file.read()
                    data = json.loads(strfile)
                    sss = data[imgname]['BoxInfo']
                    for idx, i in enumerate(sss):
                        if i["label"]["label_id"] == nowid:
                            i["label"]["label"] = need_label
                    after = data
                with open(fileObj, 'w', encoding='ANSI') as f:
                    json.dump(after, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(e)
            print("您输入的格式不规范，请重新输入")

    def print_have_id(self):
        exist_id = set()
        line1 = walk_path(self.BASE_DIR, [], 'txt')
        for fileObj in line1:
            imgname = fileObj.split('\\')[-1].replace('.txt', '.jpg')
            try:
                with open(fileObj, 'r+', encoding='ANSI') as file:
                    strfile = file.read()
                    data = json.loads(strfile)
                    sss = data[imgname]['BoxInfo']
                    for idx, i in enumerate(sss):
                        exist_id.add("id:"+str(i['label']['label_id']) + ':' + i['label']['label'])
            except Exception as e:
                print(e)
            finally:
                print(exist_id)
                return exist_id


    def run(self):
        imgfile = [x for x in os.listdir(self.BASE_DIR) if x[x.rfind('.') + 1:].lower() in SUPPOTED_FORMATS]
        delay = int(1000 / FPS)

        while True:
            try:
                img = ShowRemark.load_sample(os.path.join(self.BASE_DIR,imgfile[self._index]))[0]
                cv2.namedWindow(self.window_name)
                cv2.setMouseCallback(self.window_name, self._mouse_ops)
                canvas = self._draw_bbox(img)
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(delay)
                self.auto_remark()
                # print(self.ST)

                # 插入标签
                if key == ord("p") or key == ord("P"):
                    if self.imagefiles[self._index] not in [x for x in self.imagefiles if os.path.exists(
                            os.path.join(self.BASE_DIR, x.split('.')[0] + '.txt'))]:
                        self.save_txt()
                        self.accurate_chazen()
                        continue
                    if self.imagefiles[self._index] in [x for x in self.imagefiles if os.path.exists(os.path.join(self.BASE_DIR,x.split('.')[0] + '.txt' ))]:
                        self.save_txt()
                        self.have_label_infomation_to_accurate_chazen()

                # 退出微调模式
                if key == ord("k") or key == ord("K"):
                    if self.ST == True:
                        self.ST = False
                    else:
                        self.ST = True

                # 查看前一张图片
                if key == ord("A") or key == ord("a"):
                    if self._index -1 < 0:
                        continue
                    self.clear_state()
                    self._index -= 1
                    self.read_txt()

                #查看后一张图片
                if key == ord("d") or key == ord("D"):
                    self.save_txt()
                    self.clear_state()
                    self._index += 1
                    if self._index > len(self._filelist) - 1:
                        self._index = len(self._filelist) - 1
                    self.read_txt()

                # 检查模式
                if key == ord("h") or key == ord("H"):
                    self._detect_mode = True

                # 关闭检查模式
                if key == ord("b") or key == ord("B"):
                    self._detect_mode = False

                # 更换id
                if key == ord("i") or key == ord("I"):
                    self.change_id()

                # 更换标签状态
                if key == ord("c") or key == ord("C"):
                    newlabel = input("请输入新标签：")
                    self._cur_label = newlabel

                # 替换修改指定id的框
                # import time
                if key == ord("j") or key == ord("J"):
                    imgbox_id = input("请输入需要被替换的图片顺序:")
                    replace_id = input("请输需要取代替换图片顺序:")
                    self._BoxInfo[int(imgbox_id)]["points"] = self._BoxInfo[int(replace_id)]["points"]
                    self._BoxInfo[int(imgbox_id)]["box"] = self._BoxInfo[int(replace_id)]["box"]
                    self.remove_BoxInfo_form_id(replace_id)

                # 删除指定框
                if key == ord("l") or key == ord("L"):
                    imgbox_id = input("请输入需要删除的图片顺序:")
                    self.remove_BoxInfo_form_id(imgbox_id)

                # # 补充标签
                # if key == ord("e") or key == ord("E"):
                #     self.add_kong_label()

                #退出
                if key == 27:
                    print("Ddd")
                    cv2.destroyAllWindows()
                    now_label = self.print_have_id()
                    s = input("当前的标签结果为：{},是否需要修改(y/n):".format(now_label))
                    if s.lower() == 'y':
                        while True:
                            self.change_one_label()
                            Quit = input("退出请按q:")
                            if Quit.lower() == "q":
                                break
                    newpath = input("请输入新地址：")
                    label = input("请输入需要新增的标签：")
                    action = ShowRemark(newpath, label)  # 地址  ， 标签
                    action.run()
            except Exception as e:
                print(e)






if __name__ == '__main__':
    path = input("请输入路径：")
    label = input("请输入需要新增的标签：")
    action = ShowRemark(path,label) # 地址  ， 标签
    action.run()