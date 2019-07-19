[TOC]
# wechat analsys
## 环境搭建
- pyechart 保存png图片
```bash 
1.https://phantomjs.org/download.html
2.安装phantomjs
3.https://nodejs.org/en/download/
4.安装notejs
5.pip install pyecharts-snapshot
```
- python = 3.5
- 依赖环境
```bash
python-opencv
itchat
base64
numpy 
pandas 
jieba
snownlp
baidu-aip
matplotlib
pyecharts 
wordcloud 
```
## 实现的功能
### 微信好友信息导出
### 微信好友性别分析
### 微信好友城市分布
### 微信好友地图分布
### 微信好友个性签名情感分析
### 微信好友头像照片墙
### 是否识好友识别
### 自动输出微信好友分析报告 

## 源代码
```bash
# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  wechat ansys
@Evn     :  1.https://phantomjs.org/download.html
            2.安装phantomjs
            3.https://nodejs.org/en/download/
            4.安装notejs
            5.pip install pyecharts-snapshot
@Date    :  2019-07-17  23:15
'''


import re
import os
import cv2
import itchat
import base64
import numpy as np
import pandas as pd
import jieba.analyse
import snownlp
from aip import AipFace  # pip install baidu-aip
from PIL import Image
import matplotlib.pyplot as plt
from pyecharts import Bar
from collections import Counter
from pyecharts import Pie
from pyecharts import Map
from docx import Document
from docx.shared import Inches
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator

document = Document()
plt.rcParams['font.sans-serif']=['SimHei']#绘图时可以显示中文
plt.rcParams['axes.unicode_minus']=False#绘图时可以显示中文


# login wechat
itchat.auto_login(hotReload = True)
friends = itchat.get_friends(update = True)

document.add_heading('微信分析报告', 0)

# save wechat infomation
StarFriend = list(map(lambda x:x['StarFriend'],friends[1:]))
RemarkName = list(map(lambda x:x['RemarkName'],friends[1:]))
sex=list(map(lambda x:x['Sex'],friends[1:]))
nickname=list(map(lambda x:x['NickName'],friends[1:]))
signature=list(map(lambda x:x['Signature'],friends[1:]))
province=list(map(lambda x:x['Province'],friends[1:]))
city=list(map(lambda x:x['City'],friends[1:]))
headimg=list(map(lambda x:x['HeadImgUrl'],friends[1:]))
info={"RemarkName":RemarkName,"nickname":nickname,"sex":sex,"province":province,"city":city,"signature":signature,'headimgurl':headimg,"StarFriend":StarFriend}
data=pd.DataFrame(info)
data1 = data.drop("headimgurl",axis=1)
data1.to_csv("./wechatinfo.csv")

# p = document.add_paragraph('一、微信详细的通讯录')
# for index in range(1,472):
#     friend = friends[index]
#     print(friend)
#     p.add_run(str(friend))

# where is your friends?
city_5=Counter(city).most_common(18) # 返回出现次数最多的20条
bar = Bar('好友所在城市TOP18', '', title_pos='center', width=800, height=500)
attr, value = bar.cast(city_5)
bar.add('', attr, value, is_visualmap=True, visual_text_color='#fff', is_more_utils=True,is_label_show=True)
bar.show_config()
bar.render('好友所在城市TOP18.png')

p = document.add_paragraph('一、微信好友分析\n')
document.add_paragraph('了解你的微信好友分布在那些城市？每个城市有多少人？\n', style='List Number')
document.add_picture('好友所在城市TOP18.png', width=Inches(5))
p.add_run('好友所在城市TOP18\n')

# how many your friends sex?
sexs=list(map(lambda x:x['Sex'],friends[1:])) #提取好友性别
value = [sexs.count(1), sexs.count(2), sexs.count(0)]#对性别进行计数
sex_attr=['male','female','unknown']
pie = Pie('好友性别比例', '好友总人数：%d' % len(sex), title_pos='center')
pie.add('', sex_attr, value, radius=[30, 75], rosetype='area', is_label_show=True,is_legend_show=True, legend_top='bottom')
pie.show_config()
pie.render('好友性别比例.png')

document.add_paragraph('了解你微信好友，目前有多少的男生和女生，查看好友性别比例\n', style='List Number')
document.add_picture('好友性别比例.png', width=Inches(5))
p.add_run('好友性别比例\n')

# show where is your friends
provinces_count = data.groupby('province', as_index=True)['province'].count().sort_values()
attr = list(map(lambda x: x if x != '' else '未知', list(provinces_count.index)))#未填写地址的改为未知
value = list(provinces_count)
map_1 = Map("微信好友位置分布图",title_pos="center",width=1000, height=500)
map_1.add('', attr, value, is_label_show=True, is_visualmap=True, visual_text_color='#000',visual_range=[0,120])
map_1.render(path='./微信好友分布.png')

document.add_paragraph('微信好友分布\n', style='List Number')
document.add_picture('微信好友分布.png', width=Inches(5))
p.add_run('微信好友分布\n')


# make Signature's world clound
signatures = ''
emotions = []

for friend in friends:
    signature = friend['Signature']
    if signature != None:
        signature = signature.strip().replace("span", "").replace("class", "").replace("emoji", "")  # 去除无关数据
        # print(signature)
        signature = re.sub(r'1f(\d.+)', "", signature)
        signature = re.sub('\s', "", signature)
        signature = re.sub('\W', "", signature)
        signature = re.sub('[a-zA-Z0-9]', "", signature)

    if len(signature) > 0:
        # print(signature)
        nlp = snownlp.SnowNLP(signature)
        emotions.append(nlp.sentiments)  # nlp.sentiments：权值
        signatures += " ".join(jieba.analyse.extract_tags(signature, 5))  # 关键字提取

back_coloring = np.array(Image.open("1.png"))  # 图片可替换
word_cloud2 = WordCloud(font_path='PingFang.ttc', background_color='white', max_words=2000, mask=back_coloring,margin=15)
word_cloud2.generate(signatures)
image_colors = ImageColorGenerator(back_coloring)
plt.figure(figsize=(6, 5), dpi=160)
plt.imshow(word_cloud2.recolor(color_func=image_colors))
plt.axis("off")
plt.show(0)
word_cloud2.to_file("signatures.png")

p = document.add_paragraph('二、微信状态情感分析\n')
document.add_paragraph('微信好友情感分析\n', style='List Number')
document.add_picture('signatures.png', width=Inches(5))
p.add_run('微信好友情感分析\n')


# Emotional analysis of friends' personality signature
count_positive = len(list(filter(lambda x: x > 0.66, emotions)))  # 大于0.66为积极
count_negative = len(list(filter(lambda x: x < 0.33, emotions)))  # 小于0.33为消极
count_neutral = len(list(filter(lambda x: x >= 0.33 and x <= 0.66, emotions)))
value = [count_positive, count_negative, count_neutral]
att_attr = ['积极', '消极', '中性']
bar = Bar('个性签名情感分析', title_pos='center', width=800, height=500)
bar.add('', att_attr, value, visual_range=[0, 200], is_visualmap=True, is_label_show=True)
bar.show_config()
bar.render(path='微信好友个性签名情感分析.png')

document.add_paragraph('微信好友个性签名情感分析\n', style='List Number')
document.add_picture('微信好友个性签名情感分析.png', width=Inches(5))
p.add_run('微信好友个性签名情感分析\n')


# Download your friends headimages
# baseFolder = './HeadImages'
# if(os.path.exists(baseFolder) == False):
#     os.makedirs(baseFolder)
# # Analyse Images
# image_tags = ''
# for index in range(1,472):
#     friend = friends[index]
#     print(friend)
# # Save HeadImages
#     imgFile = baseFolder + '/Image%s.jpg' % str(index)
#     imgData = itchat.get_head_img(userName = friend['UserName'])
#     # print(imgData)
#     # if(os.path.exists(imgFile) == False):
#     #     with open(imgFile,'wb') as file:
#     #          file.write(imgData)

# make Photo wall
raw_name = "heart.jpeg"
res_file = "./HeadImages" # 资源照片路径
mw = 100 # 单个照片的尺寸
## 照片列表
def get_picture_list(picture_list):
    for filename in os.listdir(res_file):
        if filename != ".DS_Store":
            filepath = os.path.join(res_file, filename)
            picture_list.append(filepath)

## 照片器样式
def load_raw(raw_name, data_list, _size):
    im = Image.open(raw_name)
    w, h = im.size
    for i in range(w):
        for j in range(h):
            v = im.getpixel((i, j))
            if v != 0:
                # 将灰度图的像素映射到照片墙的坐标内
                x = int(i * _size[0] / w)
                y = int(j * _size[1] / h)
                data_list[x][y] = 1
                print(data_list)

# 绘制一张照片到指定位置
def draw_picture(save_image, x, y, im_name):
        in_image = Image.open(im_name)
        in_image = in_image.resize((mw, mw),Image.ANTIALIAS)
        save_image.paste(in_image, ((x-1)*mw, (y-1)*mw))

# 照片墙能容纳的最大照片数量 20 * 20
w, h = (40, 40)
data_list = [[0 for col in range(h)] for row in range(w)]

# 加载灰度图, 照片墙样式
load_raw(raw_name, data_list, (w, h))
# 创建一张新的照片
save_image = Image.new('RGBA', (mw * w, mw * h))

# 获取所有照片路径名称
picture_list = []
get_picture_list(picture_list)

pos = 0
print(picture_list)
# 按照样式, 缩放绘制照片到指定位置
for i in range(w):
    for j in range(h):
        if data_list[i][j] > 0:
            draw_picture(save_image, i, j, picture_list[pos])
            print(i, j)
            pos += 1
            pos = pos % len(picture_list)

# 保存
save_image.show()
save_image.save("./Photo_Wall.png")

p = document.add_paragraph('三、微信好友照片墙\n')
document.add_paragraph('微信好友头像\n', style='List Number')
document.add_picture('Photo_Wall.png', width=Inches(5))
p.add_run('微信好友头像照片墙\n')

# show friend head picture
img=Image.open("./faces.jpg")
plt.figure("好友头像")
plt.imshow(img)
plt.show(0)

# baidu face Recognition ：is your friends？
APP_ID  = "16832720"
API_KEY = "oK0GlKCnruNx2Bp5HEQokC5C"  #请替换为自己的apikey，注册地址https://login.bce.baidu.com/?account=
SECRET_KEY = "Xpr6vSiNi9sgwskoLk4Sfdac9hAmkalx"
client = AipFace(APP_ID, API_KEY, SECRET_KEY) #初始化aipface对象

filepath = "./faces.jpg"
with open(filepath, "rb") as fp:
    base64_data = base64.b64encode(fp.read())
image = str(base64_data, 'utf-8')
imageType = "BASE64"
options = {}
options["face_field"] = "age,gender,beauty,expression"
options["max_face_num"] = 10
options["face_type"] = "LIVE"
result = client.detect(image, imageType, options)
print("result",result)



facelist=result['result']['face_list']
output=pd.DataFrame(facelist)
output.drop(['angle','face_token'],axis=1,inplace=True)
output.head(2)


img=cv2.imread('faces.jpg')
location=result['result']['face_list'][0]['location']
left_top=(location['left'],location['top'])
right_bottom=(left_top[0]+location['width'],left_top[1]+location['height'])
cv2.rectangle(img,(282,219),(328,264),(255,0,0),2)#周迅
cv2.rectangle(img,(206,40),(246,80),(255,0,0),2) #杜鹃
cv2.rectangle(img,(344,40),(384,85),(255,0,0),2)#李宇春
cv2.rectangle(img,(154,190),(194,230),(255,0,0),2)#章子怡
cv2.rectangle(img,(87,119),(125,159),(255,0,0),2)#李冰冰
cv2.imshow('img',img)
plt.imshow(img)
plt.show(0)

document.add_page_break()

document.save('微信好友分析报告.docx')
```