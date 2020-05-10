# Amiya-Is-Not-Donkey

阿米娅不是驴

看到一篇知乎回答受到启发，自己也试了试。

首先用 [GoogleImagesDownloader](https://github.com/WuLC/GoogleImagesDownloader) 工具下载兔子（rabbit）和驴子（donkey）的图片。

寻找一个合适又简单的模型，这里用 简单 CNN 和 resnet，来训练一个分类模型。

最后搭建模型和训练模型，然后让 AI 来判断！阿米娅是不是驴？

# 准备

使用 selenium 库
from selenium import webdriver
如果用 Chrome，就要下载 ChromeDriver（FireFox 也要下对应的 webdriver）：
[https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)
我的是 ChromeDriver 81.0.4044.138 版本。
下好后把 放到 PATH 目录列表中，我把它放在 C:\Program Files (x86)\chromedriver 里。
