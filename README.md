# Modern UI Design

Hey guys, I'm excited to introduce a new module (Module 7) which is centered around upgrading your dashboard to a modern User Interface (UI)!

This video is a demo of the new UI, that I built recently!!

[![Everything Is AWESOME](https://img.youtube.com/vi/StTqXEQ2l-Y/0.jpg)](https://www.youtube.com/watch?v=StTqXEQ2l-Y "Everything Is AWESOME")

I will be adding lectures soon on the whole design process of modernizing the User Interface and why aesthetics in computer vision is important. 

## Features

Here are some of the features that I have upgraded from the original Dashboard:

* White card style theme
* Color Scheme - with CSS Stylesheets
* Side menu-bar navigation menu
* Icons using Dash-Iconify

## Future Work

- [ ]  Swap YOLOX with YOLOv7
- [ ]  Clean up Code
- [ ]  Future work

## Implement this Design

There are 2 ways in which you can run this design: 

### Method 1

You can git clone the latest work from Github. 
git clone https://github.com/augmentedstartups/ModernComputerVisionUI.git

Download YOLOX weights and deepsort folders from your original YOLOX Folder into the cloned folder called ModernComputerVisionUI.

Download the Models

|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8| [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.7 | 48.0 | 11.1 |63.7 | 185.3 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |


### Method 2

Download the full project after enrolling in the full YOLOX Dashboard Course - Here https://www.augmentedstartups.com/yolox-pro-computer-vision-dashboard

I would run it using PyCharm Community because we include the virtual environment (venv) that you can use to run the project.




