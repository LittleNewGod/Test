# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         ChangeSketch
# Author:       xin
# Date:         2019/10/21
# IDE:  PyCharm
# -------------------------------------------------------------------------------

#第一版：

class ChangeSketch:
    def __init__(self,img_before,img_after):
        self.img_before=img_before
        self.img_after=img_after

    def change(self,a, b, alpha):
        return min(int(a * 255 / (256 - b * alpha)), 255)

    def draw(self):
        from PIL import Image, ImageFilter, ImageOps

        img = Image.open(self.img_before)
        blur = 25
        alpha = 1.0
        img1 = img.convert('L')  # 图片转换为灰色
        img2 = img1.copy()
        img2 = ImageOps.invert(img2)
        for i in range(blur):  # 模糊度
            img2 = img2.filter(ImageFilter.BLUR)
        width, height = img1.size
        for x in range(width):
            for y in range(height):
                a = img1.getpixel((x, y))
                b = img2.getpixel((x, y))
                img1.putpixel((x, y), self.change(a, b, alpha))
        img1.show()
        img1.save(self.img_after)

test = ChangeSketch(r'D:\Code\github\Test\2.jpg', r'D:\Code\github\Test\sketch_result\21.jpg')
test.draw()


#############################分隔线######################################
#第二版
class ChangeSketch2:
    def __init__(self,img_before,img_after):
        self.img_before=img_before
        self.img_after=img_after

    def change(self):
        from PIL import Image
        import os

        image = self.img_before
        img = Image.open(image)
        new = Image.new("L", img.size, 255)
        width, height = img.size
        img = img.convert("L")
        # print(img.size)
        # print(img.mode) #RBG
        #
        # img_get = img.getpixel((0, 0))
        # print(img_get) #三原色通道
        #
        # img_L=img.convert('L')
        # print(img_L)
        # img_get_L=img_L.getpixel((0,0)) #换算 得到灰度值
        # print(img_get_L)

        # 定义画笔的大小
        Pen_size = 3
        # 色差扩散器
        Color_Diff = 6
        for i in range(Pen_size + 1, width - Pen_size - 1):
            for j in range(Pen_size + 1, height - Pen_size - 1):
                # 原始的颜色
                originalColor = 255
                lcolor = sum([img.getpixel((i - r, j)) for r in range(Pen_size)]) // Pen_size
                rcolor = sum([img.getpixel((i + r, j)) for r in range(Pen_size)]) // Pen_size

                # 通道----颜料
                if abs(lcolor - rcolor) > Color_Diff:
                    originalColor -= (255 - img.getpixel((i, j))) // 4
                    new.putpixel((i, j), originalColor)

                ucolor = sum([img.getpixel((i, j - r)) for r in range(Pen_size)]) // Pen_size
                dcolor = sum([img.getpixel((i, j + r)) for r in range(Pen_size)]) // Pen_size

                # 通道----颜料
                if abs(ucolor - dcolor) > Color_Diff:
                    originalColor -= (255 - img.getpixel((i, j))) // 4
                    new.putpixel((i, j), originalColor)

                acolor = sum([img.getpixel((i - r, j - r)) for r in range(Pen_size)]) // Pen_size
                bcolor = sum([img.getpixel((i + r, j + r)) for r in range(Pen_size)]) // Pen_size

                # 通道----颜料
                if abs(acolor - bcolor) > Color_Diff:
                    originalColor -= (255 - img.getpixel((i, j))) // 4
                    new.putpixel((i, j), originalColor)

                qcolor = sum([img.getpixel((i + r, j - r)) for r in range(Pen_size)]) // Pen_size
                wcolor = sum([img.getpixel((i - r, j + r)) for r in range(Pen_size)]) // Pen_size

                # 通道----颜料
                if abs(qcolor - wcolor) > Color_Diff:
                    originalColor -= (255 - img.getpixel((i, j))) // 4
                    new.putpixel((i, j), originalColor)
        new.save(self.img_after)

        i = os.system('mshta vbscript createobject("sapi.spvoice").speak("%s")(window.close)' % '您的图片转换好了')
        os.system(self.img_after)

test2 = ChangeSketch2(r'D:\Code\github\Test\2.jpg', r'D:\Code\github\Test\sketch_result\22.jpg')
test2.change()

#############################分隔线######################################
#第三版
class ChangeSketch3:
    def __init__(self, img_before, img_after):
        self.img_before = img_before
        self.img_after = img_after

    def change(self):
        from PIL import Image
        # image 是 PIL库中代表一个图像的类
        import numpy as np
        # 打开一张图片 “F:\PycharmProjects\cui.jpg” 是图片位置
        a = np.asarray(Image.open(self.img_before)
                       .convert('L')).astype('float')
        depth = 10.  # 浮点数，预设深度值为10
        grad = np.gradient(a)  # 取图像灰度的梯度值
        grad_x, grad_y = grad  # 分别取横纵图像的梯度值
        grad_x = grad_x * depth / 100.  # 根据深度调整 x 和 y 方向的梯度值
        grad_y = grad_y * depth / 100.
        A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)  # 构造x和y轴梯度的三维归一化单位坐标系
        uni_x = grad_x / A
        uni_y = grad_y / A
        uni_z = 1. / A

        vec_el = np.pi / 2.2  # 光源的俯视角度，弧度值
        vec_az = np.pi / 4.  # 光源的方位角度，弧度值
        dx = np.cos(vec_el) * np.cos(vec_az)  # 光源对 x 轴的影响，np.cos(vec_el)为单位光线在地平面上的投影长度
        dy = np.cos(vec_el) * np.sin(vec_az)  # 光源对 y 轴的影响
        dz = np.sin(vec_el)  # 光源对 z 轴的影响

        b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)  # 梯度与光源相互作用，将梯度转化为灰度
        b = b.clip(0, 255)  # 为避免数据越界，将生成的灰度值裁剪至0‐255区间

        im = Image.fromarray(b.astype('uint8'))  # 重构图像
        im.show()
        im.save(self.img_after)  # 保存图片的地址

test3 = ChangeSketch3(r'D:\Code\github\Test\2.jpg', r'D:\Code\github\Test\sketch_result\23.jpg')
test3.change()

#############################分隔线######################################
#第四版
class ChangeSketch4:
    def __init__(self, img_before, img_after):
        self.img_before = img_before
        self.img_after = img_after

    def change(self):
        import cv2
        import numpy as np
        import random
        import math

        img = cv2.imread(self.img_before, 1)
        imgInfo = img.shape
        height = imgInfo[0]
        width = imgInfo[1]
        # cv2.imshow('src', img)
        # sobel 1 算子模版 2 图片卷积 3 阈值判决
        # [1 2 1          [ 1 0 -1
        #  0 0 0            2 0 -2
        # -1 -2 -1 ]       1 0 -1 ]

        # [1 2 3 4] [a b c d] a*1+b*2+c*3+d*4 = dst
        # sqrt(a*a+b*b) = f>th
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = np.zeros((height, width, 1), np.uint8)
        for i in range(0, height - 2):
            for j in range(0, width - 2):
                gy = gray[i, j] * 1 + gray[i, j + 1] * 2 + gray[i, j + 2] * 1 - gray[i + 2, j] * 1 - gray[
                    i + 2, j + 1] * 2 - \
                     gray[i + 2, j + 2] * 1
                gx = gray[i, j] + gray[i + 1, j] * 2 + gray[i + 2, j] - gray[i, j + 2] - gray[i + 1, j + 2] * 2 - gray[
                    i + 2, j + 2]
                grad = math.sqrt(gx * gx + gy * gy)
                if grad > 100:
                    dst[i, j] = 0
                else:
                    dst[i, j] = 255
        cv2.imshow('dst', dst)
        cv2.imwrite(self.img_after, dst)
        cv2.waitKey(0)

# test4 = ChangeSketch4('D:/Code/github/Test/2.jpg', 'D:/Code/github/Test/sketch_result/24.jpg')
# test4.change()

#############################分隔线######################################
#第五版

class ChangeSketch5:
    def __init__(self, img_before, img_after):
        self.img_before = img_before
        self.img_after = img_after

    def change(self):
        import cv2

        img_gray = cv2.imread(self.img_before, 0)
        img_blur = cv2.GaussianBlur(img_gray, ksize=(21, 21),
                            sigmaX=0, sigmaY=0)  #调整高斯滤波值
        img_blend = cv2.divide(img_gray, img_blur, scale=256)
        img_result = cv2.cvtColor(img_blend, cv2.COLOR_GRAY2BGR)
        cv2.imshow("imshow", img_result)
        cv2.imwrite(self.img_after, img_result)
        cv2.waitKey(0)  # 防止图片窗口闪退

test5 = ChangeSketch5('D:/Code/github/Test/2.jpg', 'D:/Code/github/Test/sketch_result/25.jpg')
test5.change()

class ChangeSketch6:
    def __init__(self, img_before, img_after):
        self.img_before = img_before
        self.img_after = img_after

    def change(self):
        import cv2

        img_rgb = cv2.imread(self.img_before)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)#滤波函数
        img_edge = cv2.adaptiveThreshold(img_gray, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, blockSize=3, C=2)
        cv2.imshow("imshow", img_edge)
        cv2.imwrite(self.img_after, img_edge)
        cv2.waitKey(0)  # 防止图片窗口闪退

test6 = ChangeSketch6('D:/Code/github/Test/2.jpg', 'D:/Code/github/Test/sketch_result/26.jpg')
test6.change()












