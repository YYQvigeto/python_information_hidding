import os,signal
import matplotlib.pyplot as plt
import hashlib
import numpy as np
import math
import multiprocessing
import time
from moviepy.editor import *
import cv2
from multiprocessing import pool

# from numba import jit
# %matplotlib inline


# 通用方法，可以读取中文名字的图片，只需要输入图片是几通道即可
# cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
def readimg(filename, mode):
    raw_data = np.fromfile(filename, dtype=np.uint8)  # 先用numpy把图片文件存入内存：raw_data，把图片数据看做是纯字节数据
    img = cv2.imdecode(raw_data, mode)  # 从内存数据读入图片
    return img


# opencv读取的格式 是 BGR!


# 图像展示方法
def showImg(img, windowName):
    # 图像展示
    cv2.imshow(windowName, img)
    # 按任意键 向下执行，如果不是0则等待n 毫秒
    cv2.waitKey(0)
    # 清除展示窗口
    cv2.destroyAllWindows()


# showImg(img_color,"测试")


# hash方法
class Hashing:
    # 传入字节数组，获取十六进制的md5值
    def md5(self, data):
        return hashlib.md5(data).hexdigest()

    # 传入文件名，读取对应的字节
    def getBytesFromFile(self, fileName):
        content = open(fileName, "rb")
        content = content.read()
        return content


# 混沌系统类
class ChaoSystem:
    # 基于三角函数的混沌系统，传入四个参数，产生number长度的伪随机二进制字符串
    # 方法：X_n+1 = A * sin(X_n-X_B)^2
    def generate_StrBin(self, A, X_B, X_1, number):
        res = ""
        temp = X_1
        for i in range(number):
            temp = A * pow(math.sin(temp - X_B), 2)
            #                 print(str(temp) +  "\n")
            if (temp >= A * 2 / 3):
                res += "1"
            else:
                res += "0"
        return res

    # 使用md5的hash的十六进制字符串作为密钥，产生基于混沌系统的伪随机二进制字符串
    # hashStr：十六进制md5，max_dct表示待嵌入的dct矩阵数量
    def use_hashStr_toGenerate(self, hashStr, max_dct):
        A = abs(int(hashStr[0:8], 16)) % 100
        if (A < 10):
            A += 10
        X_B = abs(int(hashStr[8:16], 16)) % 100
        if (X_B < 10):
            X_B += 10
        X_1 = abs(int(hashStr[16:24], 16)) % 100
        if (X_1 < 10):
            X_1 += 10
        number = abs(int(hashStr[24:32], 16)) % 100
        if (number < int(0.5 * max_dct)):
            number = int(0.7 * max_dct)
        bin_str = self.generate_StrBin(A, X_B, X_1, number)
        return bin_str


class DCT_WaterMark:

    def DCT_process(self, img, binStr, x):
        """
        嵌入图像主函数,提供三通道img图像，binStr为二进制字符串,x为阈值，x越大表示嵌入其嵌入强度越高
        :param img: 三通道数组
        :param binStr: 二进制字符串
        :param x: 嵌入强度阈值。
        :return: 处理过后的img和嵌入数据
        """
        (b, g, r) = cv2.split(img)  # 图像分层

        number = self.DCT_process_OneChannel(r, binStr, x)

        if (number < len(binStr)):
            binStr_2 = binStr[number:]
            print(len(binStr_2), "嵌入了一次图层，剩下的长度为")

            number_2 = self.DCT_process_OneChannel(g, binStr_2, x)

            if (number_2 < len(binStr_2)):
                binStr_3 = binStr_2[number_2:]
                number_3 = self.DCT_process_OneChannel(b, binStr_3, x)
                if (number_3 < len(binStr_3)):
                    raise Exception("二进制字符串数据过多，无法全部嵌入！")
        img_merge = cv2.merge((b, g, r))

        return img_merge

    def DCT_process_OneChannel(self, img_oneChannel, binStr, x):
        """
        将二进制字符串嵌入单通道图像中，结果返回实际嵌入的数量
        :param img_oneChannel: 单通道数组（二维数组）
        :param binStr: 二进制字符串
        :param x: 嵌入强度
        :return: 返回实际嵌入的数据数量
        """
        #         print(img_oneChannel.shape[0]," ",img_oneChannel.shape[1],"\n")
        number = 0  # 嵌入数量，或者待嵌入坐标
        y_1 = 0
        x_1 = 0  # 待嵌入 8*8矩阵的左上坐标
        n = 8  # 决定是 8*8 矩阵

        img_height = img_oneChannel.shape[0]  # 图像高度
        img_width = img_oneChannel.shape[1]  # 图像宽度

        while (number < len(binStr)):
            img = img_oneChannel[y_1:y_1 + 8, x_1:x_1 + 8]  # 截取8*8的数组
            #             print(img.shape)

            # 将8*8矩阵精度转换 float32
            img = np.float32(img)
            # print(img)

            # dct变换
            img = cv2.dct(img)
            # print(img)
            number_1 = img[4, 1]  # 获取坐标[4,1]的dct系数
            number_2 = img[3, 2]  # 获取坐标[3,2]的dct系数

            a = x  # 嵌入强度

            if (binStr[number] == "1"):
                # 如果待嵌入数据是 1
                if (number_1 >= number_2):
                    if (abs(number_2 - number_1) <= a):  # 阈值比较
                        if (number_1 >= number_2):
                            number_1 += a
                        else:
                            number_2 += a
                        img[4, 1] = number_1
                        img[3, 2] = number_2
                else:
                    if (abs(number_2 - number_1) <= a):
                        if (number_1 >= number_2):
                            number_1 += a
                        else:
                            number_2 += a
                    img[4, 1] = number_2
                    img[3, 2] = number_1





            else:
                if (number_1 >= number_2):
                    if (abs(number_2 - number_1) <= a):
                        if (number_1 >= number_2):
                            number_1 += a
                        else:
                            number_2 += a
                    img[4, 1] = number_2
                    img[3, 2] = number_1
                else:
                    if (abs(number_2 - number_1) <= a):
                        if (number_1 >= number_2):
                            number_1 += a
                        else:
                            number_2 += a
                    img[4, 1] = number_1
                    img[3, 2] = number_2

            # dct逆变换
            img = cv2.idct(img)

            # 四舍五入
            img = np.floor(img)
            # print(img)

            img = self.searchAndReplace(img).astype(np.uint8)

            img_oneChannel[y_1:y_1 + 8, x_1:x_1 + 8] = img

            # img = cv2.dct(np.float32(img))
            # print(img[4,1]," ",img[3,2],"\n")

            number += 1

            # 待嵌入矩阵坐标改变
            x_1 += 8
            if ((x_1 + 8) > img_width):
                x_1 = 0
                y_1 += 8
                if (y_1 + 8 > img_height):
                    return number

        return number

    def max_dct(self, img):
        """
        传入三通道图像数组，返回该图像图层最多有多少个8*8矩阵
        """
        x = img.shape[1] * 1.0 / 8
        y = img.shape[0] * 1.0 / 8

        x_1 = math.floor(x)
        y_1 = math.floor(y)

        return (int)(x_1 * y_1 * 3)

    def max_dct_byWidthAndHeight(self, height, width):
        """
        传入矩阵高和宽，计算出其最多有多少个8*8矩阵
        """
        y = height * 1.0 / 8
        x = width * 1.0 / 8
        y_1 = math.floor(y)
        x_1 = math.floor(x)

        return (int)(x_1 * y_1)

    def searchAndReplace(self, img_dct):
        """
        防止矩阵元素溢出，如果有一个元素超过255，则重置成255
        """
        for a in range(8):
            for b in range(8):
                if (img_dct[a, b] > 255):
                    img_dct[a, b] = 255
                if (img_dct[a, b] < 0):
                    img_dct[a, b] = 0
        return img_dct

    def DCT_converse(self, img, number):
        """
        提取三通道图像的二进制水印序列
        :param img : 三通道图像
        :param number : 水印数量
        """
        res = ""
        (b, g, r) = cv2.split(img)
        str_1 = self.DCT_converse_OneChannel(r, number);
        res += str_1

        if (len(str_1) < number):
            number_2 = number - len(str_1)
            str_2 = self.DCT_converse_OneChannel(g, number_2)
            res += str_2

            if (len(str_2) < number_2):
                number_3 = number_2 - len(str_2)
                str_3 = self.DCT_converse_OneChannel(b, number_3)
                res += str_3
                if (len(str_3) < number_3):
                    raise Exception("传入的水印数量过大，无法完全提取出来")

        return res

    def DCT_converse_OneChannel(self, img_oneChannel, bin_length):
        """
        处理单通道图像数组，返回二进制字符串 尽量提取出来
        :param img_oneChannel : 单通道图像
        :param bin_length : 字符串长度
        :return : 结果返回二进制字符串
        """
        res = ""
        number = 0  # 提取数量，或者待提取错标
        y_1 = 0  # 待提取8*8矩阵的左上角坐标
        x_1 = 0
        n = 8  # 8*8矩阵

        img_height = img_oneChannel.shape[0]  # 图像高度
        img_width = img_oneChannel.shape[1]  # 图像宽度

        while (number < bin_length):
            img = img_oneChannel[y_1:y_1 + 8, x_1:x_1 + 8]

            # 将8*8矩阵精度转换 float32
            img = np.float32(img)
            # print(img)

            # dct变换
            img = cv2.dct(img)
            # print(img)
            number_1 = img[4, 1]  # 获取坐标[4,1]的dct系数
            number_2 = img[3, 2]  # 获取坐标[3,2]的dct系数

            if (number_1 >= number_2):
                res += "1"
            else:
                res += "0"
            number += 1

            # 待提取坐标矩阵改变
            x_1 += 8
            if ((x_1 + 8) > img_width):
                x_1 = 0
                y_1 += 8
                if (y_1 + 8 > img_height):
                    return res

        return res

    def rate_accuracy(self, temp_1, temp_2):
        """
        对比两串二进制字符串，返回正确率
        """
        res = 0
        if (len(temp_1) != len(temp_2)):
            raise Exception("两个字符串长度不一致")
        n = 0
        for i in range(len(temp_1)):
            if (temp_1[i] == temp_2[i]):
                n += 1.0

        res = n * 1.0 / len(temp_1)

        return res

    def accurate_mod_byMd5(self, md5):
        """
        根据md5的十六进制字符串计算出res，res值域为[0,2] 0-b 1-g 2-r
        """
        number = int(md5, 16)
        # print(number)
        return number % 3


class My_Frames:
    def __init__(self, no, frames):
        self.no = no  # 优先级
        self.frames = frames  # 三通道帧的集合
        self.flag_end = False  # 标记该片段是否是最后的结尾

    def getNo(self):
        return self.no

    def getFrames(self):
        return self.frames

    def setFrames(self, frames):
        self.frames = frames

    def getFlag_End(self):
        return self.flag_end

    def setFlag_End(self, flag):
        self.flag_end = flag


def getNo(myFrames):
    return myFrames.getNo()


# 图像水印类
class VideoWaterMark_DCT:
    def __init__(self):
        self.dct_WaterMark = DCT_WaterMark()
        self.hashing = Hashing()
        self.chaoSystem = ChaoSystem()
        self.q = multiprocessing.JoinableQueue(20)  # 并发传递数据的队列 生产者不断读取视频帧，然后包装成MyFrames类，添加到这里
        self.complete_q = multiprocessing.JoinableQueue(50)  # 消费者线程 将数据进行水印嵌入后，添加到这里。
        self.cpu_number = multiprocessing.cpu_count()  # cpu核心数量

    def insert_waterMark(self, videoPath, savePath, keyPath, a):
        """
        该方法单线程方法，运行效率太低了
        :param videoPath 视频路径
        :param keyPath 密钥证书路径
        :param a 嵌入强度
        """
        # 读取视频
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened()):
            # fps帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 视频宽
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            # 视频高
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            # 计算出单图层多少个8*8矩阵
            max_dct = self.dct_WaterMark.max_dct_byWidthAndHeight(height, width)
            # 根据密钥信息，计算出 嵌入的二进制字符串
            md5 = self.hashing.md5(self.hashing.getBytesFromFile(keyPath))
            strBin = self.chaoSystem.use_hashStr_toGenerate(md5, max_dct)
            # 计算出 嵌入模式 0-嵌入b层，1-嵌入g层,2-b嵌入r层
            mod = self.dct_WaterMark.accurate_mod_byMd5(md5)

            print(max_dct, "max_dct\n")
            print(md5, "md5\n")
            print(len(strBin), "strBin\n")
            print(mod, "mod\n")
            out = cv2.VideoWriter(savePath, int(fourcc), int(fps), (int(width), int(height)), True)
            number = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    # 拆分rgb
                    #                     b,g,r = cv2.split(frame)
                    #                     temp = r - frame[:,:,2]
                    #                     print(temp)
                    # showImg(frame,"_1")

                    temp_img = frame[:, :, mod]
                    # print("之前",temp_img)
                    self.dct_WaterMark.DCT_process_OneChannel(temp_img, strBin, a)
                    # print("之后", temp_img)
                    # showImg(frame, "_1_")
                    frame[:, :, mod] = temp_img
                    # 对视频每一帧进行dct水印写入

                    out.write(frame)
                    # showImg(frame,"1")
                    # number += 1
                    # if(number == 18): break

                    # 调试
                    # converse = self.dct_WaterMark.DCT_converse_OneChannel(frame[:,:,mod],len(strBin))
                    # rate = self.dct_WaterMark.rate_accuracy(converse,strBin)
                    # print("对比率",rate)
                else:
                    break

            cap.release()
            out.release()

    def sole_the_audioProblem(self, fromPath_1, fromPath_2, toPath):
        """
        将fromPath_1的音频提取并和fromPath_2的视频相结合，生成toPath文件，注意toPath必须是avi格式的
        :param fromPath_1:
        :param fromPath_2:
        :param toPath:
        :return:
        """
        # 获取视频
        video_clip = VideoFileClip(fromPath_2)
        # 获取音频
        audio_clip = AudioFileClip(fromPath_1)
        # 视频里面添加声音
        final_video = video_clip.set_audio(audio_clip)
        # 保存视频
        final_video.write_videofile(toPath, codec="rawvideo")

    def multiProcessing_foreplay(self, videoPath):
        """
        多进程执行之前需要做的资源划分，将视频每一帧读取到list中，结果返回
        :param videoPath: 载体视频路径
        :return:
        """
        # 获取cpu核心数量
        cpu_number = multiprocessing.cpu_count()
        print(cpu_number)  # 我的是12个核心数

        # 将每一帧读取到list_frames中
        list_frames = []
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened()):
            # fps帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 视频宽
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            # 视频高
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    #
                    list_frames.append(frame)
                else:
                    break
        cap.release();
        print(len(list_frames))
        return list_frames
        # for i in range(cpu_number):
        # 读取视频

    def videoFrames_insert_waterMark(self, myFrames: My_Frames, binStr, a, mod, q: multiprocessing.Queue):
        """
        传入帧列表，根据二进制字符串和嵌入强度进行水印嵌入
        :param myFrames: 帧列表
        :param binStr: 二进制字符串
        :param a: 嵌入强度
        :param mod: 嵌入模式 0-b 1-g 2-r
        :param q: 并发开发的队列
        :return: 返回嵌入后的帧列表
        """
        for frame in myFrames.getFrames():
            temp_img = frame[:, :, mod]
            self.dct_WaterMark.DCT_process_OneChannel(temp_img, binStr, a)

            frame[:, :, mod] = temp_img
            converse_binStr = self.dct_WaterMark.DCT_converse_OneChannel(frame[:, :, mod], len(binStr))
            rate = self.dct_WaterMark.rate_accuracy(converse_binStr, binStr)
            print("rate", rate)

        q.put(myFrames)

    def insert_main(self, videoPath, savePath, keyPath, a):
        """
        这是嵌入主进程，在里面生成n个进程不断进行嵌入信息
        :param videoPath:
        :param savePath:
        :param keyPath:
        :param a:
        :return:
        """
        if (__name__ == "__main__"):

            print(__name__)
            # 获取cpu核心数量
            cpu_number = multiprocessing.cpu_count()
            print(cpu_number)  # 我的是12个核心数

            list_temp = []

            cap = cv2.VideoCapture(videoPath)
            # fps帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 视频宽
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            # 视频高
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            # 计算出单图层多少个8*8矩阵
            max_dct = self.dct_WaterMark.max_dct_byWidthAndHeight(height, width)
            # 根据密钥信息，计算出 嵌入的二进制字符串
            md5 = self.hashing.md5(self.hashing.getBytesFromFile(keyPath))
            strBin = self.chaoSystem.use_hashStr_toGenerate(md5, max_dct)
            # 计算出 嵌入模式 0-嵌入b层，1-嵌入g层,2-b嵌入r层
            mod = self.dct_WaterMark.accurate_mod_byMd5(md5)

            n = 0;  # 记录第几个frames
            list_frame = []

            out = cv2.VideoWriter(savePath, int(fourcc), int(fps), (int(width), int(height)), True)

            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    list_frame.append(frame)
                    if (len(list_frame) >= 10):
                        n += 1
                        list_temp.append(My_Frames(n, list_frame))
                        list_frame = []
                    if (len(list_temp) >= cpu_number):
                        n = 0;
                        # 创建 len(list_frame)长度的进程进行数据嵌入
                        process_list = []
                        for i in range(len(list_temp)):
                            temp_process = multiprocessing.Process(target=self.videoFrames_insert_waterMark,
                                                                   args=(list_temp[i], strBin, a, mod, self.q))
                            process_list.append(temp_process)
                            temp_process.start()
                        for i in process_list:
                            i.join()
                        # 只有子进程全部运行结束后才可以继续
                        # 将嵌入完成的数据 写入磁盘中

                        # for i in list_temp:
                        #     for j in i:
                        #         out.write(j)

                        # 从q队列中获取数据
                        list_myFrames = []
                        for i in range(self.q.qsize()):
                            temp_myFrames = self.q.get()
                            list_myFrames.append(temp_myFrames)
                        list_myFrames = list_myFrames.sort(key=getNo)

                        for i in list_myFrames:
                            for j in i.getFrames():
                                out.write(j)

                        # 调试
                        converse_binStr = self.dct_WaterMark.DCT_converse_OneChannel(
                            list_myFrames[0].getFrames()[0][:, :, mod],
                            len(strBin))
                        rate = self.dct_WaterMark.rate_accuracy(converse_binStr, strBin)
                        print(rate, "---调试")

                        # 释放list_frame
                        list_temp.clear()
                        process_list.clear()
                else:

                    list_temp.append(My_Frames(n, list_frame))
                    list_frame = []
                    process_list = []
                    for i in range(len(list_temp)):
                        temp_process = multiprocessing.Process(target=self.videoFrames_insert_waterMark,
                                                               args=(list_temp[i], strBin, a, mod, self.q))
                        process_list.append(temp_process)
                        temp_process.start()
                    for i in process_list:
                        i.join()
                    # 只有子进程全部运行结束后才可以继续
                    # 将嵌入完成的数据 写入磁盘中
                    # for i in list_temp:
                    #     for j in i:
                    #         out.write(j)
                    list_myFrames = []
                    for i in range(self.q.qsize()):
                        temp_myFrames = self.q.get()
                        list_myFrames.append(temp_myFrames)
                    list_myFrames = list_myFrames.sort(key=getNo)
                    for i in list_myFrames:
                        for j in i.getFrames():
                            out.write(j)

                    # 调试
                    converse_binStr = self.dct_WaterMark.DCT_converse_OneChannel(
                        list_myFrames[0].getFrames()[0][:, :, mod],
                        len(strBin))
                    rate = self.dct_WaterMark.rate_accuracy(converse_binStr, strBin)
                    print(rate, "---调试")

                    # 释放list_frame
                    list_temp.clear()
                    process_list.clear()
                    break

            cap.release()
            out.release()

    def converse_video(self, videoPath, keyPath):
        # 根据密钥信息获取水印序列
        cap = cv2.VideoCapture(videoPath)
        # fps帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 视频宽
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # 视频高
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 计算出单图层多少个8*8矩阵
        max_dct = self.dct_WaterMark.max_dct_byWidthAndHeight(height, width)
        # 根据密钥信息，计算出 嵌入的二进制字符串
        md5 = self.hashing.md5(self.hashing.getBytesFromFile(keyPath))
        strBin = self.chaoSystem.use_hashStr_toGenerate(md5, max_dct)
        # 计算出 嵌入模式 0-嵌入b层，1-嵌入g层,2-b嵌入r层
        mod = self.dct_WaterMark.accurate_mod_byMd5(md5)

        number = 0;
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                converse_strBin = self.dct_WaterMark.DCT_converse_OneChannel(frame[:, :, mod], len(strBin))

                ac = self.dct_WaterMark.rate_accuracy(strBin, converse_strBin)

                print(number, "帧", ac)

                number += 1
            else:
                break

        pass

    def producer(self, videoPath, q: multiprocessing.JoinableQueue):
        """
        这是消费者线程，它不断读取视频帧，然后包装成MyFrames类，添加到q队列中
        :param q:
        :param videoPath:
        :return:
        """
        print("进入生产者主方法！！！！！")
        # 读取视频
        cap = cv2.VideoCapture(videoPath)
        no = 0  # 记录MyFrames的顺序
        list_frames = []  # 三通道视频帧 的集合
        while (cap.isOpened()):
            ret, frame = cap.read()
            if (ret == True):
                list_frames.append(frame)
                if (len(list_frames) >= 10):
                    my_1 = My_Frames(no, list_frames)
                    no += 1
                    # 放到队列中
                    q.put(my_1)

                    print("生产者：序号", no - 1, "将myFrames写入q中", len(my_1.getFrames()))

                    # 清空list_frames
                    list_frames = []


            else:
                # 表示该片段是结尾的片段
                my_1 = My_Frames(no, list_frames)
                my_1.setFlag_End(True)
                no += 1
                # 放到队列中
                q.put(my_1)

                print("生产者：序号", no - 1, "将结尾myFrames写入q中", ",序号为", my_1.getNo(), ",flag为", my_1.getFlag_End(),"长度为",len(list_frames))

                # 清空list_frames
                list_frames = []

                q.join()
                cap.release()
                break

    def consumer(self, keyPath, height, width, a, q: multiprocessing.JoinableQueue,
                 complete_q: multiprocessing.JoinableQueue):

        print("进入消费者进程！！")

        # 计算出单图层多少个8*8矩阵
        max_dct = self.dct_WaterMark.max_dct_byWidthAndHeight(height, width)
        # 根据密钥信息，计算出 嵌入的二进制字符串
        md5 = self.hashing.md5(self.hashing.getBytesFromFile(keyPath))
        strBin = self.chaoSystem.use_hashStr_toGenerate(md5, max_dct)
        # 计算出 嵌入模式 0-嵌入b层，1-嵌入g层,2-b嵌入r层
        mod = self.dct_WaterMark.accurate_mod_byMd5(md5)

        count = 0 # 消费者从q中读取数据 失败的次数
        while (True):
            my_frames_1 = None
            try:
                my_frames_1 = q.get(block=True, timeout=5)
                # 如果消费者一直读取不是结束符号
                list_frames = my_frames_1.getFrames()
                print("消费者：刚获取q中数据，", my_frames_1)
                for i in range(len(list_frames)):
                    temp_img = list_frames[i][:, :, mod]
                    self.dct_WaterMark.DCT_process_OneChannel(temp_img, strBin, a)
                    list_frames[i][:, :, mod] = temp_img

                # 数据处理完后 放到complete_q中
                my_frames_2 = My_Frames(my_frames_1.getNo(), list_frames)
                my_frames_2.setFlag_End(my_frames_1.getFlag_End())
                complete_q.put(my_frames_2)
                # self.complete_q.put(my_frames_1.setFrames(list_frames))
                # 一次处理完成
                q.task_done()
                print("消费者：", "处理序号为", my_frames_1.getNo(),"的数据")
                count = 0
            except BaseException as e:
                count += 1
                print("消费者，处理错误，次数",count,"次！！！！！！！！！！！！！！！！")
                print(e)
            if(count >= 5):
                exit(0)

    def saveTheInsertVideo(self, complete_q: multiprocessing.JoinableQueue, savePath, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(savePath, int(fourcc), int(fps), (int(width), int(height)), True)
        # 该方法是也算是一个消费者，它不断消费complete_q中的元素
        dic = dict()
        no = 0  # 从no为1的myFrames进行写入磁盘
        while (True):
            try:
                # 最多阻塞5秒
                my_frames_1 = complete_q.get(block=True, timeout=3)
                print("磁盘写入者：序号", my_frames_1.getNo(), my_frames_1.getFlag_End())
                dic[my_frames_1.getNo()] = my_frames_1
                print("写入字典", dic[my_frames_1.getNo()], "no为", my_frames_1.getNo())
                complete_q.task_done()
            except BaseException:
                print("磁盘写入者：正在不断进行读取并写入磁盘，请稍等。")

            if (no in dic):
                print("磁盘写入者：正在进入写入流程")
                my_frames_temp = dic.pop(no)
                no += 1
                print(my_frames_temp, "从字典中找到元素了", "序号为", my_frames_temp.getNo(), "，flag为",
                      my_frames_temp.getFlag_End())
                # 将帧集合写入到磁盘中
                for i in my_frames_temp.getFrames():
                    out.write(i)
                # 如果该片段是最后的就退出该线程
                if (my_frames_temp.getFlag_End() == True):
                    print("磁盘写入者：这是最后一段数据，正在退出进程！！！！！")
                    out.release()
                    # 子进程杀死自己
                    # os.kill(os.getpid(), signal.SIGTERM)
                    exit(0)
                    return None


# video_ = VideoWaterMark_DCT()
# video_.insert_waterMark("QHC_3.mp4", "QHC_temp.avi", "miyao", 100)
# video_.sole_the_audioProblem("QHC_3.mp4","QHC_temp.avi","QHC_to.avi")
# video_.multiProcessing_foreplay("QHC.mp4")
# if(__name__ == "__main__"):
#     videoPath="QHC_2.mp4"
#     savePath="QHC_temp.avi"
#     keyPath="miyao"
#     a=15
# video_.converse_video("QHC_to.avi","miyao")
# video_.converse_video("QHC_temp_3.avi","miyao")
if __name__ == "__main__":
    # 该方法是 多进程水印嵌入
    video_ = VideoWaterMark_DCT()
    # 读取视频
    videoPath = "QHC_3.mp4"
    savePath = "QHC_temp_3.avi"
    keyPath = "miyao"
    a = 100
    cap = cv2.VideoCapture(videoPath)
    # fps帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 视频宽
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 视频高
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 计算出单图层多少个8*8矩阵
    max_dct = video_.dct_WaterMark.max_dct_byWidthAndHeight(height, width)
    # 根据密钥信息，计算出 嵌入的二进制字符串
    md5 = video_.hashing.md5(video_.hashing.getBytesFromFile(keyPath))
    strBin = video_.chaoSystem.use_hashStr_toGenerate(md5, max_dct)
    # 计算出 嵌入模式 0-嵌入b层，1-嵌入g层,2-b嵌入r层
    mod = video_.dct_WaterMark.accurate_mod_byMd5(md5)

    print(max_dct, "max_dct\n")
    print(md5, "md5\n")
    print(len(strBin), "strBin\n")
    print(mod, "mod\n")

    # 生产者，不断读取视频，并写入到q中
    process_producer = multiprocessing.Process(target=video_.producer, args=(videoPath, video_.q))

    # 消费者
    process_consumers = []
    for i in range(video_.cpu_number):
        process_consumer = multiprocessing.Process(target=video_.consumer,
                                                   args=(keyPath, height, width, a, video_.q, video_.complete_q))
        process_consumer.daemon = True
        process_consumers.append(process_consumer)
        print("总共生成", len(process_consumers), "个消费者")

    # 磁盘写入者
    process_saver = multiprocessing.Process(target=video_.saveTheInsertVideo,
                                            args=(video_.complete_q, savePath, fps, width, height))


    process_producer.start()
    time.sleep(3)
    print("开启生产者")
    for process_consumer in process_consumers:
        process_consumer.start()
    print("开启消费者")

    process_saver.start()
    print("开启磁盘写入者")
    process_producer.join()
    print("\n-----生成者线程和消费者进程结束了，正在等待磁盘写入者进程结束。-----\n")
    process_saver.join()
    print("磁盘写入者进程已经结束了")

    # 啥子父进程
    os.kill(os.getpid(), signal.SIGTERM)

    # video_.saveTheInsertVideo(video_.complete_q, savePath, fps, width, height)

    # 开启消费者的消费者，即写入磁盘进程

    # temp1 = video_.complete_q.get()
    # print("temp1", temp1.getNo())
    # temp2 = video_.complete_q.get()
    # print("temp2", temp2.getNo())
    #
    # # 对MyFrames进行测试
    # res_binStr = video_.dct_WaterMark.DCT_converse_OneChannel(temp1.getFrames()[0][:,:,mod],len(strBin))
    # rate = video_.dct_WaterMark.rate_accuracy(strBin,res_binStr)
    # print(rate,"计算率")

def test(meg):
    print(meg)
    time.sleep(5)

# 使用进程池子实现,该方法有bug，暂时不要使用：
if __name__ == "__main__2":
    # 请忽视这里的方法，该方法是多进程水印嵌入，但是还有待完善。
    video_ = VideoWaterMark_DCT()
    # 读取视频
    videoPath = "QHC_4.mp4"
    savePath = "QHC_temp_4.avi"
    keyPath = "miyao"
    a = 100
    cap = cv2.VideoCapture(videoPath)
    # fps帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 视频宽
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 视频高
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 计算出单图层多少个8*8矩阵
    max_dct = video_.dct_WaterMark.max_dct_byWidthAndHeight(height, width)
    # 根据密钥信息，计算出 嵌入的二进制字符串
    md5 = video_.hashing.md5(video_.hashing.getBytesFromFile(keyPath))
    strBin = video_.chaoSystem.use_hashStr_toGenerate(md5, max_dct)
    # 计算出 嵌入模式 0-嵌入b层，1-嵌入g层,2-b嵌入r层
    mod = video_.dct_WaterMark.accurate_mod_byMd5(md5)

    print(max_dct, "max_dct\n")
    print(md5, "md5\n")
    print(len(strBin), "strBin\n")
    print(mod, "mod\n")

    pool =  multiprocessing.Pool(processes=10)


    # 生产者进程
    pool.apply_async(func=video_.producer,args=(videoPath, video_.q))
    #
    # #消费者进程
    # for i in range(1):
    #     pool.apply_async(func=video_.consumer, args=(keyPath, height, width, a, video_.q, video_.complete_q))
    #
    # # 磁盘写入者进程
    # pool.apply_async(func=video_.saveTheInsertVideo, args=(video_.complete_q, savePath, fps, width, height))

    # for i in range(8):
    #
    #     pool.apply_async(func=test,args=("这是测试",))

    pool.close()
    pool.join()
    print(pool)



