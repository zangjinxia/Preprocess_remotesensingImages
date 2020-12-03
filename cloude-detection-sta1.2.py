'''
@author zangjinxia
@date 2020-12-3
@brief 实现GF1两种传感器和GF2和HY1C的云量统计功能
'''
import ogr
import gdal
import numpy as np
# import cv2 as cv
# import scipy.stats as st
# import distutils
import os
import sys
from skimage import data, filters
# import matplotlib.pyplot as plt
from time import *
from numba import jit
#!/usr/bin/env Python
# coding=utf-8

class Dataset:
    def read_img(self, filename):
        dataset = gdal.Open(filename)

        width = dataset.RasterXSize
        height = dataset.RasterYSize
        band = dataset.RasterCount
        im_data = dataset.ReadAsArray(0, 0, width, height)

        geotrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        # data = np.zeros([width, height, band])

        return im_data,proj,geotrans

    def read_img2(self, filename):
        dataset = gdal.Open(filename)

        width = dataset.RasterXSize
        height = dataset.RasterYSize
        band = dataset.RasterCount
        im_data = dataset.ReadAsArray(0, 0, width, height)
        type = im_data.dtype
        geotrans = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        # data = np.zeros([width, height, band])

        return im_data, proj, geotrans,band,width,height,type

    def write_tiff(self, filename, proj, geotrans, data):
        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64
        # 判断栅格数据的数据类型
        if 'int8' in data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(data.shape) == 3:
            bands, height, width = data.shape
        else:
            bands = 1
            height, width = data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, width, height, bands, datatype)

        dataset.SetGeoTransform(geotrans)
        dataset.SetProjection(proj)

        if bands == 1:
            dataset.GetRasterBand(1).WriteArray(data)
        else:
            for i in range(bands):
                dataset.GetRasterBand(i + 1).WriteArray(data[i])
        del dataset

    def write_tiff2(self, filename, proj, geotrans, data,minx,maxy):
        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64
        # 判断栅格数据的数据类型
        if 'int8' in data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(data.shape) == 3:
            bands, height, width = data.shape
        else:
            bands = 1
            height, width = data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, width, height, bands, datatype)
        geotrans_update = (minx, geotrans[1],geotrans[2],maxy,geotrans[4],geotrans[5])
        dataset.SetGeoTransform(geotrans_update)
        dataset.SetProjection(proj)

        if bands == 1:
            dataset.GetRasterBand(1).WriteArray(data)
        else:
            for i in range(bands):
                dataset.GetRasterBand(i + 1).WriteArray(data[i])
        del dataset

@jit
def guiyihua(array,wid,hegt):
    ymax = 255
    ymin = 0
    xmax = np.nanmax(array)   #如果矩阵有nan值时，求最大值则用该函数，没有时用np.max()
    xmin = np.nanmin(array)
    # xmax = max(map(max, array))
    # xmin = min(map(min, array))
    for i in range(wid):
        for j in range(hegt):
            array[i][j] = round(((ymax - ymin) * (array[i][j] - xmin) / (xmax - xmin)) + ymin)
    return array
def NDWI(B1,B2):
    """
求归一化水体指数
    :param B1: 绿波段
    :param B2: 近红波段
    :return: NDWI矩阵
    """
    # result = (float(B1)-float(B2))/(float(B1)+float(B2))
    result1 = (B1 - B2) / (B1 + B2)
    return result1
def cloud_detection(data):
    width, height = data.shape
    # 对蓝波段进行阈值分割
    b = data
    blue = b.copy()
    blue_gyh = guiyihua(blue, width, height)
    imgblue = np.uint8(blue_gyh)
    imgblue = np.where(imgblue==0,imgblue.max(),imgblue)
    # imgblue = np.where(imgblue == 0, imgblue.min(), imgblue)
    thresh = filters.threshold_otsu(imgblue)  # 返回一个阈值
    th1_blue = (blue <= thresh) * 1.0

    # plt.imshow(th1_blue)
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()

    return th1_blue

if __name__ == '__main__':
    begin_time = time()
    print('start')
    if len(sys.argv) != 3:
        # distutils.log.error("not enougth input parameters")
        sys.exit(-1)
    # 创建文件
    rasterPath = sys.argv[1]   #tif影像   I:/data/GF_radiance/Vege_test/GF1caijian.tif
    txtpath = sys.argv[2]   #存放结果信息的txt文件
    path = os.path.dirname(txtpath)
    if not os.path.exists(path):
        os.makedirs(path)
    open(txtpath,'w')

   # 判断数据是什么类型的
    tifName = os.path.basename(rasterPath)
    satelliteType = tifName[0:3]
    # sensorType = tifName[4:7]
    if satelliteType =='GF1' or 'GF2' or 'GF6':
        #读取数据
        dataset = Dataset()
        data, proj, geotrans = dataset.read_img(rasterPath)
        b1 = np.array(data[0])
        b2 = np.array(data[1])
        b3 = np.array(data[2])
        b4 = np.array(data[3])
        gf1 = np.array([b1, b2, b3, b4])

        width,height = b1.shape

        #云检测
        blueband = gf1[0]
        th1blue = cloud_detection(blueband)
    if satelliteType == 'H1C':
        sensorType = tifName[9:12]
        if sensorType == 'CZI':
            dataset = Dataset()
            data, proj, geotrans = dataset.read_img(rasterPath)
            b1 = np.array(data[0])
            b2 = np.array(data[1])
            b3 = np.array(data[2])
            b4 = np.array(data[3])
            gf1 = np.array([b1, b2, b3, b4])
            width, height = b1.shape
            # 云检测
            blueband = gf1[0]
            th1blue = cloud_detection(blueband)

    # print('云检测完成')
    # print('开始统计云量')

    cloud_num = sum(sum(th1blue == 0))

    cloud_percent = cloud_num/(width*height)*100
    f = open(txtpath, 'w')
    f.write(str(cloud_percent))
    print('云量为:',round(cloud_percent,2),'%')

    end_time = time()
    run_time = end_time-begin_time
    # print ('该循环程序运行时间：',run_time)

