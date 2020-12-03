'''
@author zangjinxia
@date 2020-12-3
@brief 实现GF1卫星影像的海陆掩模，输出陆地文件和海水文件共两个文件
'''

import matplotlib.pyplot as plt
import gdal
import os
import numpy as np
import cv2 as cv
import sys
import distutils
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



    #将矩阵归一化到0-255范围
def guiyihua(array,wid,hegt):
    ymax = 255
    ymin = 0
    xmax = max(map(max, array))
    xmin = min(map(min, array))
    for i in range(wid):
        for j in range(hegt):
            array[i][j] = round(((ymax - ymin) * (array[i][j] - xmin) / (xmax - xmin)) + ymin)
            # array[i][j] = round(((ymax - ymin) * (array[i][j] - 65) / (xmax - 65)) + ymin)
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

def NDVI(B1,B2):
    """
    计算NDVI
    :param B1: 近红波段
    :param B2: 红波段
    :return: NDVI矩阵
    """
    result1 = (B1 - B2) / (B1 + B2)
    return result1

if __name__ == '__main__':
    if len(sys.argv) != 3:
        distutils.log.error("not enougth input parameters")
        sys.exit(-1)
    inpath = sys.argv[1]   #D:\AAdata\GF_radiance\Vege_test\GF1caijian.tif 需掩膜的tif数据
    outpath = sys.argv[2]  #D:\AAdata\GF_radiance 掩膜后的输出路径
    if (os.path.exists(outpath) == False):
        os.makedirs(outpath)
    dataset = Dataset()
    data, proj, geotrans = dataset.read_img(inpath)
    # print(proj)
    # print(geotrans)
    b1 = np.array(data[0],dtype = float)
    b2 = np.array(data[1],dtype = float)
    b3 = np.array(data[2],dtype = float)
    b4 = np.array(data[3],dtype = float)
    print(b1.shape)
    ndwi = NDWI(b2,b4)
    b4 = np.where(b4==-9999,0,b4)

    # plt.imshow(ndwi)
    # plt.axis('off')# 不显示坐标轴
    # plt.show()
    # print(2)

    width,height = b1.shape
    # print(width)
    # print(height)
    # b4.reshape(1,width*height)
    # b4.tolist()
    list = np.sort(b4,axis=0)
    list1 = np.sort(list,axis=1)

    ndwi255 = guiyihua(b4,width,height)
    # plt.imshow(ndwi255)
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()

    img = np.uint8(ndwi255)
    # im = Image.fromarray(np.float64(ndwi))   #将矩阵转为灰度图像
    #自适应分割
    ret1, th1 = cv.threshold(img,0,1, cv.THRESH_OTSU) #ret1为阈值，th1为返回的二值化矩阵。
    print(ret1)
    print(th1)
    os.chdir(outpath)
    #对原影像掩膜,得到水体
    b1water = b1*th1*0.0001
    b2water = b2*th1*0.0001
    b3water = b3*th1*0.0001
    b4water = b4*th1*0.0001
    # b1water = np.where(b1water == 0, 'nan', b1water)
    # b2water = np.where(b2water == 0, 'nan', b2water)
    # b3water = np.where(b3water == 0, 'nan', b3water)
    # b4water = np.where(b4water == 0, 'nan', b4water)
    gf1water = np.array([b1water,b2water,b3water,b4water])
    dataset.write_tiff('water.tif', proj, geotrans, gf1water)
    print('水体提取完成')
    #对掩膜文件求反，得到陆地文件
    b1land = np.where(b1water == 0, b1, 0)
    b2land = np.where(b2water == 0, b2, 0)
    b3land = np.where(b3water == 0, b3, 0)
    b4land = np.where(b4water == 0, b4, 0)
    gf1land = np.array([b1land,b2land,b3land,b4land])
    dataset.write_tiff('land.tif', proj, geotrans, gf1land)
    print('陆地提取完成')







