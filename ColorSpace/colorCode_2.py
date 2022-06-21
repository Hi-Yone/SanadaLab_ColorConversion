#%%
import numpy as np

col = np.array([-0.1606398, -0.08090146, 0.1367964])

'''TS_dkl2lms; DKL空間からlms値に変換する関数
    USAGE:
        stdlms = np.array([13, 6, 1])   #白色点におけるLMS。今回は20cd/m^2 のEEW; Yxy = [20, 1/3, 1/3]
        thr = np.array([0.6593, 0.07845, 0.984725]) #DKL空間の正規化係数。CF実験の4名の閾値の平均を使用
        
        dkl = [90, 0, 1]    #[elevation(deg), azimuth(deg),radius]
        hoge = TS_dkl2lms(dkl, stdlms, thr)
        print(hoge)
'''

# 色空間パラメータ
stdlms = np.array([ [13.35686225], [6.703407806], [0.1359502087] ])   #白色点におけるLMS。今回は20cd/m^2 のEEW; Yxy = [20, 1/3, 1/3]
thr = np.array([0.6593, 0.07845, 0.984725]) #DKL空間の正規化係数。CF実験の4名の閾値の平均を使用

#print(stdlms)   #debug
def TS_dkl2lms(dkl, stdlms, thr):
    # 確実にndarrayにしておく。
    dkl = np.asarray(dkl)
    stdlms = np.asarray(stdlms)
    thr = np.asarray(thr)
    # 極座標 > 直交座標
    RG = dkl[2]*np.sin(dkl[0])*np.cos(dkl[1])
    BY = dkl[2]*np.sin(dkl[0])*np.sin(dkl[1])
    LUM = dkl[2]*np.cos(dkl[0])
    DKLcart = np.array([LUM, RG, BY])
    #print(DKLcart)  #debug
    
    # dkl2lms; 見やすくするためにl,m,s別々にやる。
    l = (stdlms[0]*(DKLcart[0]*thr[0]) + stdlms[1]*(DKLcart[1]*thr[1]) ) / (stdlms[0]+stdlms[1])
    m = (stdlms[1]*(DKLcart[0]*thr[0]) - stdlms[1]*(DKLcart[1]*thr[1]) ) / (stdlms[0]+stdlms[1])
    s = (stdlms[2]*(DKLcart[0]*thr[0]) + stdlms[2]*(DKLcart[2]*thr[2]) ) / (stdlms[0]+stdlms[1])
    # ndarray に結合
    lmsc = np.array([l, m, s])
    #print(lmsc) #debug
    
    # 白色点のlmsを加算
    lms = lmsc + stdlms
    return lms
    
'''
TS_lms2rgb(lms, lmsk, ccmat)
'''
# 色変換行列; RGB phosphar と 錐体応答 Smith&Pokorny (1975) から計算
ccmat_lms2rgb = np.array([[0.04541331542, -0.07268940631, 0.1271141625],
                          [-0.007575273415, 0.04332873524, -0.1503869405],
                          [0.0007290446725, -0.005912107729, 2.225003631]])
# K に対する錐体応答
lmsk = np.array([ [0.113834211], [0.06260095507], [0.001835484215] ])

def TS_lms2rgb(lms, lmsk, ccmat):
    lms = np.asarray(lms)
    lmsk = np.asarray(lmsk)
    ccmat = np.asarray(ccmat)
    #
    lmsd = lms - lmsk
    rgb = np.dot(ccmat, lmsd)
    
    return rgb
    
'''TS_rgb2RGB(rgb, gamma)
    線形変換した rgb にガンマ補正をかける。
'''
def TS_rgb2RGB(rgb, gamma):
    #print(rgb)
    res = np.power(rgb, 1/gamma)
    #print(res)
    # 転置して return
    return res.T

'''
PsychoPyのRGBは -1～+1、NormalizedRGB は 0～+1
なので変換する必要あり。めんどくさい。
'''
def TS_NormRGB2PsychopyRGB(normRGB):
    return 2*(normRGB-0.5)


azimuth = 180
dkl = np.array([0, azimuth, 15])                    # [elevation, azimath, radius]
lms = TS_dkl2lms(dkl, stdlms, thr)                  # dkl2lms
rgb = TS_lms2rgb(lms, lmsk, ccmat_lms2rgb)          # lms2rgb
RGB = TS_rgb2RGB(rgb, 2.3)                          # rgb2RGB
RGB = TS_NormRGB2PsychopyRGB(RGB)                   # RGB2PsychoPyRGB

col = RGB

print(*col)
# %%
