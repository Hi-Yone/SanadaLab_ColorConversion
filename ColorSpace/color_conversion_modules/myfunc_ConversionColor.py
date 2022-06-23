# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# RGBとxyz間の変換
class RGB_to_xyz:

        # ====================================
        # 任意のRGBの三刺激値(0~255)からxyz色度座標値を求める。
        # ====================================
        def RGB2xyz(self, R, G, B):
                x, y, z = 0, 0, 0   # 求めるx,y,z

                R = R/255; G = G/255; B = B/255 # 0~1に正規化

                # RGBからCIE1931XYZ表色系への変換行列
                # -> 座標が変換後にXYZ軸に位置すること。白色点が全ての色から等距離(1/3)となること。Yが輝度値となること。
                Xrgb = [2.7689, 1.7517, 1.1302]
                Yrgb = [1.0, 4.5907, 0.0601]
                Zrgb = [0.0, 0.0565, 5.5943]

                to_XYZ_matrix = np.array([Xrgb, Yrgb, Zrgb])    # XYZへの変換行列
                RGB = np.array([R, G, B])                       # 変換元RGB値
                X, Y, Z = np.dot(to_XYZ_matrix, RGB.T)          # 内積をとる
                S = X+Y+Z       # X,Y,Zの総和

                x=X/S; y=Y/S; z=Z/S
                x = np.round(x, 6); y = np.round(y, 6); z = np.round(z, 6)
                return ((x, y, z), (X,Y,Z))     # 戻り値：x,y,zの座標値と輝度値Y

        # ====================================
        # 任意のxyの座標値と輝度値(cd/m^2)からRGB刺激値(0~255)を求める。
        # ====================================
        def xyL2RGB(self, x, y, L):   # 入力するx,yは高精度の値でないといけない。(少数6桁以上)
                R, G, B = 0, 0, 0

                # 三刺激値XYZを求める
                X = (x/y)*L
                Y = L                   # Yは輝度値
                Z = ((1-x-y)/y)*L       # Zはxとyから定まる

                to_RGB_matrix = np.array([[2.7689, 1.7517, 1.1302],     # RGBへの変換行列
                                        [1.0, 4.5907, 0.0601],
                                        [0.0, 0.0565, 5.5943]])
                to_RGB_matrix = np.linalg.inv(to_RGB_matrix)            # 逆行列を求める
                XYZ = np.array([X, Y, Z])                               # 変換元行列
                R, G, B = np.dot(to_RGB_matrix, XYZ.T)                  # 内積をとる
                R = int(R*255); G = int(G*255); B = int(B*255)          # 0~255の範囲に戻す
                return (R, G, B)        # RGBの刺激値

        # ====================================
        # ３次元空間にxyz座標を色分けしてプロットする
        # ====================================
        def xyz_for_plot(self):     # xyzグラフ表示用関数
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=0, azim=45)
                ax.set_title('XYZ color space')
                ax.grid()

                x_arr=[]; y_arr=[]; z_arr=[]

                for i in range(1,256,16):
                        for j in range(1,256,16):
                                for k in range(1, 256, 16):
                                        xyz, XYZ = self.RGB2xyz(k, j, i)
                                        x=xyz[0]; y=xyz[1]; z=xyz[2]; Y=XYZ[1]
                                        x_arr.append(x); y_arr.append(y); z_arr.append(z)
                                        R, G, B = self.xyL2RGB(x, y, Y)
                                        colCode = (R/255, G/255, B/255) # プロット時の色の範囲は0~1
                                        ax.scatter(x, y, z, color=colCode)


# Luvとxyzの間の変換
class Luv_to_RGB:

        # ====================================
        # RGBの三刺激値(0~255)からL*u*v*色度座標値を求める。
        # ====================================
        def RGB2Luv(self, R, G, B):
                inst_RGB_to_xyz = RGB_to_xyz()        # インスタンス生成

                L_star=0; u_star=0; v_star=0            # 求めるL,u,v

                xyz, XYZ = inst_RGB_to_xyz.RGB2xyz(R, G, B)   # 初めにxyz座標値に変換（zは使用しない。）
                x=xyz[0]; y=xyz[1]; z=xyz[2]; Y=XYZ[1]
                # Y, u', v'：試料物体の刺激値及び色度
                ud = (4*x)/(-2*x + 12*y + 3)            # u'
                vd = (9*y)/(-2*x + 12*y + 3)            # v'

                # 光源はD65を使用する。Yn =100に規格化されている。
                # http://yamatyuu.net/other/color/cie1976luv/index.html に完全拡散反射面がなんたらとか、D65だとどうだとかが書いてある。
                Xn = 95.04; Yn = 100; Zn = 108.89            # 完全拡散反射面におけるXYZ刺激値

                udn = (4*Xn)/(Xn + 15*Yn + 3*Zn)        # u'n：完全拡散反射面の色度
                vdn = (9*Yn)/(Xn + 15*Yn + 3*Zn)        # v'n：完全拡散反射面の色度

                # L*のとる範囲は0~100
                YYn = Y/Yn
                # L*がX,Y,Zの値に対して適用可能かを判定
                if YYn >0.008856:
                        modified_YYn = YYn**(1/3)          # 修正
                if YYn <=0.008856:
                        modified_YYn = 7.787*YYn + 16/116  # 修正
                
                L_star = 116* modified_YYn - 16         # L*を求める
                u_star = 13*L_star*(ud - udn)           # u*を求める
                v_star = 13*L_star*(vd - vdn)           # v*を求める

                return (L_star,u_star,v_star)   # 戻り値：L*, u*, v*

        # ====================================
        # L*u*v*色度座標値からRGB(0~255)を求める。
        # ====================================        
        def Luv2RGB(self, L_star, u_star, v_star):
                R=0; G=0; B=0   # 求めるRGB

                # 光源はD65を使用する。Yn =100に規格化されている。
                # http://yamatyuu.net/other/color/cie1976luv/index.html に完全拡散反射面がなんたらとか、D65だとどうだとかが書いてある。
                Xn = 95.04; Yn = 100; Zn = 108.89       # 感染拡散反射面におけるXYZ刺激値

                udn = (4*Xn)/(Xn + 15*Yn + 3*Zn)        # u'n：完全拡散反射面の色度
                vdn = (9*Yn)/(Xn + 15*Yn + 3*Zn)        # v'n：完全拡散反射面の色度

                ud = u_star/(13*L_star) + udn           # ud : 
                vd = v_star/(13*L_star) + vdn           # vd : 
                
                if L_star <= 8:                         # 
                        Y = Yn*L_star*(3/29)**3         # Yの刺激値
                elif L_star > 8:                        # 
                        Y = Yn*((L_star+16)/116)**3     # Yの刺激値

                X = (9*Y*ud)/(4*vd)                     # Xの刺激値
                Z = Y*(12-3*ud-20*vd)/(4*vd)            # Zの刺激値

                to_RGB_matrix = np.array([[2.7689, 1.7517, 1.1302],     # RGBへの変換行列
                                        [1.0, 4.5907, 0.0601],
                                        [0.0, 0.0565, 5.5943]])
                to_RGB_matrix = np.linalg.inv(to_RGB_matrix)            # 逆行列を求める
                XYZ = np.array([X, Y, Z])                               # 変換元行列
                R, G, B = np.dot(to_RGB_matrix, XYZ.T)                  # 内積をとる
                R = int(R*255); G = int(G*255); B = int(B*255)          # 0~255の範囲に戻す

                return (R, G, B)        # RGBの刺激値

        # ====================================
        # ３次元空間にL*u*v*座標を色分けしてプロットする
        # ====================================
        def Luv_for_plot(self):
                inst_RGB_to_xyz = RGB_to_xyz()
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=0, azim=45)
                ax.set_title('L*u*v*  color space')
                ax.grid()

                L_star_arr=[]; u_star_arr=[]; v_star_arr=[]

                for i in range(1,256,16):
                        for j in range(1,256,16):
                                for k in range(1, 256, 16):
                                        L_star, u_star, v_star = self.RGB2Luv(k, j, i)
                                        L_star_arr.append(L_star); u_star_arr.append(u_star); v_star_arr.append(v_star)
                                        xyz, XYZ= inst_RGB_to_xyz.RGB2xyz(k, j, i)
                                        x=xyz[0]; y=xyz[1]; Y=XYZ[1]
                                        R, G, B = inst_RGB_to_xyz.xyL2RGB(x, y, Y)     # プロットのカラー指定用にRGBを求める

                                        colCode = (R/255, G/255, B/255) # プロット時の色の範囲は0~1
                                        colCode = (R/255, G/255, B/255) # プロット時の色の範囲は0~1
                                        ax.scatter(L_star, u_star, v_star, color=colCode)


class Lab_to_RGB:

        # ====================================
        # RGBの三刺激値(0~255)からL*a*b*色度座標値を求める。
        # ====================================
        def RGB2Lab(self, R, G, B):
                inst_RGB_to_xyz = RGB_to_xyz()        # インスタンス生成

                L_star=0; a_star=0; b_star=0;           # 求めるLab

                xyz, XYZ = inst_RGB_to_xyz.RGB2xyz(R, G, B)   # 初めにxyz座標値に変換（zは使用しない。）
                x=xyz[0]; y=xyz[1]; L=XYZ[1]
                X = (x/y)*L
                Y = L                   # Yは輝度値
                Z = ((1-x-y)/y)*L       # Zはxとyから定まる

                # 光源はD65を使用する。Yn =100に規格化されている。
                # http://yamatyuu.net/other/color/cie1976luv/index.html に完全拡散反射面がなんたらとか、D65だとどうだとかが書いてある。
                Xn = 95.04; Yn = 100; Zn = 108.89            # 完全拡散反射面におけるXYZ刺激値

                # L*のとる範囲は0~100
                YYn = Y/Yn
                # L*がX,Y,Zの値に対して適用可能かを判定
                if YYn >0.008856:
                        modified_YYn = YYn**(1/3)          # 修正
                if YYn <=0.008856:
                        modified_YYn = 7.787*YYn + 16/116  # 修正
                
                XXn = X/Xn
                if XXn >0.008856:
                        modified_XXn = XXn**(1/3)          # 修正
                if XXn <=0.008856:
                        modified_XXn = 7.787*XXn + 16/116  # 修正

                ZZn = Z/Zn
                if ZZn >0.008856:
                        modified_ZZn = ZZn**(1/3)          # 修正
                if ZZn <=0.008856:
                        modified_ZZn = 7.787*ZZn + 16/116  # 修正

                L_star = 116* modified_YYn - 16                 # L*を求める
                a_star = 500*(modified_XXn - modified_YYn)      # a*を求める
                b_star = 200*(modified_YYn - modified_ZZn)      # b*を求める

                return (L_star, a_star, b_star)         # 戻り値：L*, a*, b*


        # ====================================
        # L*u*v*色度座標値からRGB(0~255)を求める。
        # ====================================   
        def Lab2RGB(self, L_star, a_star, b_star):
                R=0; G=0; B=0           # 求めるRGB値
                return (R, G, B)        # 戻り値：R,G,B


        # ====================================
        # ３次元空間にL*a*b*座標を色分けしてプロットする
        # ====================================
        def Lab_for_plot(self):
                inst_RGB_to_xyz = RGB_to_xyz()
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.view_init(elev=0, azim=45)
                ax.set_title('L*a*b*  color space')
                ax.grid()

                L_star_arr=[]; a_star_arr=[]; b_star_arr=[]

                for i in range(1,256,16):
                        for j in range(1,256,16):
                                for k in range(1, 256, 16):
                                        L_star, a_star, b_star = self.RGB2Lab(k, j, i)
                                        L_star_arr.append(L_star); a_star_arr.append(a_star); b_star_arr.append(b_star)
                                        xyz, XYZ= inst_RGB_to_xyz.RGB2xyz(k, j, i)
                                        x=xyz[0]; y=xyz[1]; Y=XYZ[1]
                                        R, G, B = inst_RGB_to_xyz.xyL2RGB(x, y, Y)     # プロットのカラー指定用にRGBを求める

                                        colCode = (R/255, G/255, B/255) # プロット時の色の範囲は0~1
                                        colCode = (R/255, G/255, B/255) # プロット時の色の範囲は0~1
                                        ax.scatter(L_star, a_star, b_star, color=colCode)



class lms_to_RGB:
        def RGB2lms(self, R, G, B):
                l=0; m=0; s=0           # 求めるlms


                return 0




#%%

inst = RGB_to_xyz()
inst.xyz_for_plot()


# inst = Luv_to_RGB()
# l ,u, v= inst.RGB2Luv(0, 0, 255)
# print(l, u, v)
# print(inst.Luv2RGB(l, u, v))

#%%

        # RGB->xyzの変換について、、

        # CIE表色系の変換行列
        # # Xrgb=(2.7689, 1.7517, 1.1302)
        # # Yrgb=(1, 4.5907, 0.0601)
        # # Zrgb=(0, 0.0565, 5.5943)

        # # 原刺激XYZのrgb系における色度座標（教科書のものを使用）
        # # Xrgb = (1.2750, -0.2778, 0.0028)
        # # Yrgb = (-1.7392, 2.761, -0.0279)
        # # Zrgb = (-0.7431, 0.1409, 1.6022)

        # 同じにならなかった。CIE_RGB（白色点E）      英語版wikiも見たらこれがCIE-standard
        # # X = 0.4898*R + 0.3101*G + 0.2001*B
        # # Y = 0.1769*R + 0.8124*G + 0.0107*B
        # # Z = 0.0000*R + 0.0100*G + 0.9903*B

        # sRGB（白色点 D65 (0.333, 0.333)）
        # # X = 0.4224*R 0.3576*G 0.1805*B
        # # Y = 0.2326*R 0.7152*G 0.0722*B
        # # Z = 0.0193*R 0.1192*G 0.9505*B

        # sRGB (白色点 C)
        # # X = 0.5778*R 0.1825*G 0.1902*B
        # # Y = 0.3070*R 0.6170*G 0.0761*B
        # # Z = 0.0181*R 0.0695*G 1.0015*B



'''
# 原刺激の座標値から算出する方法 -> 使用しなくてよさそう。（変換行列があれば十分。）
def rgb2xyz(R, G, B):
        # ====================================
        # 任意のRGBの三刺激値(0~255)からxyz色度座標値を求める。
        # ====================================
        x, y, z = 0, 0, 0   # 求めるx,y,z

        R = R/255; G = G/255; B = B/255 # 0~1に正規化

        # 原刺激XYZのrgb系における色度座標の初期化
        Xr, Xg, Xb = (0, 0, 0)
        Yr, Yg, Yb = (0, 0, 0)
        Zr, Zg, Zb = (0, 0, 0)

        # 原刺激XYZのrgb系における色度座標（教科書のものを使用）
        Xrgb = (1.2750, -0.2778, 0.0028)
        Yrgb = (-1.7392, 2.761, -0.0279)
        Zrgb = (-0.7431, 0.1409, 1.6022)

        # 色光CがX,Y,Zのいずれかの軸に一致している場合 + 色光Cが基礎刺激である場合を考える
        # Xを求める(係数行列)
        valX = [[Yrgb[0], Yrgb[1], Yrgb[2]],\
                [Zrgb[0], Zrgb[1], Zrgb[2]],\
                [1, 1, 1]]
        ansX = [0, 0, 1]

        # # Yを求める(係数行列)
        valY = [[Xrgb[0], Xrgb[1], Xrgb[2]],\
                [Zrgb[0], Zrgb[1], Zrgb[2]],\
                [1, 1, 1]]
        ansY = [0, 0, 1]

        # # Zを求める(係数行列)
        valZ = [[Xrgb[0], Xrgb[1], Xrgb[2]],\
                [Yrgb[0], Yrgb[1], Yrgb[2]],\
                [1, 1, 1]]
        ansZ = [0, 0, 1]

        # # 連立方程式を解いて、XYZそれぞれの係数を算出する
        Xr, Xg, Xb = np.round(solve(valX, ansX), 5)
        Yr, Yg, Yb = np.round(solve(valY, ansY), 5)
        Zr, Zg, Zb = np.round(solve(valZ, ansZ), 5)

        # # 色光Cの三刺激値RGBが与えられた時のXYZの刺激値
        X = (Xr*R + Xg*G + Xb*B)
        Y = (Yr*R + Yg*G + Yb*B)
        Z = (Zr*R + Zg*G + Zb*B)

        # X,Y,Zの総和
        S = X+Y+Z

        x=X/S; y=Y/S; z=Z/S

        return x, y, z, Y
'''

# %%
