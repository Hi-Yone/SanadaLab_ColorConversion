#%%
# カラーパッチ作成するだけ
from cv2 import imwrite
import numpy as np
import matplotlib.pyplot as plt

def makeImages(ch1, ch2, ch3):
# ===============================
# カラーパッチを作成
# ===============================
    arr = np.array([[[ch1,ch2,ch3] for __ in range(512)] for __ in range(512)])
    return arr

def saveImages():
# ===============================
# カラーパッチを保存
# ===============================
    for i in range(len(color)):
        print(color[i])
        img = makeImages(*color[i])
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        imwrite('./colorPatch/color' + str(i) + '.png', img)
    
plt.figure
ch1=0; ch2=0; ch3=0
color=np.array([(0, 0, 0), (128, 128, 128), (255, 255, 255), 
                        (255, 0, 0), (0, 255, 0), (0, 0, 255),
                        (255, 255, 0), (255, 0, 255), (0, 255, 255)])
saveImages()

# %%
