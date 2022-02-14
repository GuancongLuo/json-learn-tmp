
from email.mime import image
import cv2
img= cv2.imread("000000.png")          #定义图片位置
print(img)
img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #转化为灰度图 因为我读取的直接是灰度标签图片就不用转化了
print(img)

def onmouse(event, x, y, flags, param):   #标准鼠标交互函数
  if event==cv2.EVENT_MOUSEMOVE:      #当鼠标移动时
    print(img[y,x])           #显示鼠标所在像素的数值，注意像素表示方法和坐标位置的不同
def main():
  cv2.namedWindow("img")          #构建窗口
  cv2.setMouseCallback("img", onmouse)   #回调绑定窗口
  while True:               #无限循环
    cv2.imshow("img",img)        #显示图像
    if cv2.waitKey() == ord('q'):break  #按下‘q'键，退出
  cv2.destroyAllWindows()         #关闭窗口
if __name__ == '__main__':          #运行
  main()