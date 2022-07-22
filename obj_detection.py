from cv2 import cv2
import numpy as np
class obj_detect():
    def __init__(self,link):
        self.link = link
    def find_shape_obj(self):
        img =cv2.imread(self.link)
        img = cv2.resize(img,(440,320))
        imgG = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        imgblur = cv2.GaussianBlur(imgG,(3,3),2)
        imgCanny =cv2.Canny(imgblur,50,50) #visualization of faces
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)  # the square of the figure
            if area>1000:
                Conter_img = cv2.drawContours(img, cnt, -1, (255, 0, 0),
                                      3)  #  conter on img
                peri = cv2.arcLength(cnt, True)  # lenght of arc
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                objCor = len(approx)
                x, y, w, h = cv2.boundingRect(approx)
                if objCor == 4:
                    name1 = "Square"
                    cv2.rectangle(Conter_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(Conter_img, name1, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (165, 255, 0), 2)
                elif objCor == 3:
                    name2 = "Triangle"
                    cv2.rectangle(Conter_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(Conter_img, name2, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (165, 255, 0), 2)
                else:
                    name3 = "Round"
                    cv2.rectangle(Conter_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(Conter_img, name3, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (165, 255, 0), 2)
        return Conter_img,imgCanny,imgG

    def __del__(self): pass #just clean our code
if __name__ =='__main__':
    #write here link
    link =r"img_video/figure.jpg"
    x = obj_detect(link)
    conter_img,edge_img,blur_img =x.find_shape_obj()
    img_sum = np.hstack((edge_img,blur_img))
    cv2.imshow("all_img",img_sum)
    cv2.imshow("img_final",conter_img)
    cv2.waitKey(0)
