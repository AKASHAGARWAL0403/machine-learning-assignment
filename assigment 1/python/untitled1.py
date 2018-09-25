# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:17:39 2018

@author: lenovo
"""

import cv2
import numpy as np

windows="window"
img=np.zeros((512,512,3),np.uint8)
cv2.namedWindow(windows)

mode=True
(ix,iy)=(-1,-1)

def draw_shape(event, x, y, flag, param):
    global drawing,ix,iy,mode
    
    if event ==cv2.EVENT_LBUTTONDOWN:
        (ix,iy)=x,y
        drawing=True
        
    elif event ==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(200,100,150),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)
    
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.rectangle(img,(ix,iy),(x,y),(200,100,150),-1)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)
        
        
cv2.setMouseCallback(windows,draw_shape)
    
def main():
    global mode
    while(True):
        cv2.imshow(windows,img)
        k=cv2.waitKey(5)
        if k==ord('m') or k==ord('M'):
            mode=not mode
        elif k== 27:
            break
        
            
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()
        