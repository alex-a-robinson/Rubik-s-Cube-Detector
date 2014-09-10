#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import cv2
import math
from collections import Counter
from copy import deepcopy

cap = cv2.VideoCapture(0)

BLACK_MIN = np.array([0,0,0],np.uint8)
BLACK_MAX = np.array([255,255,50],np.uint8)

WHITE_MIN = np.array([0,0,50],np.uint8)
WHITE_MAX = np.array([255,55,255],np.uint8)

YELLOW_MIN = np.array([20,50,50],np.uint8)
YELLOW_MAX = np.array([50,255,255],np.uint8)

GREEN_MIN = np.array([50,50,50],np.uint8)
GREEN_MAX = np.array([100,255,255],np.uint8)
    
BLUE_MIN = np.array([100,50,50],np.uint8)
BLUE_MAX = np.array([120,255,255],np.uint8)

ORANGE_MIN = np.array([5, 50, 50],np.uint8)
ORANGE_MAX = np.array([20, 255, 255],np.uint8)

RED_MIN = np.array([150, 50, 50],np.uint8)
RED_MAX = np.array([180, 255, 255],np.uint8)

RED_MIN_HIGH = np.array([0, 50, 50],np.uint8)
RED_MAX_HIGH = np.array([5, 255, 255],np.uint8)

surf = cv2.SURF(75000)
surf.upright = True
surf.extended = True

def drawGridLine(g):
    try:
        x1, y1 = int(g[0].pt[0]), int(g[0].pt[1])
        x2, y2 = int(g[-1].pt[0]), int(g[-1].pt[1])
        cv2.line(frame, (x1, y1), (x2, y2), (0,0, 255))
    except:
        pass

def findColour(pt, size=20):
    x, y = int(pt.pt[0]), int(pt.pt[1])
    if not np.all(white_mask[y-size/2:y+size/2, x-size/2:x+size/2] == 0):
        return 'white'
    elif not np.all(yellow_mask[y-size/2:y+size/2, x-size/2:x+size/2] == 0):
        return 'yellow'
    elif not np.all(green_mask[y-size/2:y+size/2, x-size/2:x+size/2] == 0):
        return 'green'
    elif not np.all(blue_mask[y-size/2:y+size/2, x-size/2:x+size/2] == 0):
        return 'blue'
    elif not np.all(orange_mask[y-size/2:y+size/2, x-size/2:x+size/2] == 0):
        return 'orange'
    elif not np.all(red_mask[y-size/2:y+size/2, x-size/2:x+size/2] == 0):
        return 'red'
    else:
        return 'unknown'
            
def drawColours(face, size=20, border=2):
    for point in face:
        colour = findColour(point)
        if colour != 'unknown':
            x, y = int(point.pt[0]), int(point.pt[1])
            if border > 0:
                cv2.circle(frame, (x, y), size + border, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), size, COLOURS[colour], -1)


COLOURS = {'white'  : (255, 255, 255),
           'yellow' : (0, 255, 255),
           'green'  : (0, 255, 0),
           'blue'   : (255, 0, 0),
           'orange' : (0, 127, 255),
           'red'    : (0, 0, 255),
           'unknown': (200, 200, 200)}

def drawFlatCube(size = 10, offsetX = 10, offsetY = 10):
    for row in range(0, size * 12, size):
        for col in range(size * 3, size * 3 + size * 3, size):
            cv2.rectangle(frame, (col + offsetY, row + offsetX), (col + offsetY + size, row + offsetX + size), (200, 200, 200), 1)
    
    for row in range(size * 3, size * 3 + size * 3, size):
        for col in range(0, size * 9, size):
            cv2.rectangle(frame, (col + offsetY, row + offsetX), (col + offsetY + size, row + offsetX + size), (200, 200, 200), 1)
    

def storeColours(rows, face, copy=False):
    colours = []
    for row in rows:
        for point in row:
            colour = findColour(point)
            colours.append(colour)
            
    if copy:
        c = deepcopy(completeCube)
        c[face] = colours
        return c
    else:
    
        completeCube[face] = colours
            
def addColourToCube(cube): #TODO : Check ordering
    for i, face in enumerate(cube):
        size = 10
        if i == 0:
            xStart = 10 + 3 * size
            y = 10
        elif i == 1:
            xStart = 10
            y = 10 + 3 * size
        elif i == 2:
            xStart = 10 + 3 * size
            y = 10 + 3 * size
        elif i == 3:
            xStart = 10 + 6 * size
            y = 10 + 3 * size
        elif i == 4:
            xStart = 10 + 3 * size
            y = 10 + 6 * size
        elif i == 5:
            xStart = 10 + 3 * size
            y = 10 + 9 * size
        
        for j, col in enumerate(face):
            if j % 3 == 0:
                x = xStart
            cv2.rectangle(frame, (x + 1, y + 1), (x + 8, y + 8), COLOURS[col], -1)
            x += size
            if j != 0 and j % 3 == 0:
                y += size
                

'''
def addColourToCube(rows, face):
    size = 10
    if face == 1:
        xStart = 10 + 3 * size
        y = 10
    elif face == 2:
        xStart = 10
        y = 10 + 3 * size
    elif face == 3:
        xStart = 10 + 3 * size
        y = 10 + 3 * size
    elif face == 4:
        xStart = 10 + 6 * size
        y = 10 + 3 * size
    elif face == 5:
        xStart = 10 + 3 * size
        y = 10 + 6 * size
    elif face == 6:
        xStart = 10 + 3 * size
        y = 10 + 9 * size
        
    for row in rows:
        x = xStart            
        for point in row:
            colour = findColour(point)
            if colour != 'unknown':
                cv2.rectangle(frame, (x + 1, y + 1), (x + 8, y + 8), COLOURS[colour], -1)
            x += size
        y += size
'''
completeCube = [[], [], [], [], [], []]
FACE_NUM = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
    black_mask = cv2.inRange(hsv_img, BLACK_MIN, BLACK_MAX)
    
    contours,hier = cv2.findContours(black_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(frame.shape,np.uint8)
    
    for cnt in contours:
        if cv2.contourArea(cnt)>15000:  # remove small areas like noise etc
            hull = cv2.convexHull(cnt)    # find the convex hull of contour
            hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
            
            if len(hull)==4:
                x,y,w,h = cv2.boundingRect(cnt)
                mask[y:y+h,x:x+w] = frame[y:y+h,x:x+w]
                
    hsv_img = cv2.cvtColor(mask,cv2.COLOR_BGR2HSV)
        
    white_mask = cv2.inRange(hsv_img, WHITE_MIN, WHITE_MAX)
    yellow_mask = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
    green_mask = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)
    blue_mask = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX)
    orange_mask = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
    red_mask = cv2.inRange(hsv_img, RED_MIN, RED_MAX)
    red_mask_high = cv2.inRange(hsv_img, RED_MIN_HIGH, RED_MAX_HIGH)
    
    red_mask = red_mask | red_mask_high
    
    kernel = np.ones((25,25),np.uint8)
    white_mask = cv2.erode(white_mask, kernel)
    yellow_mask = cv2.erode(yellow_mask, kernel)
    green_mask = cv2.erode(green_mask, kernel)
    blue_mask = cv2.erode(blue_mask, kernel)
    orange_mask = cv2.erode(orange_mask, kernel)
    red_mask = cv2.erode(red_mask, kernel)
    
    mask = white_mask | yellow_mask | green_mask | blue_mask | orange_mask | red_mask
    
    kp, des = surf.detectAndCompute(mask,None)
    
    face = []
    x_list = {}
    y_list = {}
    
    for i, point in enumerate(kp):
        if i > 8: # Only draw the first 9 points
            break
        x, y = int(point.pt[0]), int(point.pt[1])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        x_list.update({x : point})
        y_list.update({y : point})
        face.append(point)
    
    completeCubeCopy = deepcopy(completeCube) 
        
    if len(face) == 9:
        y_list_sorted = sorted(y_list)
        # First row
        top_row = []
        for y in y_list_sorted[0:3]:
            top_row.append(y_list[y])
        top_row.reverse()
        # Middle row
        mid_row = []
        for y in y_list_sorted[3:6]:
            mid_row.append(y_list[y])
        mid_row.reverse()
        # Bottom row
        bot_row = []
        for y in y_list_sorted[6:9]:
            bot_row.append(y_list[y])
        bot_row.reverse()
        
        drawGridLine(top_row)
        drawGridLine(mid_row)
        drawGridLine(bot_row)
        
        x_list_sorted = sorted(x_list)
        # First col
        top_col = []
        for x in x_list_sorted[0:3]:
            top_col.append(x_list[x])
        # Middle col
        mid_col = []
        for x in x_list_sorted[3:6]:
            mid_col.append(x_list[x])
        # Bottom col
        bot_col = []
        for x in x_list_sorted[6:9]:
            bot_col.append(x_list[x])
        
        drawGridLine(top_col)
        drawGridLine(mid_col)
        drawGridLine(bot_col)
        
        completeCubeCopy = storeColours([top_row, mid_row, bot_row], FACE_NUM, True)
        
        
        drawColours(face)
    
    drawFlatCube()
    addColourToCube(completeCubeCopy)
        
    cv2.imshow('win',  frame)
    
    if len(face) == 9 and cv2.waitKey(1) & 0xFF == 32:
        storeColours([top_row, mid_row, bot_row], FACE_NUM)
        FACE_NUM += 1
        print(FACE_NUM)
        if FACE_NUM == 6:
            print('DONE!')
            break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()