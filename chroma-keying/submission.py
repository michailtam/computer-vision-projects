# Enter your code here
import os
import cv2 
import numpy as np


### TEST
def onBlur(*args):
    global scaleBlur; global frame
    frame = cv2.GaussianBlur(frame, (args[0],args[0]), 0,0)
    #cv2.imshow(windowName, frame)
    #print((args[0],args[0]))
### END TEST

def onSelectPatch(action, x, y, flags, userdata):
    if action == cv2.EVENT_LBUTTONDOWN:
        global patch_size; global frame; global bgcolor; global image_bg
        center_patch_x = int(patch_size/2)
        center_patch_y = int(patch_size/2)

        patch = frame[x-center_patch_x:x, y-center_patch_y:y]
        bgcolor = patch[int(patch_size/2)-1, int(patch_size/2)-1] # Select the color at the center of the patch
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        green_low = np.array(patch_hsv[0,0], dtype=np.uint8)
        green_high = np.array([70,255,255], dtype=np.uint8)
        
        # Create a mask
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_hsv = cv2.GaussianBlur(frame_hsv, (21,21), 0)
        mask = cv2.inRange(frame_hsv, green_low, green_high)
        
        # Smooth the mask to remove the bordered extensions.
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element, iterations=5)
        mask = cv2.dilate(mask, element, iterations=7)
        mask = cv2.bitwise_not(mask)
        
        # Apply mathematical operations to achieve the result.
        mask_3d = cv2.merge((mask, mask, mask))
        framefl32 = np.float32(frame)/255
        maskf32 = np.float32(mask_3d)/255
        
        final = cv2.multiply(image_bg, (1 - maskf32))
        #final = cv2.add(framefl32, final)
        cv2.imshow('test', mask)
        cv2.imshow('test2', final)
        

def onChangeTolerance(*args):
    global scaleFactor
    scaleFactor = args[0]


if __name__ == "__main__":
    
    file_path = 'CVDL Master Program/Projects/Submissions/Chroma Keying'
    video_file = os.path.join(file_path, 'greenscreen-asteroid.mp4')
    bgimage_file = os.path.join(file_path, 'space.jpg')

    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    _, frame = cap.read()
    
    # Load and resize the background image accordingly
    image_bg = cv2.imread(bgimage_file, cv2.IMREAD_COLOR) 
    if image_bg is not None:
        image_bg = cv2.resize(image_bg, (frame_width, frame_height), interpolation= cv2.INTER_LINEAR)
        image_bg = np.float32(image_bg)/255
    
    windowName = "Chrome Key"
    trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"
    scaleFactor = 1
    maxTolerance = 255
    patch_size = 4

    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(windowName, onSelectPatch)

    bgcolor = None
    
    # Create sliders
    cv2.createTrackbar("Tolerance", windowName, scaleFactor, maxTolerance, onChangeTolerance)
    #cv2.createTrackbar("Softness", windowName, scaleFactor, maxTolerance, onChangeTolerance)
    #cv2.createTrackbar("Color cast", windowName, scaleFactor, maxTolerance, onChangeTolerance)
    #cv2.createTrackbar("Blur", windowName, 1, 25, onBlur) ## TEST

    while cap.isOpened():
        ret, frame = cap.read() # Reads the next frame

        if ret == True:
            if bgcolor is None:
                cv2.imshow(windowName, frame)
            else:
                #frame[np.where(frame == bgcolor)] = 0
                cv2.imshow(windowName, frame)

        key = cv2.waitKey(25)

        # If ESC key gets pressed.
        if key == 27:
            break

    cap.release()
