# Enter your code here
import os
import cv2 
import numpy as np
import argparse


def onSelectPatch(action, x, y, flags, userdata):
    # Check for Left Btn Down event
    global bg_changed
    if action == cv2.EVENT_LBUTTONDOWN:
        # Check if the background has changed.
        if not bg_changed:
            global frame; global mean_color; global green_upper; global patch_size; global saved_mean

            # Extract the color of the patch and save it globally.
            patch = frame[x-int(patch_size/2):x+int(patch_size/2), y-int(patch_size/2):y+int(patch_size/2)]
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mean_color = np.mean(patch_hsv, axis=0) # Calculate the mean values of the HSV-patch
            green_upper = np.array(np.mean(mean_color, axis=0), dtype=np.uint8) # Calculate the mean color of the upper green
            saved_mean = green_upper

def onToleranceSlider(*args):
    global green_upper; global green_lower; global saved_mean
    if green_upper is not None:
        green_upper = saved_mean
        tolerance = args[0] * 2.55 # Convert the trackbar value to the range 0-255 the tolerance in percent.
        # Calculate the green lower and upper bound using the tolerance.
        green_lower = np.uint8(np.clip(green_upper - tolerance, 0,255))
        green_upper = np.uint8(np.clip(green_upper + tolerance, 0,255))

def onSoftnessSlider(*args):
    global scaleFactorSoftness; global scaleFactorSoftness
    if args[0] % 2:
        scaleFactorSoftness = args[0]

def onColorCast(*args):
    global scaleFactorColCast
    green_lower[1] = args[0] # Change the S-channel
    green_lower[2] = args[0] # Channel the V-channel
        
def chromaKeying(image):
    # Create a mask based on the saved color.
    global mask; global image_bg; global green_lower; global green_upper; 
    global bg_changed; global scaleFactorSoftness; global scaleFactorColCast

    # Create the mask based on the green color.
    frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, green_lower, green_upper)
    mask_inv = cv2.bitwise_not(mask)

    # Apply morph and smooting operations.
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, element, iterations=5)
    mask_inv = cv2.erode(mask_inv, element, iterations=2)
    mask = cv2.bitwise_not(mask_inv)
    mask_inv = cv2.GaussianBlur(mask_inv, (scaleFactorSoftness,scaleFactorSoftness), 1)

    # Change the background by adding the foreground to the new background
    fg = cv2.bitwise_and(image, image, mask=mask_inv)
    bg = cv2.bitwise_and(image_bg, image_bg, mask=mask)
    fg = np.float32(fg)/255
    final_image = cv2.add(fg, bg)
    bg_changed = True
    return final_image


if __name__ == "__main__":
    video_file = 'greenscreen-asteroid.mp4'
    bgimage_file = 'space.jpg'

    # Create the video capturing device object and save the properties.
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    _, frame = cap.read()
    mask = np.zeros_like(frame[:,:,:1]) # Create the default mask
    
    # Load and resize the background image accordingly
    image_bg = cv2.imread(bgimage_file, cv2.IMREAD_COLOR)
    if image_bg is not None:
        image_bg = cv2.resize(image_bg, (frame_width, frame_height), interpolation= cv2.INTER_LINEAR)
        image_bg = np.float32(image_bg)/255

    # Setup the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", "-l", type=int, default=1, help='Determine if the video should be looped.')
    args = vars(ap.parse_args())

    # Setup the High GUI properties
    windowName = "Chrome Key"
    trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"
    scaleFactorTolerance = 0
    maxTolerance = 100 # in percent
    scaleFactorSoftness = 1
    maxSoftness = 100 # in percent
    scaleFactorColCast = 236
    maxColCast = 236

    bg_changed = False # The flag which determines if the background has changed
    patch_size = 10 # Initialize the patch size

    # First setup the color range to be the maximum (i.e. everything in the frame gets shown)
    green_lower = np.array([30,128,255], dtype=np.uint8)
    green_upper = None
    saved_mean = -1

    # Setup the window properties.
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(windowName, onSelectPatch)
    
    # Create the High GUI sliders.
    cv2.createTrackbar("Tolerance", windowName, scaleFactorTolerance, maxTolerance, onToleranceSlider)
    cv2.setTrackbarMin("Tolerance", windowName, 2) # Set the minimum trackbar value for the tolerance
    cv2.createTrackbar("Softness", windowName, scaleFactorSoftness, maxSoftness, onSoftnessSlider)
    cv2.createTrackbar("Color cast", windowName, scaleFactorColCast, maxColCast, onColorCast)
    cv2.setTrackbarMin("Color cast", windowName, 80) # Set the minimum trackbar value for the color cast

    # Read the frames and apply the desired operations.
    while cap.isOpened():
        ret, frame = cap.read() # Reads the next frame
        print(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if ret == True:
            # Check if the color patch was selected.
            if green_upper is not None:
                frame_changed = chromaKeying(frame)
                cv2.imshow(windowName, frame_changed)
            else:
                cv2.imshow(windowName, frame)
                
        key = cv2.waitKey(25)

        if not ret:
            print("Test")
            start_frame = 20 #int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 4.5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)


        # If ESC key gets pressed.
        if key == 27:
            break

    cap.release()
