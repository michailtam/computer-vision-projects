# Enter your code here
import cv2 
import numpy as np


def drawPatches(x, y, r, color=(0,255,0)):
    ''' Draws the patches into the image. '''
    top_leftx = x-int(r/2) 
    top_lefty = y-int(r/2)
    img_anot = output.copy()

    # Coordinates of each patch.
    coords = [
        (top_leftx, top_lefty, top_leftx+r, top_lefty+r),
        (top_leftx, top_lefty-r, top_leftx+r, top_lefty),
        (top_leftx+r, top_lefty-r, top_leftx+(2*r), top_lefty),
        (top_leftx+r, top_lefty, top_leftx+(2*r), top_lefty+r),
        (top_leftx+r, top_lefty+r, top_leftx+(2*r), top_lefty+(2*r)),
        (top_leftx, top_lefty+r, top_leftx+r, top_lefty+(2*r)),
        (top_leftx-r, top_lefty+r, top_leftx, top_lefty+(2*r)),
        (top_leftx-r, top_lefty, top_leftx, top_lefty+r),
        (top_leftx-r, top_lefty-r, top_leftx, top_lefty)
    ]
    for (tlx, tly, tbx, tby) in coords:
        cv2.rectangle(img_anot, (tlx, tly), (tbx, tby), color=color, thickness=1)
    return img_anot

def getNearbyPatches(x, y, r):
    ''' Retrieves all the 8 nearby patches. '''
    nearby_patches = {}
    top_leftx = x-int(r/2) # The top left-x coordinate of the selected blemish patch
    top_lefty = y-int(r/2) # The top left-y coordinate of the selected blemish patch

    # Extract the eight nearby patches.
    nearby_patches['p1'] = output[top_lefty-r:top_lefty, top_leftx:top_leftx+r] # Top
    nearby_patches['p2'] = output[top_lefty-r:top_lefty, top_leftx+r:top_leftx+(2*r)] # Top-right
    nearby_patches['p3'] = output[top_lefty:top_lefty+r, top_leftx+r:top_leftx+(2*r)] # Right
    nearby_patches['p4'] = output[top_lefty+r:top_lefty+(2*r), top_leftx+r:top_leftx+(2*r)] # Bottom-right
    nearby_patches['p5'] = output[top_lefty+r:top_lefty+(2*r), top_leftx:top_leftx+r] # Bottom
    nearby_patches['p6'] = output[top_lefty+r:top_lefty+(2*r), top_leftx-r:top_leftx] # Bottom-left
    nearby_patches['p7'] = output[top_lefty:top_lefty+r, top_leftx-r:top_leftx] # Left
    nearby_patches['p8'] = output[top_lefty-r:top_lefty, top_leftx-r:top_leftx] # Top-left
    return nearby_patches

def calcGradientsMean(patch):
    ''' Calculates the mean of the gradients in x and y direction. '''
    patch_grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Calculate the gradients in x-direction.
    sobel_x = cv2.Sobel(patch_grey, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobel_x)
    sobel_x = np.uint8(abs_sobelx)
    gradient_x = np.mean(sobel_x)

    # Calculate the gradients in y-direction.
    sobel_y = cv2.Sobel(patch_grey, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobely = np.absolute(sobel_y)
    sobel_y = np.uint8(abs_sobely)
    gradient_y = np.mean(sobel_y)
    
    return (gradient_x, gradient_y)

def findBestPatch(patches):
    ''' Determines the best matching patch to replace the blemish. '''
    gradient_means = {k: calcGradientsMean(p) for k,p in patches.items()} # Calculate the mean of the gradients
    key_min_x = min(gradient_means.keys(), key=(lambda k: gradient_means[k][0])) # Calculate the min gradient mean in x direction
    key_min_y = min(gradient_means.keys(), key=(lambda k: gradient_means[k][1])) # Calculate the min gradient mean in y direction
    
    # Check if the mean of the x and y gradient matches. If this is the case, return the respective patch.
    if key_min_x == key_min_y:
        return patches[key_min_x]
    else:
        return None
    
def removeBlemish(patch, pos):
    ''' Removes the blemish by applying seamless cloning. '''
    mask = 255 * np.ones(patch.shape, patch.dtype)
    return cv2.seamlessClone(patch, output, mask, pos, cv2.NORMAL_CLONE)
    
# Callback function
def on_mouse(action, x, y, flags, userdata):
    if action == cv2.EVENT_LBUTTONDOWN:
        global output
        nearby_patches = getNearbyPatches(x, y, radius) # Get the 8 nearby patches be the radius
        best_patch = findBestPatch(nearby_patches) # Finds the best matching nearby patch
        if best_patch is not None:
            output = removeBlemish(best_patch, pos=(x, y))
        #img_anotated = drawPatches(x,y,radius) #NOTE: For test purposes
        #cv2.imshow("Blemish Removal", img_anotated) #NOTE: For test purposes

    elif action == cv2.EVENT_LBUTTONUP:
        cv2.imshow("Blemish Removal", output)

if __name__ == "__main__":
    image_path = 'blemish.png'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    global radius; radius = 20
    output = image.copy() # Create the output image which gets modified every time a blemish gets removed
    print('The radius of the patch used is:', radius, 'pixels.')

    cv2.imshow("Blemish Removal", image)
    cv2.setMouseCallback("Blemish Removal", on_mouse) # Register the callback function for the mouse events
    key = 0
    while key != 27: # Wait until the ESC key gets pressed
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()