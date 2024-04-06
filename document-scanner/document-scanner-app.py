# Enter your code here
import cv2
import numpy as np


def annotate():
    global image_cp, pts_src, pts_dest, output

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    #cv2.imshow('test', mask2)  ## FOR DEBUG PURPOSES ONLY
    contours, hierarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        # Handle only contours greater than 10 pixels arc length
        if cv2.arcLength(contour, True) > 10:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the list of destination points is empty, if it is initialize it
            if len(pts_dest) == 0:
                pts_dest = np.array([approx[1][0], approx[2][0], approx[3][0], approx[0][0]], dtype=float)
            pts_src = np.array([approx[1][0], approx[2][0], approx[3][0], approx[0][0]], dtype=float)

            # Counter-clockwise contour coordinates
            topLeftApprox = tuple(approx[1][0])
            bottomLeftApprox = tuple(approx[2][0])
            bottomRightApprox = tuple(approx[3][0])
            topRightApprox = tuple(approx[0][0])

            # Draw the circle's and numbers of the approximated corners
            image_cp = image.copy()
            cv2.circle(image_cp, (topLeftApprox), 30, (255, 0, 255), 3, cv2.LINE_8, 0)
            cv2.putText(image_cp, '0', (topLeftApprox[0] + 5, topLeftApprox[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(image_cp, (bottomLeftApprox), 30, (255, 0, 255), 3, cv2.LINE_8, 0)
            cv2.putText(image_cp, '1', (bottomLeftApprox[0] + 5, bottomLeftApprox[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(image_cp, (bottomRightApprox), 30, (255, 0, 255), 3, cv2.LINE_8, 0)
            cv2.putText(image_cp, '2', (bottomRightApprox[0] + 5, bottomRightApprox[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(image_cp, (topRightApprox), 30, (255, 0, 255), 3, cv2.LINE_8, 0)
            cv2.putText(image_cp, '3', (topRightApprox[0] + 5, topRightApprox[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawContours(image_cp, [approx], -1, (255, 0, 100), 3)

            # Apply Perspective Transformation
            h, status = cv2.findHomography(pts_src, pts_dest)
            image_cp = cv2.warpPerspective(image_cp, h, (image_cp.shape[1], image_cp.shape[0]))
            output = cv2.warpPerspective(image, h, (image_cp.shape[1], image_cp.shape[0]))

            # Provide instructions to the user
            cv2.putText(image_cp, '1. Select the document to extract   2. Drag the circles to correct', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_cp, '3. Press q to Extract and resize to a width of 500 pixels', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Detection', image_cp) # Display the warped image

def select_region():
    global region_selected, image_cp
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_cp, mask, roi, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    region_selected = True # Set the flag to determine that the ROI of the document was selected

def onMouse(event, x, y, flags, param):
    global region_selected, selecting, roi, startPt, modifying_roi, pts_dest, moving_rect

    if event == cv2.EVENT_LBUTTONDOWN and not region_selected:
        if startPt is None:
            startPt = (x,y) # Save the top-left corner of the ROI
            selecting = True

    elif event == cv2.EVENT_LBUTTONUP and not region_selected:
            roi = (min(startPt[0], x), min(startPt[1], y), abs(startPt[0]-x), abs(startPt[1]-y))
            selecting = False
            select_region() # Extract the ROI (i.e. create mask)

    elif event == cv2.EVENT_MOUSEMOVE and selecting and not region_selected:
        image_cp = image.copy()
        cv2.rectangle(image_cp, startPt, (x,y), (255,0,0), 2)
        cv2.imshow('Detection', image_cp)

    elif event == cv2.EVENT_LBUTTONDOWN and region_selected:
        # Determine the corner points of the destination rect
        tl = pts_dest[0]  # Top left corner
        bl = pts_dest[1]  # Bottom left corner
        br = pts_dest[2]  # Bottom right corner
        tr = pts_dest[3]  # Top right corner

        # Define the contours of the rectangle for each corner to click in
        cntsRect_0 = np.array([[tl[:1] - 20, tl[1:2] - 20], [tl[:1] - 20, tl[1:2] + 20], [tl[:1] + 20, tl[1:2] + 20],
                               [tl[:1] + 20, tl[1:2] - 20]], dtype=np.int32)
        cntsRect_1 = np.array([[bl[:1] - 20, bl[1:2] - 20], [bl[:1] - 20, bl[1:2] + 20], [bl[:1] + 20, bl[1:2] + 20],
                               [bl[:1] + 20, bl[1:2] - 20]], dtype=np.int32)
        cntsRect_2 = np.array([[br[:1] - 20, br[1:2] - 20], [br[:1] - 20, br[1:2] + 20], [br[:1] + 20, br[1:2] + 20],
                               [br[:1] + 20, br[1:2] - 20]], dtype=np.int32)
        cntsRect_3 = np.array([[tr[:1] - 20, tr[1:2] - 20], [tr[:1] - 20, tr[1:2] + 20], [tr[:1] + 20, tr[1:2] + 20],
                               [tr[:1] + 20, tr[1:2] - 20]], dtype=np.int32)

        ## FOR DEBUG PURPOSES ONLY
        # rect_1 = list(map(lambda tup: (int(tup[0]), int(tup[1])), cntsRect_1))
        # rect_2 = list(map(lambda tup: (int(tup[0]), int(tup[1])), cntsRect_2))
        # rect_3 = list(map(lambda tup: (int(tup[0]), int(tup[1])), cntsRect_3))
        # rect_4 = list(map(lambda tup: (int(tup[0]), int(tup[1])), cntsRect_4))
        # cv2.rectangle(image_cp, rect_4[0], rect_4[2], (0,255,0), 2)
        # cv2.imshow('Detection', image_cp)

        # Check which corner point was selected
        if cv2.pointPolygonTest(cntsRect_0, (x, y), False) >= 0:
            moving_rect[0] = True
        elif cv2.pointPolygonTest(cntsRect_1, (x, y), False) >= 0:
            moving_rect[1] = True
        elif cv2.pointPolygonTest(cntsRect_2, (x, y), False) >= 0:
            moving_rect[2] = True
        elif cv2.pointPolygonTest(cntsRect_3, (x, y), False) >= 0:
            moving_rect[3] = True

        modifying_roi = True # Determines if a corner point was selected for modification

    elif event == cv2.EVENT_LBUTTONUP and region_selected:
        moving_rect = [False, False, False, False] # Reset all flags in the corners moving list
        modifying_roi = False

    elif event == cv2.EVENT_MOUSEMOVE and modifying_roi:
        # Check which corner point gets modified
        if moving_rect[0]:
            pts_dest[0] = np.asarray((x, y), dtype=np.float32)
        elif moving_rect[1]:
            pts_dest[1] = np.asarray((x, y), dtype=np.float32)
        elif moving_rect[2]:
            pts_dest[2] = np.asarray((x, y), dtype=np.float32)
        elif moving_rect[3]:
            pts_dest[3] = np.asarray((x, y), dtype=np.float32)


if __name__ == "__main__":
    startPt = None
    pts_src = [] # The saved source points
    pts_dest = [] # The saved destination points
    image = cv2.imread("msc_doc.jpg")
    image_cp = image.copy()
    output = image.copy()

    # Provide instructions to the user
    cv2.putText(image_cp, '1. Select the document to extract   2. Drag the circles to correct', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_cp, '3. Press q to Extract and resize to a width of 500 pixels', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2, cv2.LINE_AA)

    roi = np.zeros(image.shape[:2], dtype = np.uint8)
    mask = np.zeros(image.shape[:2], dtype = np.uint8) # mask initialized to BG
    region_selected = False # Determines if the region was selected
    modifying_roi = False # Determines if the user modifies the ROI
    selecting = False
    moving_rect = [False, False, False, False] # Flags that determine the number of the corner that is changed
    aspect_ratio = image.shape[1] / image.shape[0]

    # Initialize the windows
    cv2.namedWindow('Detection')
    cv2.setMouseCallback('Detection', onMouse)
    cv2.imshow('Detection', image_cp)

    while(1):
        key = cv2.waitKey(1)

        if key == 27: # Check if ESC was pressed to exit
            break
        elif key == ord('r'): # Reset the settings
            img_new = image.copy()
            image_cp = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.imshow('Detection', img_new)
            region_selected = False
            selecting = False

        elif key == ord('q'): # Extract the warped image and resize it to 500 px width preserving the aspect ratio
            # Convert the destination points to a list, convert the data type to int32 and create a mask of it
            mask_ = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            points = list(map(lambda tup: (int(tup[0]), int(tup[1])), pts_dest.tolist()))
            points = np.array([points], dtype=np.int32)
            cv2.fillPoly(mask_, points, 255) # Create the mask of the warped image

            image_new = cv2.bitwise_and(output, output, mask=mask_)
            gray = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            warped_image = image_new[y:y + h, x:x + w]
            cv2.imshow('Extracted Document', warped_image)
            cv2.imwrite('warped_image.png', warped_image)

        if region_selected:
            annotate()

    cv2.destroyAllWindows()