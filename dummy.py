import dlib
import cv2
import numpy as np

def landmarks_to_np(landmarks, dtype="int"):

    num = landmarks.num_parts
    
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords
 
def get_centers(img, landmarks):

    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255,0,0), 1) 
    cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)
    scale = desired_dist / dist 
    angle = np.degrees(np.arctan2(dy,dx)) 
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
    
    return aligned_face
def judge_mobile_reflection_in_spectacle(img, left_eye_center, right_eye_center):
    # Assuming left_eye_center and right_eye_center are the centers of the eyes
    # and we can calculate a rough bounding box for the spectacles based on these.
    
    # Calculate the region of interest (ROI) for spectacles
    # This is a simple approach; you may need to adjust it based on your application's accuracy.
    x_min = min(left_eye_center[0], right_eye_center[0])
    x_max = max(left_eye_center[0], right_eye_center[0])
    y_min = min(left_eye_center[1], right_eye_center[1]) - 10  # Slightly above the eye centers
    y_max = max(left_eye_center[1], right_eye_center[1]) + 10  # Slightly below the eye centers
    width = x_max - x_min
    height = y_max - y_min
    
    # Expand the ROI to cover the entire spectacle area
    x_min -= width // 2
    x_max += width // 2
    y_min -= height // 2  # Adjust these values based on the expected spectacle size
    roi = img[y_min:y_max, x_min:x_max]
    
    # Convert to grayscale if not already
    if len(roi.shape) > 2 and roi.shape[2] == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi
    
    # Enhance contrast via histogram equalization
    equalized_roi = cv2.equalizeHist(gray_roi)
    
    # Threshold to identify bright regions
    _, bright_regions = cv2.threshold(equalized_roi, 220, 255, cv2.THRESH_BINARY)
    
    # Calculate the percentage of bright pixels
    bright_pixels = cv2.countNonZero(bright_regions)
    total_pixels = roi.size
    bright_ratio = bright_pixels / total_pixels
    
    # Debug: Visualize the ROI and bright regions
    cv2.imshow("ROI", roi)
    cv2.imshow("Bright Regions", bright_regions)
    
    # Determine the presence of a mobile reflection
    # The threshold might need adjustment based on testing
    return bright_ratio > 0.0455  # Assuming a reflection if > 10% of the ROI is bright


predictor_path = "./data/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    
    for i, rect in enumerate(rects):

        x_face = rect.left()
        y_face = rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face

        cv2.rectangle(img, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,255,0), 2)
        cv2.putText(img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
     
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)

        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)
          # After detecting glasses and aligning the face
        # Assume `landmarks` is obtained from dlib or another facial landmark detector
        left_eye_center, right_eye_center = get_centers(img,landmarks)  # You need to implement this based on your landmarks

# Call the modified function
        reflection_detected = judge_mobile_reflection_in_spectacle(img, left_eye_center, right_eye_center)

        if reflection_detected:
            print("Reflection detected in spectacle")
        else:
            print("No reflection detected")

        
    cv2.imshow("Result", img)
    
    k = cv2.waitKey(5) & 0xFF
    if k==27:   
        break

cap.release()
cv2.destroyAllWindows()
