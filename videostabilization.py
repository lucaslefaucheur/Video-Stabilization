import numpy as np
import cv2
import sys
import matplotlib.pyplot as mat

def findMatchesBetweenImages(image_1, image_2, num_matches):
    """Returns the top list of matches between two input images.

    Parameters
    ----------
    prev : numpy.ndarray
        The first image (can be a grayscale or color image)

    frame : numpy.ndarray
        The second image (can be a grayscale or color image)

    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can

    Returns
    -------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches : list<cv2.DMatch>
        A list of the top num_matches matches between the keypoint descriptor
        lists from image_1 and image_2
    """
    feat_detector = cv2.ORB_create(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),key=lambda x: x.distance)[:num_matches]
    
    return image_1_kp, image_2_kp, matches
    
def findTransform(prev, frame, num_matches):
    """Returns the transformation between two input images.

    Parameters
    ----------
    prev : numpy.ndarray
        The first image (can be a grayscale or color image)

    frame : numpy.ndarray
        The second image (can be a grayscale or color image)
        
    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can

    Returns
    -------
    tx : float
        The transformation along the x-axis between the keypoints of image 1
        and image 2.
    
    ty : float
        The transformation along the y-axis between the keypoints of image 1
        and image 2.
    
    ta : float
        The transformation in rotation between the keypoints of image 1 and
        image 2.
    
    ts : float
        The transformation in scale between the keypoints of image and image
        2.
    """
    image_1_kp, image_2_kp, matches = findMatchesBetweenImages(prev, frame, num_matches)
    
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i in range(len(matches)):
        queryIdx = matches[i].queryIdx
        trainIdx = matches[i].trainIdx
        
        image_1_points[i,0] = image_1_kp[queryIdx].pt
        image_2_points[i,0] = image_2_kp[trainIdx].pt
    
    transform,_ = cv2.estimateAffinePartial2D(image_1_points, image_2_points)
    
    tx = transform[0][2]
    ty = transform[1][2]
    ta = np.arctan2(transform[1][0], transform[0][0])
    ts = np.sqrt(transform[1][0]**2 + transform[0][0]**2)
    
    return tx, ty, ta, ts

def getTransforms(video):
    """Returns the transformations between each consecutive frames of the
    video.
    
    Parameters
    ----------
    video : cv2.videoCapture
        The sequence of frames that will be processed.
    
    Returns
    -------
    transforms : list<list<float>>
        The transformations between each consecutive frames of the video.
    """
    transforms = [[], [], [], []]
    _, prev = video.read()
    nbr_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_nbr in range(nbr_frames-1):
        success, frame = video.read()
        if not success:
            break
            
        tx, ty, ta, ts = findTransform(prev, frame, 100)
        
        transforms[0].append(tx)
        transforms[1].append(ty)
        transforms[2].append(ta)
        transforms[3].append(ts)
        
        prev = frame
        sys.stdout.write("\rTransforms Extraction: {0}%".format(int((frame_nbr/nbr_frames)*100+1)))
        sys.stdout.flush()
    
    print()
    return transforms

def findPolynomial(x_data, y_data, start, end, degree, DIFFERENCE):
    """Returns either a constant or linear or parabolic segment,
    approximating the data in y_data. It computes the result using recursion.
    
    Parameters
    ----------
    x_data : np.array<int>
        A list corresponding to a sample of frame numbers for which we are
        looking for a polynomial, and have not found one yet.
    
    y_data : np.array<float>
        A list corresponding to a sample of the camera trajectory for which
        we are looking for a polynomial, and have not found one yet.
    
    start : integer
        The index of the first element to use from the x_data and y_data
        lists.
    
    end : integer
        The index of the last element to use from the x_data and y_data
        lists.
    
    degree : integer
        The degree of the polynomial computed.
            0 : constant
            1 : linear
            2 : polynomial
    
    DIFFERENCE : integer
        The maximum difference between the original trajectory and the
        polynomial computed.
    
    Returns
    -------
    start : integer
        The index of the first element of the polynomial computed.
    
    end : integer
        The index of the last element of the polynomial computed.
    
    polynomial : np.array<float>
        The polynomial approximating the y_data.
    """
    index_start = np.where(x_data == start)[0][0]
    index_end = np.where(x_data == end)[0][0]
    a = np.polyfit(x_data[index_start:index_end], y_data[index_start:index_end], degree)
    polynomial = np.zeros(end-start)
    for i in range(len(polynomial)):
        for j in range(a.size):
            polynomial[i] += a[j]*x_data[i+index_start]**(a.size-j-1)
    for i in range(len(polynomial)):
        if abs(polynomial[i] - y_data[i+index_start]) > DIFFERENCE:
            if end - start < max(5, 50-40*degree):
                if end+3 <= x_data[-1]:
                    return findPolynomial(x_data, y_data, start+2, start+5, degree, DIFFERENCE)
                else:
                    return None, None, None
            else:
                return start, end, polynomial
    if end+3 <= x_data[-1]:
        return findPolynomial(x_data, y_data, start, end+2, degree, DIFFERENCE)
    else:
        if end - start < max(5, 50-40*degree):
            return None, None, None
        else:
            return start, end, polynomial

def findSmoothTrajectory(trajectory, DIFFERENCE):
    """Returns the trajectory smoothed using constant, linear and parabolic
    segments.
    
    Parameters
    ----------
    trajectory : list<float>
        The trajectory that must be stabilized.
    
    DIFFERENCE : integer
        Maximum difference allowed between the original path and the new path.
        
    Returns
    -------
    smoothTrajectory :
        The trajectory smoothed using polynomials.
    """
    x_data = np.arange(len(trajectory))
    y_data = trajectory
    smoothTrajectory = np.zeros(len(trajectory))

    INTERVAL = 50
    
    for degree in range(3):
        temp = []
        count = 0
        for frame_nbr in range(len(trajectory)):
            if smoothTrajectory[frame_nbr] == 0.:
                count+=1
                if count == len(trajectory):
                    temp.append(np.arange(0,frame_nbr+1))
                elif frame_nbr == len(trajectory)-1 and count > INTERVAL+1:
                    temp.append(np.arange(frame_nbr+1+INTERVAL-count,frame_nbr+1))
            else:
                if count > INTERVAL+1:
                    temp.append(np.arange(frame_nbr+1+INTERVAL-count,frame_nbr+1))
                count = 0

        for i in range(len(temp)):
            start = temp[i][0]
            end = start+3
            while True:
                if len(x_data[start:temp[i][-1]]) <= 6:
                    break
                start, end, polynomial = findPolynomial(x_data[start:temp[i][-1]], y_data[start:temp[i][-1]], start, end, degree, DIFFERENCE)
                if start == None:
                    break
                else:
                    for j in range(len(polynomial)):
                        smoothTrajectory[j+start] = polynomial[j]
                    start = end+INTERVAL
                    end = start+3
                if end+1 >= temp[i][-1]:
                    break
    
    for frame_nbr in range(0, 10):
        if smoothTrajectory[frame_nbr] == 0.:
            smoothTrajectory[frame_nbr] = trajectory[frame_nbr]
    for frame_nbr in range(len(trajectory)-10, len(trajectory)):
        if smoothTrajectory[frame_nbr] == 0.:
            smoothTrajectory[frame_nbr] = trajectory[frame_nbr]

    count = 0
    for frame_nbr in range(len(trajectory)):
        if smoothTrajectory[frame_nbr] == 0.:
            count+=1
        else:
            if count >= INTERVAL:
                x_data = np.arange(frame_nbr-count-2,frame_nbr)
                y_data = [smoothTrajectory[frame_nbr-count-1]]
                for i in range(count):
                    y_data.append(trajectory[frame_nbr-count+i])
                y_data.append(smoothTrajectory[frame_nbr])
                a = np.polyfit(x_data, y_data, 2)
                polynomial = np.zeros(count+2)
                for i in range(count+2):
                    for j in range(a.size):
                        polynomial[i] += a[j]*x_data[i]**(a.size-j-1)
                for i in range(count+2):
                    smoothTrajectory[frame_nbr-count+i-1] = polynomial[i]
            count = 0
    
    for frame_nbr in range(len(trajectory)):
        if smoothTrajectory[frame_nbr] == 0.:
            smoothTrajectory[frame_nbr] = trajectory[frame_nbr]
            
    return smoothTrajectory

def getSmoothTrajectories(trajectories):
    """Returns every trajectory smoothed. The trajectories along the x_axis
    and the y_axis are smoothed using constant, linear and parabolic
    segments. The trajectory of the rotation is smoothed by a constant
    segment. The trajectory of the scale is smoothed by a linear segment.
    
    Parameters
    ----------
    trajectories : list<list<float>>
        The trajectories along the x_axis, the y_axis, the rotation and the
        scale.
    
    Returns
    -------
    smoothTrajectories : list<list<float>>
        The trajectories smoothed.
    """
    smoothTrajectories = [[], [], [], []]
    
    for trajectory_nbr in range(4):
        if trajectory_nbr == 3:
            smoothTrajectories[trajectory_nbr] = findSmoothTrajectory(trajectories[trajectory_nbr], 0.1)
        else:
            if trajectory_nbr == 2:
                smoothTrajectories[trajectory_nbr] = findSmoothTrajectory(trajectories[trajectory_nbr], 0.1)
            else:
                smoothTrajectories[trajectory_nbr] = findSmoothTrajectory(trajectories[trajectory_nbr], 100)
            curve_pad = np.lib.pad(smoothTrajectories[trajectory_nbr], (15, 15), 'edge')
            smoothTrajectories[trajectory_nbr] = np.convolve(curve_pad, np.ones(31)/31, 'same')
            smoothTrajectories[trajectory_nbr] = smoothTrajectories[trajectory_nbr][15:-15]
        sys.stdout.write("\rStabilizing: {0}%".format(25*(trajectory_nbr+1)))
        sys.stdout.flush()
    
    print()
    return smoothTrajectories

def graphs(trajectories, smoothTrajectories):
    """Displays four graphs:
    - Top Left: the original and the smoothed trajectories along the x-axis.
    - Top Right: the original and the smoothed trajectories along the y-axis.
    - Bottom Left: the original and the smoothed trajectories of the rotation.
    - Bottom Right: the original and the smoothed trajectories of the scale.
    
    In blue: the original video trajectories.
    In orange: the smoothed video trajectories.
    
    Parameters
    ----------
    trajectories : list<list<float>>
        The cumulative sums of the transformations between each consecutive
        frames.
    
    smoothedTrajectories : list<list<float>>
        The smoothed cumulative sums of the transformations between each
        consecutive frames.
    """
    mat.subplot(2, 2, 1)
    mat.plot(trajectories[0])
    mat.plot(smoothTrajectories[0])
    mat.subplot(2, 2, 2)
    mat.plot(trajectories[1])
    mat.plot(smoothTrajectories[1])
    mat.subplot(2, 2, 3)
    mat.plot(trajectories[2])
    mat.plot(smoothTrajectories[2])
    mat.subplot(2, 2, 4)
    mat.plot(trajectories[3])
    mat.plot(smoothTrajectories[3])
    mat.show()

def fixBorder(frame, CROP_PERCENTAGE):
    """Returns a frame cropped according to a percentage.
    
    Parameters
    ----------
    frame : cv2.videoCapture.frame
        The original frame.
        
    CROP_PERCENTAGE : integer
        The percentage representing the amount that will be cropped from the
        original video.
    
    Returns
    -------
    frame : cv2.videoCapture.frame
        The original frame cropped.
    
    """
    matrix = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), 0, 1+0.01*CROP_PERCENTAGE)
    frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
    
    return frame

def writeVideo(smoothTransforms, video, videoName, CROP_PERCENTAGE):
    """Writes a video file from an original video, modified according to a transformation matrix, and cropped according to a percentage.
    
    Parameters
    ----------
    smoothTransforms :
        The new transformations between each consecutive frames of the video, smoothed.
    
    video : cv2.videoCapture
        The sequence of frames from the original video that will be
        processed.
        
    videoName : string
        Name of the output video.
    
    CROP_PERCENTAGE : integer
        The percentage representing the amount that will be cropped from the
        original video.
    """
    CROP_PERCENTAGE = min(100, CROP_PERCENTAGE)
    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(videoName, 0x7634706d, fps, (W, H))
    m = np.zeros((2,3), np.float32)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(len(smoothTransforms[0])):
        success, frame = video.read()
        if not success:
            break
        
        dx = smoothTransforms[0, i]
        dy = smoothTransforms[1, i]
        da = smoothTransforms[2, i]
        ds = smoothTransforms[3, i]
        
        m[0,0] = np.cos(da)*ds
        m[0,1] = -np.sin(da)*ds
        m[1,0] = np.sin(da)*ds
        m[1,1] = np.cos(da)*ds
        m[0,2] = dx
        m[1,2] = dy
        
        frame_stabilized = cv2.warpAffine(frame, m, (W, H))
        frame_stabilized = fixBorder(frame_stabilized, CROP_PERCENTAGE)
        out.write(frame_stabilized)
        
        sys.stdout.write("\rVideo Retargeting: {0}%".format(int(i/len(smoothTransforms[0])*100+1)))
        sys.stdout.flush()
        
    out.release()
    print()
