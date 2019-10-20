import cv2
import numpy as np
import sys

if len(sys.argv) < 3:
    print("Usage: python stichimage.py img1 img2")
    sys.exit(0)

img1name = sys.argv[1]
img2name = sys.argv[2]

img1_ = cv2.imread(img1name)
img1_ = cv2.resize(img1_, (0,0), fx=1, fy=1)
img1 = cv2.cvtColor(img1_, cv2.COLOR_BGR2GRAY)

img2_ = cv2.imread(img2name)
img2_ = cv2.resize(img2_, (0,0), fx=1, fy=1)
img2 = cv2.cvtColor(img2_, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#cv2.imshow('img1_mod', cv2.drawKeypoints(img1, kp1, None))
#cv2.imshow('img2_mod', cv2.drawKeypoints(img2, kp2, None))
#cv2.waitKey(0)

match = cv2.BFMatcher()
matches = match.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
img3 = cv2.drawMatches(img1_, kp1, img2_, kp2, good, None, **draw_params)
cv2.resizeWindow("output", 2400, 2400)
cv2.imshow("output", img3)
cv2.waitKey(0)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    cv2.imshow("output", img2)
    cv2.waitKey(0)
else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

dst = cv2.warpPerspective(img1_, M, (img1.shape[1] + img1_.shape[1], img1.shape[0]))
dst[0:img2_.shape[0],0:img2_.shape[1]] = img2_
# cv2.imshow("output", dst)
# cv2.waitKey(0)

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

# cv2.imshow("output", dst)
# cv2.waitKey(0)

cv2.imwrite(img1name+"_"+img2name+"_stitched_crop.png", trim(dst))