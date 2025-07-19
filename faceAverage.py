#!/usr/bin/env python
# pacman2108, 2025

import os
import cv2
import numpy as np
import math
import sys

def readPoints(path):
    """Read landmark points from text files in the specified directory."""
    pointsArray = []
    for filePath in sorted(os.listdir(path)):
        if filePath.endswith(".txt"):
            points = []
            try:
                with open(os.path.join(path, filePath)) as file:
                    for line in file:
                        x, y = line.split()
                        points.append((int(x), int(y)))
                pointsArray.append(points)
            except Exception as e:
                print(f"[!] Error reading {filePath}: {e}")
    return pointsArray

def readImages(path):
    """Read and validate images from the specified directory."""
    imagesArray = []
    for filePath in sorted(os.listdir(path)):
        if filePath.endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(os.path.join(path, filePath))
            if img is None:
                print(f"[!] Failed to read image: {filePath}")
                continue
            img = np.float32(img) / 255.0
            imagesArray.append(img)
    return imagesArray

def similarityTransform(inPoints, outPoints):
    """Compute similarity transform between two sets of points."""
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)
    
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    
    # Fake third point for affine transform
    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]
    inPts.append([int(xin), int(yin)])
    
    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]
    outPts.append([int(xout), int(yout)])
    
    # Use cv2.estimateAffinePartial2D and handle return value
    tform, _ = cv2.estimateAffinePartial2D(np.array(inPts), np.array(outPts), method=cv2.RANSAC)
    return tform if tform is not None else np.eye(2, 3, dtype=np.float32)

def rectContains(rect, point):
    """Check if a point is inside a rectangle."""
    return (rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3])

def calculateDelaunayTriangles(rect, points):
    """Calculate Delaunay triangulation for given points within a rectangle."""
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    
    for t in triangleList:
        pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if all(rectContains(rect, p) for p in pt):
            ind = []
            for j in range(3):
                for k in range(len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
    
    return delaunayTri

def constrainPoint(p, w, h):
    """Constrain point to be within image boundaries."""
    return (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))

def applyAffineTransform(src, srcTri, dstTri, size):
    """Apply affine transform to a triangular region."""
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, 
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warpTriangle(img1, img2, t1, t2):
    """Warp and blend triangular regions from img1 to img2."""
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    
    t1Rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2Rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    t2RectInt = [(int(t2[i][0] - r2[0]), int(t2[i][1] - r2[1])) for i in range(3)]
    
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0))
    
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, (r2[2], r2[3]))
    img2Rect = img2Rect * mask
    
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] *= (1.0 - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] += img2Rect

def average_faces(path, output_path='average.png', w=1000, h=1000):
    """Average faces by aligning them based on landmarks."""
    if not os.path.exists(path):
        print(f"[!] Directory {path} does not exist.")
        return
    
    images = readImages(path)
    all_points = readPoints(path)
    
    if not images or not all_points:
        print("[!] No valid images or landmarks found.")
        return
    
    if len(images) != len(all_points):
        print("[!] Mismatch between number of images and landmark files.")
        return
    
    boundary_pts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2),
                            (w-1,h-1), (w/2,h-1), (0,h-1), (0,h/2)], dtype=np.float32)
    eyecorner_dst = [(int(0.3 * w), int(h / 3)), (int(0.7 * w), int(h / 3))]
    
    images_norm = []
    points_norm = []
    points_avg = np.zeros((len(all_points[0]) + len(boundary_pts), 2), np.float32)
    
    for i in range(len(images)):
        if len(all_points[i]) < 68:  # Assuming 68 landmarks per face
            print(f"[!] Skipping image {i}: insufficient landmarks.")
            continue
        
        eyecorner_src = [all_points[i][36], all_points[i][45]]
        tform = similarityTransform(eyecorner_src, eyecorner_dst)
        img = cv2.warpAffine(images[i], tform, (w, h))
        
        points = cv2.transform(np.reshape(np.array(all_points[i], dtype=np.float32), (-1, 1, 2)), tform)
        points = np.reshape(points, (-1, 2))
        points = np.append(points, boundary_pts, axis=0)
        
        images_norm.append(img)
        points_norm.append(points)
        points_avg += points / len(images)
    
    if not images_norm:
        print("[!] No valid images after processing.")
        return
    
    dt = calculateDelaunayTriangles((0, 0, w, h), points_avg)
    output = np.zeros((h, w, 3), np.float32)
    
    for i in range(len(images_norm)):
        print(f"[+] Aligning face {i + 1}/{len(images_norm)}...")
        img_temp = np.zeros((h, w, 3), np.float32)
        for tri in dt:
            tin = [constrainPoint(points_norm[i][idx], w, h) for idx in tri]
            tout = [constrainPoint(points_avg[idx], w, h) for idx in tri]
            warpTriangle(images_norm[i], img_temp, tin, tout)
        output += img_temp
    
    output /= len(images_norm)
    cv2.imwrite(output_path, 255 * output)
    print(f"[✓] Saved averaged face to {output_path}")

if __name__ == '__main__':
    path = 'faces/'
    average_faces(path)

folder_path = input("File location: ")\


def sanity_check(folder_path):
    images = set()
    landmarks = set()

    for file in os.listdir(folder_path):
        base, ext = os.path.splitext(file)
        ext = ext.lower()

        if ext in ['.jpg', '.jpeg', '.png']:
            images.add(base)
        elif ext == '.txt':
            landmarks.add(base)

    print(f"\n🖼️ Images found: {len(images)}")
    print(f"📍 Landmark files found: {len(landmarks)}\n")

    only_images = images - landmarks
    only_landmarks = landmarks - images

    if only_images:
        print("⚠️ These images have NO matching .txt landmark files:")
        for name in sorted(only_images):
            print(f"  - {name}")
    else:
        print("✅ All images have matching landmarks.")

    if only_landmarks:
        print("\n⚠️ These landmark files have NO matching images:")
        for name in sorted(only_landmarks):
            print(f"  - {name}")
    else:
        print("✅ All landmark files have matching images.")

sanity_check(folder_path)
