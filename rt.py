import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_s', root='./insightface_model')
app.prepare(ctx_id=0, det_size=(640, 640))

img1 = cv2.imread('me1.jpg')
img2 = cv2.imread('me2.jpg')

if img1 is None or img2 is None:
    print("错误：找不到图片，请检查路径。")
    exit()

faces1 = app.get(img1)
faces2 = app.get(img2)

if len(faces1) == 0 or len(faces2) == 0:
    print("错误：某张图片中没有检测到人脸。")
    exit()

embedding1 = faces1[0].embedding
embedding2 = faces2[0].embedding

dot_product = np.dot(embedding1, embedding2)
norm1 = np.linalg.norm(embedding1)
norm2 = np.linalg.norm(embedding2)
similarity = dot_product / (norm1 * norm2)

print(f"两张脸的余弦相似度为: {similarity:.4f}")

threshold = 0.5
if similarity > threshold:
    print("✅ 系统判定：是同一个人！打卡成功。")
else:
    print("❌ 系统判定：不是同一个人！")