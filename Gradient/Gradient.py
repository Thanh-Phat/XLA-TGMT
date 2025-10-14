import cv2
import numpy as np
import matplotlib.pyplot as plt

#  Đọc ảnh gốc và chuyển sang ảnh xám
img = cv2.imread(r"D:\BaoCao\XLA-TGMT\img\anhmeo.jpg")      
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Làm mượt ảnh để giảm nhiễu (lọc Gaussian)
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

#  Tính đạo hàm theo hướng x và y (Sobel)
sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

#  Toán tử Prewitt (tự định nghĩa kernel)
prewitt_kernelx = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
prewitt_kernely = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
prewittx = cv2.filter2D(gray_blur, -1, prewitt_kernelx)
prewitty = cv2.filter2D(gray_blur, -1, prewitt_kernely)
prewitt = cv2.magnitude(prewittx.astype(float), prewitty.astype(float))

#  Toán tử Roberts Cross (kernel 2x2)
roberts_cross_x = np.array([[1, 0],
                            [0, -1]], dtype=int)
roberts_cross_y = np.array([[0, 1],
                            [-1, 0]], dtype=int)
robertsx = cv2.filter2D(gray_blur, -1, roberts_cross_x)
robertsy = cv2.filter2D(gray_blur, -1, roberts_cross_y)
roberts = cv2.magnitude(robertsx.astype(float), robertsy.astype(float))

#  Hiển thị kết quả
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(gray, cmap='gray')
plt.title('Ảnh gốc (Grayscale)')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(sobel, cmap='gray')
plt.title('Biên phát hiện bằng Sobel')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(prewitt, cmap='gray')
plt.title('Biên phát hiện bằng Prewitt')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(roberts, cmap='gray')
plt.title('Biên phát hiện bằng Roberts')
plt.axis('off')

plt.tight_layout()
plt.show()
