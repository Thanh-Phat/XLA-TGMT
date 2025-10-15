import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Đọc ảnh gốc và chuyển sang ảnh xám ---
img = cv2.imread(r"D:\BaoCao\XLA-TGMT\img\anhmeo.jpg")  # đổi đường dẫn đúng với thư mục của bạn
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Làm mượt ảnh để giảm nhiễu ---
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# --- Phát hiện biên bằng Canny ---
# Thường ngưỡng thấp = 50, ngưỡng cao = 150 (bạn có thể thay đổi để thử nghiệm)
edges = cv2.Canny(gray_blur, 50, 150)

# --- Hiển thị kết quả ---
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title('Ảnh gốc (Grayscale)')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title('Biên phát hiện bằng Canny')
plt.axis('off')

plt.tight_layout()
plt.show()
