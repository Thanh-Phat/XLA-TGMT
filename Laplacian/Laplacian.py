# Nếu bạn chưa có ảnh "meocute.jpg", hãy upload từ máy tính bạn lên Colab
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang ảnh xám
img = cv2.imread(r'D:\BaoCao\XLA-TGMT\img\meocute.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Không tìm thấy file 'meocute.jpg'.")

# Làm mịn bằng Gaussian
sigma = 1.4
gk = int(6 * sigma + 1)
if gk % 2 == 0:
    gk += 1

blurred = cv2.GaussianBlur(img, (gk, gk), sigma)
# Áp dụng toán tử Laplacian
ksize = 3
lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
# Hàm phát hiện zero-crossing
def zero_crossing(lap, threshold=0.01):
    h, w = lap.shape
    edges = np.zeros((h,w), dtype=np.uint8)
    lap_norm = lap / (np.max(np.abs(lap)) + 1e-9)

    for y in range(1, h-1):
        for x in range(1, w-1):
            patch = lap_norm[y-1:y+2, x-1:x+2]
            p = lap_norm[y, x]
            if np.any(patch * p < 0) and (np.max(np.abs(patch)) > threshold):
                edges[y, x] = 255
    return edges

edges = zero_crossing(lap, threshold=0.03)

# Hiển thị kết quả
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Ảnh gốc (Grayscale)')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(blurred, cmap='gray')
plt.title('Ảnh sau Gaussian Blur')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(lap, cmap='gray')
plt.title('Ảnh sau Laplacian')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(edges, cmap='gray')
plt.title('Ảnh biên (Zero-Crossing)')
plt.axis('off') 

plt.tight_layout()
plt.show()
# Kết luận
print("KẾT LUẬN:")
print("- Giải thuật Laplacian phát hiện biên bằng cách tìm nơi đạo hàm bậc hai đổi dấu (zero-crossing).")
print("- Trước khi tính Laplacian, cần làm mịn ảnh bằng Gaussian để giảm nhiễu.")
print("- Kết quả nhạy với nhiễu, nhưng cho biên chính xác vị trí biến thiên cường độ mạnh.")