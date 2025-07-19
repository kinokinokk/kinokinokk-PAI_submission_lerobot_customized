import cv2
import os
from datetime import datetime

# --- カメラ設定 ---
camera_index = 2
cap = cv2.VideoCapture(camera_index)

# --- 保存先フォルダ設定 ---
base_dir = "camera_dataset/train"
os.makedirs(base_dir, exist_ok=True)

# 1〜6 の各フォルダを作成（存在しなければ）
for i in range(1, 7):
    os.makedirs(os.path.join(base_dir, str(i)), exist_ok=True)

print("カメラ起動：数字キー（1〜6）で保存、qで終了")

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラが認識できません")
        break

    # ウィンドウ表示
    cv2.imshow("Press 1-6 to save image, q to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    # 1〜6 のキーが押された場合に保存
    if key in [ord(str(d)) for d in range(1, 7)]:
        label = chr(key)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.jpg"
        save_path = os.path.join(base_dir, label, filename)
        cv2.imwrite(save_path, frame)
        print(f"{label} に保存: {save_path}")

    # qキーで終了
    elif key == ord('q'):
        print("終了")
        break

# --- 後始末 ---
cap.release()
cv2.destroyAllWindows()
