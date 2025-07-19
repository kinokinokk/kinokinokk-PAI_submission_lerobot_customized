import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

#表示用GUI
import tkinter as tk

# --------------------
# 設定
# --------------------
camera_index = 2
model_path = "dice_classifier.pth"
num_classes = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# 前処理（学習時と合わせる）
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------
# モデル読み込み
# --------------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --------------------
# カメラ起動 & 推論
# --------------------
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("カメラが見つかりません")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("画像取得に失敗")
    exit()

# OpenCV → PIL → Tensor
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)
input_tensor = transform(image_pil).unsqueeze(0).to(device)

# 推論
with torch.no_grad():
    outputs = model(input_tensor)
    pred_class = torch.argmax(outputs, 1).item()
    predicted_number = pred_class + 1  # クラス0→数字1 に補正

# 結果を表示
print(f"推論結果：{predicted_number}")





# ----------OKOSUNからのメッセージ ----------
root = tk.Tk()
root.title("OKOSUNからのメッセージ")
root.geometry("900x600")

if predicted_number == 1:
    label = tk.Label(root, text=f"ファイトだ！あともう少し頑張るぞ！！", font=("Helvetica", 30))
    label.pack(expand=True)


if predicted_number == 2:
    label = tk.Label(root, text=f"夢の中でも勉強してたことにして 再開しよっか。", font=("Helvetica", 30))
    label.pack(expand=True)


if predicted_number == 3:
    label = tk.Label(root, text=f"机の上で寝ると 首いてぇよな。勉強の代償。", font=("Helvetica", 30))
    label.pack(expand=True)


if predicted_number == 4:
    label = tk.Label(root, text=f"君が寝ている間も僕はずっと見てたんだよ・・・", font=("Helvetica", 30))
    label.pack(expand=True)


if predicted_number == 5:
    label = tk.Label(root, text=f"今日は脳みその筋トレ日。いい汗かこう。", font=("Helvetica", 30))
    label.pack(expand=True)


if predicted_number == 6:
    label = tk.Label(root, text=f"ようこそ現世へ。脳みそ 再起動完了した？", font=("Helvetica", 30))
    label.pack(expand=True)


# 閉じるボタン
button = tk.Button(root, text="閉じる", command=root.destroy)
button.pack(pady=10)

root.mainloop()




# 確認用画像に結果を描画して表示
label_text = f"Predicted: {predicted_number}"
cv2.putText(frame, label_text, (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
