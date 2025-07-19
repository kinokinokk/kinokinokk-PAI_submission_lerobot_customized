# physical_ai最終課題　BNO０５５姿勢推定コード
# ヨー角対応ver


import sys
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtGui import QMatrix4x4
from scipy.spatial.transform import Rotation as R
import serial
import threading

from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler  # クォータニオン→オイラー角変換




# Arduinoからデータを受け取る＆eulerを算出
class BNO055Viewer_Server(QtWidgets.QWidget):
    def __init__(self, title="IMU Viewer"):
        super().__init__()
        self.setWindowTitle(title)
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=5)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

        # 軸（XYZ）
        self.axes = [self.create_axis((1,0,0,1)),  # X: 赤
                     self.create_axis((0,1,0,1)),  # Y: 緑
                     self.create_axis((0,0,1,1))]  # Z: 青

        for axis in self.axes:
            self.view.addItem(axis)

        
        # 軸表示の代わりに直方体を作成
        self.box = self.create_box_mesh()
        self.view.addItem(self.box)


        self.euler = np.array([0.0, 0.0, 0.0])
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_orientation)
        self.timer.start(50) #いじった

        self.ser = serial.Serial('/dev/ttyACM2', 115200)
        self.dt = 0.1 #いじった


        # Madgwick_filterの使用にあたっての初期化
        self.madgwick = Madgwick()
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # 初期クォータニオン

    def create_axis(self, color):
        return gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=color, width=3, antialias=True)

    
    def create_box_mesh(self):
        # 頂点 (8点)
        verts = np.array([
            [1, 1, 1], [-1, 1, 1],
            [-1, -1, 1], [1, -1, 1],
            [1, 1, -1], [-1, 1, -1],
            [-1, -1, -1], [1, -1, -1]
        ]) * 0.5  # スケール調整

        # 三角形の面 (12枚)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # 前
            [4, 7, 6], [4, 6, 5],  # 後
            [0, 4, 5], [0, 5, 1],  # 上
            [3, 2, 6], [3, 6, 7],  # 下
            [1, 5, 6], [1, 6, 2],  # 左
            [0, 3, 7], [0, 7, 4],  # 右
        ])


        # 各面の色（RGB + Alpha）
        colors = np.array([
            [1, 0, 0, 1],  # 前面: 赤
            [0, 1, 0, 1],  # 背面: 緑
            [0, 0, 1, 1],  # 上面: 青
            [1, 1, 0, 1],  # 下面: 黄
            [0, 1, 1, 1],  # 左面: シアン
            [1, 0, 1, 1],  # 右面: マゼンタ
        ])
        # 頂点ごとに色を割り当て（面ごとに4点なのでそれぞれ同じ色を4回）
        vertex_colors = np.repeat(colors, 2, axis=0)

        mesh = gl.MeshData(vertexes=verts, faces=faces, vertexColors=vertex_colors)
        box_item = gl.GLMeshItem(meshdata=mesh, smooth=False, drawEdges=True, edgeColor=(0.3, 0.3, 0.3, 1))
        return box_item



    def update_orientation(self):
        rot = R.from_euler('xyz', np.radians(self.euler)).as_matrix()
        
        # ３軸を動かすためのコード（立方体にしたため、コメントアウト中）
        """
        for i in range(3):
            self.axes[i].setData(pos=np.array([[0, 0, 0], rot[:, i]]))
            """
        
        m = np.eye(4)
        m[:3, :3] = rot
        self.box.setTransform(QMatrix4x4(*m.T.flatten()))

    # Complementary filterの定義式通りの計算
    # 係数をalpha, サンプリング周期をdtとしている。（カットオフ周波数はこの二つのハイパーパラメータより求まる。）
    def complementary_filter(self, prev_angle, gyro, accel_angle, alpha=0.98, dt=0.05):
        gyro_deg = np.rad2deg(gyro)# ← ここで「ラジアン/秒 → 度/秒」に戻す
        gyro_angle = prev_angle + gyro_deg * dt
        return alpha * gyro_angle + (1 - alpha) * accel_angle

    def calc_angle(self, accel, yaw):
        accel = accel / np.linalg.norm(accel)
        ax, ay, az = accel
        roll = np.degrees(np.arctan2(ay, az))
        pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
        return np.array([roll, pitch, yaw])#加速度からオイラー角（の一種、Z, Y, X）を求める

    #地磁気センサからヨー角を計算するための関数　　　【傾斜補正付き磁気コンパス法（tilt-compensated compass）】
    def yaw_from_mag(self, mag, roll_deg_from_accel, pitch_deg_from_accel):
        roll = np.radians(roll_deg_from_accel)
        pitch = np.radians(pitch_deg_from_accel)
        mx, my, mz = mag

        mx2 = mx * np.cos(pitch) + mz * np.sin(pitch)
        my2 = mx * np.sin(roll) * np.sin(pitch) + my * np.cos(roll) - mz * np.sin(roll) * np.cos(pitch)

        yaw = np.degrees(np.arctan2(-my2, mx2))
        if yaw < 0:
            yaw += 360.0  # 0～360度に変換（お好みで）
        return yaw #ここでのヨーは度

    
    #ジャイロと磁気のヨー角を相補フィルタで融合【ヨー角用Complementary filter】（←精度悪かった。）
    def For_yaw_complementary_filter(self, prev_yaw, gyro_z, mag_yaw, alpha=0.98, dt=0.05):
        gyro_deg = np.rad2deg(gyro_z)
        gyro_yaw = prev_yaw + gyro_deg * dt

        # Wrap around (0-360)
        gyro_yaw = gyro_yaw % 360
        mag_yaw = mag_yaw % 360

        # 差分が180度以上にならないように調整
        diff = (mag_yaw - gyro_yaw + 540) % 360 - 180
        fused_yaw = (gyro_yaw + alpha * diff) % 360

        return fused_yaw




    def serial_loop(self):
        while True:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            #print("DEBUG: Serial received ->", line)
            if not line or '|' not in line:
                continue
            try:
                part1, part2 = line.split('|')
                data1 = list(map(float, part1.split(':')[1].split(',')))
                #print("DEBUG: Parsed data1 =", data1)
                ax1, ay1, az1, gx1, gy1, gz1 = data1[:6]
                magx1, magy1, magz1 = data1[6:9]
                mag = np.array([magx1, magy1, magz1])
            except Exception as e:
                print("ERROR: Failed to parse serial data:", e)
                continue

            accel_angle1 = self.calc_angle(np.array([ax1, ay1, az1]), self.euler[2])
            gyro1 = np.deg2rad(np.array([gx1, gy1, gz1]))#ジャイロセンサの角速度データを「度（°）」から「ラジアン（rad）」に変換

            self.euler = self.complementary_filter(self.euler, gyro1, accel_angle1, self.dt)
            self.euler[2] = self.yaw_from_mag(mag, accel_angle1[0], accel_angle1[1])
            print(f"roll:{self.euler[0]}, pitch:{self.euler[1]}, yaw:{self.euler[2]}")



            if self.euler[0] < 60:
                print("寝てます    OKOSUN始動")


                with open("signal.txt", "w") as f:
                    f.write("START")

                print("start signal sent!")


            """
            mag_yaw = self.yaw_from_mag(mag, accel_angle1[0], accel_angle1[1])
            self.euler[2] = self.For_yaw_complementary_filter(self.euler[2], gyro1[2], mag_yaw, self.dt)
            """
            #self.euler = self.Madgwick_filter(np.array([ax1, ay1, az1]), gyro1, mag)










def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # 2つのインスタンス作成
    viewer_server = BNO055Viewer_Server("IMU Sensor")
    viewer_server.show()


    # threadの作成
    t1 = threading.Thread(target=viewer_server.serial_loop, daemon=True)
    t1.start()

    sys.exit(app.exec_())

    #PyQtアプリは、GUIイベント（描画・タイマー・マウス入力など）をすべて exec_() が管理するメインイベントループで処理する。

if __name__ == "__main__":
    main()