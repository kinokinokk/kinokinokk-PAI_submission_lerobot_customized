#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

Adafruit_BNO055 bno1 = Adafruit_BNO055(55, 0x29); // BNO055 (1)

void setup() {
    Serial.begin(115200);
    Wire.begin();

    // BNO055 (1) 初期化
    if (!bno1.begin()) {
        Serial.println("BNO055 (1) 初期化失敗");
        while (1);
    }
    delay(1000);
}

void loop() {
    sensors_event_t acc1, gyro1, mag1;

    // BNO055 (1) データ取得
    bno1.getEvent(&acc1, Adafruit_BNO055::VECTOR_ACCELEROMETER);
    bno1.getEvent(&gyro1, Adafruit_BNO055::VECTOR_GYROSCOPE);
    bno1.getEvent(&mag1, Adafruit_BNO055::VECTOR_MAGNETOMETER);

    // 出力フォーマット: "1:ax,ay,az,gx,gy,gz,mx,my,mz"
    Serial.print("1:");
    Serial.print(acc1.acceleration.x, 2); Serial.print(",");
    Serial.print(acc1.acceleration.y, 2); Serial.print(",");
    Serial.print(acc1.acceleration.z, 2); Serial.print(",");
    Serial.print(gyro1.gyro.x, 2); Serial.print(",");
    Serial.print(gyro1.gyro.y, 2); Serial.print(",");
    Serial.print(gyro1.gyro.z, 2); Serial.print(",");
    Serial.print(mag1.magnetic.x, 2); Serial.print(",");
    Serial.print(mag1.magnetic.y, 2); Serial.print(",");
    Serial.print(mag1.magnetic.z, 2); Serial.print("|");
    
    Serial.println("");

    delay(10);
}

