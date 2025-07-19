#include <Wire.h>
#include <MPU9250_asukiaaa.h>

MPU9250_asukiaaa mySensor;

void setup() {
  Serial.begin(115200);
  Wire.begin(22, 21); // SDA, SCL

  mySensor.setWire(&Wire);
  mySensor.beginAccel();
  mySensor.beginGyro();
  mySensor.beginMag();

  Serial.println("MPU9250 initialized.");
}

void loop() {
  mySensor.accelUpdate();
  mySensor.gyroUpdate();
  mySensor.magUpdate();

  Serial.print("Accel X: "); Serial.print(mySensor.accelX());
  Serial.print(" Y: "); Serial.print(mySensor.accelY());
  Serial.print(" Z: "); Serial.print(mySensor.accelZ());

  Serial.print(" | Gyro X: "); Serial.print(mySensor.gyroX());
  Serial.print(" Y: "); Serial.print(mySensor.gyroY());
  Serial.print(" Z: "); Serial.print(mySensor.gyroZ());

  Serial.print(" | Mag X: "); Serial.print(mySensor.magX());
  Serial.print(" Y: "); Serial.print(mySensor.magY());
  Serial.print(" Z: "); Serial.print(mySensor.magZ());

  Serial.println();
  delay(10);
}
