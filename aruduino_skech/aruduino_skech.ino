#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SERVOMIN  265 
#define SERVOMAX  450 

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

int fingerChannels[] = {0, 3, 8, 11, 15};
const int numFingers = 5;

const int zero[]  = {SERVOMAX, SERVOMAX, SERVOMAX, SERVOMAX, SERVOMAX};
const int one[]   = {SERVOMAX, 250, SERVOMAX, SERVOMAX, SERVOMAX};
const int two[]   = {SERVOMAX, 250, SERVOMIN, SERVOMAX, SERVOMAX};
const int three[] = {SERVOMAX, 250, SERVOMIN, 150, SERVOMAX};
const int four[]  = {SERVOMAX, 250, SERVOMIN, 150, 250};
const int five[]  = {SERVOMIN, 250, SERVOMIN, 150, 150};

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(60); 
  executeGesture(zero);
}

void loop() {
  if (Serial.available() > 0) {
    char incomingChar = Serial.read();
    int count = incomingChar - '0';
    
    if (count >= 0 && count <= 5) {
      switch (count) {
        case 0: executeGesture(zero);  break;
        case 1: executeGesture(one);   break;
        case 2: executeGesture(two);   break;
        case 3: executeGesture(three); break;
        case 4: executeGesture(four);  break;
        case 5: executeGesture(five);  break;
      }
    }
  }
}

void executeGesture(const int gestureValues[]) {
  for (int i = 0; i < numFingers; i++) {
    pwm.setPWM(fingerChannels[i], 0, gestureValues[i]);
  }
}