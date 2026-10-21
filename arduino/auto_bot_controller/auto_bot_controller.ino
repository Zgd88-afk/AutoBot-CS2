/*
  Auto Bot Arduino Controller

  Board requirement:
  Arduino Leonardo / Micro / Pro Micro or another board that supports USB HID.

  Serial commands:
  forward, backward, backward:<milliseconds>, left, right, turn_left, turn_right, turn:<pixels>,
  aim:<x_pixels>,<y_pixels>, fire, jump, stop, release_move
*/

#include <Keyboard.h>
#include <Mouse.h>

const unsigned long BAUD_RATE = 9600;
const int TURN_STEP = 18;
const int MAX_MOUSE_STEP = 120;
const unsigned long BACKWARD_MS = 350;
const unsigned long JUMP_MS = 80;
const unsigned long FIRE_MS = 35;

String commandBuffer = "";

bool pressingForward = false;
bool pressingBackward = false;
bool pressingLeft = false;
bool pressingRight = false;

void setup() {
  Serial.begin(BAUD_RATE);
  Keyboard.begin();
  Mouse.begin();

  delay(1000);
  stopAll();
}

void loop() {
  while (Serial.available() > 0) {
    char incomingChar = Serial.read();

    if (incomingChar == '\n' || incomingChar == '\r') {
      commandBuffer.trim();
      if (commandBuffer.length() > 0) {
        handleCommand(commandBuffer);
        commandBuffer = "";
      }
    } else {
      commandBuffer += incomingChar;
    }
  }
}

void handleCommand(String command) {
  if (command == "forward") {
    moveForward();
  } else if (command == "backward") {
    moveBackwardTimed(BACKWARD_MS);
  } else if (command.startsWith("backward:")) {
    unsigned long durationMs = command.substring(9).toInt();
    moveBackwardTimed(constrain(durationMs, 0UL, 2000UL));
  } else if (command == "left") {
    moveLeft();
  } else if (command == "right") {
    moveRight();
  } else if (command == "turn_left") {
    turnLeft();
  } else if (command == "turn_right") {
    turnRight();
  } else if (command.startsWith("turn:")) {
    turnByPixels(command.substring(5).toInt());
  } else if (command.startsWith("aim:")) {
    aimByPixels(command.substring(4));
  } else if (command == "fire") {
    fireWeapon();
  } else if (command == "jump") {
    jump();
  } else if (command == "stop") {
    stopAll();
  } else if (command == "release_move") {
    releaseMoveKeys();
  }
}

void moveForward() {
  Keyboard.release('s');
  pressingBackward = false;

  Keyboard.press('w');
  pressingForward = true;
}

void moveBackwardTimed(unsigned long durationMs) {
  releaseMoveKeys();
  Keyboard.press('s');
  pressingBackward = true;
  delay(durationMs);
  Keyboard.release('s');
  pressingBackward = false;
}

void moveLeft() {
  Keyboard.release('d');
  pressingRight = false;

  Keyboard.press('a');
  pressingLeft = true;
}

void moveRight() {
  Keyboard.release('a');
  pressingLeft = false;

  Keyboard.press('d');
  pressingRight = true;
}

void turnLeft() {
  Mouse.move(-TURN_STEP, 0, 0);
}

void turnRight() {
  Mouse.move(TURN_STEP, 0, 0);
}

void turnByPixels(int pixels) {
  while (pixels != 0) {
    int step = constrain(pixels, -MAX_MOUSE_STEP, MAX_MOUSE_STEP);
    Mouse.move(step, 0, 0);
    pixels -= step;
    delay(2);
  }
}

void aimByPixels(String payload) {
  int commaIndex = payload.indexOf(',');
  if (commaIndex < 0) {
    return;
  }

  int xPixels = payload.substring(0, commaIndex).toInt();
  int yPixels = payload.substring(commaIndex + 1).toInt();

  while (xPixels != 0 || yPixels != 0) {
    int stepX = constrain(xPixels, -MAX_MOUSE_STEP, MAX_MOUSE_STEP);
    int stepY = constrain(yPixels, -MAX_MOUSE_STEP, MAX_MOUSE_STEP);
    Mouse.move(stepX, stepY, 0);
    xPixels -= stepX;
    yPixels -= stepY;
    delay(2);
  }
}

void fireWeapon() {
  Mouse.press(MOUSE_LEFT);
  delay(FIRE_MS);
  Mouse.release(MOUSE_LEFT);
}

void jump() {
  Keyboard.press(' ');
  delay(JUMP_MS);
  Keyboard.release(' ');
}

void releaseMoveKeys() {
  Keyboard.release('w');
  Keyboard.release('s');
  Keyboard.release('a');
  Keyboard.release('d');

  pressingForward = false;
  pressingBackward = false;
  pressingLeft = false;
  pressingRight = false;
}

void stopAll() {
  releaseMoveKeys();
  Keyboard.releaseAll();
}
