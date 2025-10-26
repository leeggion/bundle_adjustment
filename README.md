# Bundle Adjustment на C++

## **Цель:** 

Реализовать с нуля алгоритм Bundle Adjustment (BA) и сравнить эффективность методов Гаусса-Ньютона (GN) и Левенберга-Марквардта (LM) на различных наборах данных.

## **Необходимые зависимости (под Linux, с самого нуля)**
```
sudo apt update
sudo apt install -y build-essential cmake pkg-config git
```
Проверка:
```
g++ --version    
cmake --version
```
OpenCV:
```
sudo apt update
sudo apt install -y libeigen3-dev
sudo apt install -y libopencv-dev
```
Сборка: (прототип)
```
mkdir build && cd build
cmake ..
cmake --build .
./bundle
```