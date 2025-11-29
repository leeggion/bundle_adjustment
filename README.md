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
_temporary_
```
sudo apt-get install libceres-dev
```
Сборка:
```
mkdir build && cd build
cmake ..
cmake --build .
```
Запуск:
```
./bundle <path-to-dataset[BAL]> <LM/GN> <[optional]num_iterations>
# пример ./bundle ../data/ladybug.txt LM 200
```
Пример для создания датасета (выполнения работы SfM до BA) также в наличии и представлен в папке `test_images/test6`
```
cd src
./make_bal.sh
```
ИЛИ 
```
cd src
python ./make_bal.py
```
Все остальные датасеты можно найти на сайте [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/)