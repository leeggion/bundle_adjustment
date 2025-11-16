cmake_minimum_required(VERSION 3.10)
project(BundleAdjustment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- 1. Находим зависимости (старый стиль для OpenCV) ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui) # Оставляем как есть
find_package(Ceres REQUIRED)

# --- 2. Создаем исполняемый файл ---
add_executable(bundle
    src/main.cpp
    src/utilities/DatasetLoader.cpp
    src/utilities/math.cpp
    src/utilities/Visualize.cpp
)

# --- 3. Настраиваем пути к заголовкам (старый стиль) ---
# Мы добавляем 'src' (для ваших .hpp)
# И ${OpenCV_INCLUDE_DIRS} (переменная из старого стиля)
target_include_directories(bundle PUBLIC 
    src
    ${OpenCV_INCLUDE_DIRS} 
)

# --- 4. Линкуем библиотеки (смешанный стиль) ---
# Мы используем современные цели для Ceres и Eigen,
# но ${OpenCV_LIBS} (переменная из старого стиля) для OpenCV
target_link_libraries(bundle PUBLIC
    Ceres::ceres
    Eigen3::Eigen
    ${OpenCV_LIBS}
)