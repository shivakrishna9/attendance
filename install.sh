# add-apt-repository ppa:webupd8team/sublime-text-3
# apt-get update
# apt-get install -y sublime-text-installer

# sudo apt-get install -y build-essential cmake git pkg-config
# sudo apt-get install -y libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
# sudo apt-get install -y libgtk2.0-dev
# sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
# sudo apt-get install -y libatlas-base-dev gfortran
# sudo apt-get install -y python2.7-dev

# git clone https://github.com/Itseez/opencv.git
# cd opencv
# git checkout 3.1.0

# cd ~
# git clone https://github.com/Itseez/opencv_contrib.git
# cd opencv_contrib
# git checkout 3.1.0

# cd ~/opencv
# mkdir build
# cd build
# cmake -D CMAKE_BUILD_TYPE=RELEASE \
# 	-D CMAKE_INSTALL_PREFIX=/usr/local \
# 	-D INSTALL_C_EXAMPLES=ON \
# 	-D INSTALL_PYTHON_EXAMPLES=ON \
# 	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
# 	-D BUILD_EXAMPLES=ON ..
# make -j4
# sudo make install
# sudo ldconfig


# KEEP UBUNTU OR DEBIAN UP TO DATE

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y autoremove


# INSTALL THE DEPENDENCIES

# Build tools:
sudo apt-get install -y build-essential cmake

# GUI (if you want to use GTK instead of Qt, replace 'qt5-default' with 'libgtkglext1-dev' and remove '-DWITH_QT=ON' option in CMake):
sudo apt-get install -y qt5-default libvtk6-dev

# Media I/O:
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev

# Video I/O:
sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev

# Parallelism and linear algebra libraries:
sudo apt-get install -y libtbb-dev libeigen3-dev

# Python:
sudo apt-get install -y python-dev python-tk python-numpy python3-dev python3-tk python3-numpy

# Java:
sudo apt-get install -y ant default-jdk

# Documentation:
sudo apt-get install -y doxygen


# INSTALL THE LIBRARY (YOU CAN CHANGE '3.1.0' FOR THE LAST STABLE VERSION)

sudo apt-get install -y unzip wget
wget https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip 3.1.0.zip
rm 3.1.0.zip
mv opencv-3.1.0 OpenCV
cd OpenCV
mkdir build
cd build
cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON ..
make -j4
sudo make install
sudo ldconfig

