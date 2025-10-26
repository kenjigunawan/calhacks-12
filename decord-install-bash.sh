sudo add-apt-repository ppa:jonathonf/ffmpeg-4 
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-setuptools make cmake
sudo apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
git clone --recursive https://github.com/dmlc/decord
sudo apt-get install -y libnvidia-decode-550
sudo apt-get install -y build-essential cmake make pkg-config python3-dev python3-setuptools \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libavfilter-dev
cd Decord
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make
python3 setup.py install --user


