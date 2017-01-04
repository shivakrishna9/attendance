pip install pip -U
pip install numpy -U
pip install pandas -U
pip install h5py -U
pip install theano
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py2-none-any.whl
pip install $TF_BINARY_URL
pip install keras