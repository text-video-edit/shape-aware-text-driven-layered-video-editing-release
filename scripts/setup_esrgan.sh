# Super-resolution (Real-ESRGAN)
cd Real-ESRGAN
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
# run setup according to the repo's instructions
pip install basicsr
pipt install facexlib
pip install gfpgan
pip install -r requirements
python setup.py develop
