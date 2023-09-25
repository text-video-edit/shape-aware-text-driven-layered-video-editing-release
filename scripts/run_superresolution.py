import os, sys
from path import Path
input_img = Path(sys.argv[1]).abspath()
output_dir = os.path.split(input_img)[0]
os.chdir('Real-ESRGAN')
os.system(f'python inference_realesrgan.py \
            -n RealESRGAN_x4plus \
            -i {input_img} \
            --output {output_dir} \
            --outscale 3.5 --suffix ""')
