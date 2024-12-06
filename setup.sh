# conda remove -n ual_bench --all -y
conda create -n ual_bench python=3.10 -y
conda activate ual_bench
pip install decord
pip install omegaconf
pip install iopath
pip install opencv-python
pip install webdataset
pip install ftfy
pip install pytorchvideo
conda install conda-forge::ffmpeg
#transformers: v4.47
pip install 'accelerate>=0.26.0'
# pip install transformers
pip install joblib
pip install pandas
pip install timm
pip install einops
pip install 'torchvision==0.16.2'
pip install torchaudio==2.1.2
pip install sentencepiece
pip install numpy==1.26.3
pip install  transformers==4.47.0
