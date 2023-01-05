from setuptools import setup

setup(
    name='wfd',
    version='0.2.0',    
    description='Wave Function Diffusion',
    url='https://github.com/wdcqc/WaveFunctionDiffusion',
    author='wdcqc',
    author_email='119406052+wdcqc@users.noreply.github.com',
    license='creativeml-openrail-m',
    packages=['wfd'],
    install_requires=[
        "accelerate",
        "bitsandbytes",
        "diffusers",
        "doki_theme_jupyter",
        "einops",
        "gradio",
        "huggingface_hub",
        "numpy",
        "opencv_python",
        "packaging",
        "Pillow",
        "torch",
        "torchinfo",
        "torchvision",
        "tqdm",
        "transformers"
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Creative OpenRAIL-M',  
        'Operating System :: POSIX :: Linux',       
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 4.0',
    ],
)
