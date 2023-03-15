import setuptools

install_requires = []

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


# no need to do fancy stuff so far
packages = setuptools.find_packages()


install_requires.extend([
    'onnx',
])

setuptools.setup(
    name = 'onnx2caffe', 
    version = '0.0.1', 
    author = 'chenjun', 
    author_email = 'chenjun_csu@163.com', 
    description='export onnx to caffe', 
    long_description=readme(), 
    keywords='ONNX caffe', 
    url='https://github.com/chenjun2hao/onnx2caffe',
    packages=packages,
    install_requires=install_requires,
    classifiers=[ 
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'onnx2caffe=onnx2caffe:main',
        ],
    },
    license='Apache License 2.0', 
)