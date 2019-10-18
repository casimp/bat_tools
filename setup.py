from setuptools import setup

setup(
    name='bat_tools',
    version='0.1.0',
    author='C. Simpson',
    author_email='c.a.simpson01@gmail.com',
    packages=['bat_tools'],
    include_package_data=True,
    url='https://github.com/casimp/bat_tools',
    download_url = 'https://github.com/casimp/bat_tools/tarball/v0.1.0',
    license='LICENSE.txt',
    description='Tools for bat data pre-processing.',
    keywords = ['bat', 'ul;trasound', 'pip'],
#    long_description=open('description').read(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows"]
)
