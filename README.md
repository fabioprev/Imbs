Imbs
=========

# Requirements

Imbs requires the following packages to build:
  
  * build-essential
  * g++
  * cmake
  * libxml2
  * opencv

On Xubuntu (Ubuntu) 14.04 LTS (kernel 3.13.0-37) and later versions, these dependencies are
resolved by installing the following packages:
  
  - build-essential
  - cmake
  - libxml2
  - libxml2-dev
  - libopencv-dev

# How to build

The only development platform is Linux. We recommend a so-called out of source
build which can be achieved by the following command sequence:
  
  - mkdir build
  - cd build
  - cmake ../src
  - make -j\<number-of-cores+1\>

# Installation

Once the build phase has been successfully, the library have to be installed
so that it can be linked in other projects. This is achieved by the
following command sequence:
  
  - cd build
  - sudo make install

After the installation the header files have been copied to /usr/local/include/Imbs,
while the libimbs.so shared object has been copied to /usr/local/lib/Imbs.

The last step before correctly using the library is to logout because the file ~/.profile
has been modified by the installation.
