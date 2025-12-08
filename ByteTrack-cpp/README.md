



1.编译 win64 程序

clion 直接选择 mingw 编译工具链，target 选择 bytetrack_jni 即可

CMake => CMake Optinoes：
-DCMAKE_BUILD_TYPE=Release -DBUILD_JNI_4_WIN=ON

输出文件：
libbytetrack_jni.dll

2.编译 android 程序

直接 wsl 中进行编译
