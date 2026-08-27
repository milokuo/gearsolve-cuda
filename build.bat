@echo off
setlocal
rem VS2022 host compiler -- CUDA 12.9 rejects newer MSVC toolsets.
set "VSDEV=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VSDEV%" ( echo VS2022 Community not found, edit build.bat & exit /b 1 )
call "%VSDEV%" >nul
if not exist build mkdir build

echo == reference.exe
cl /nologo /O2 /EHsc /std:c++17 /openmp /Fo:build\ /Fe:build\reference.exe src\reference.cpp || exit /b 1

echo == gpu_search.exe
nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "/EHsc /O2" src\gpu_search.cu -o build\gpu_search.exe || exit /b 1

echo done.
