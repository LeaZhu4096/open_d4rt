@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "MNN_ROOT=D:\d4rt_mnn-vulkan\third_party\MNN"
set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VS_VCVARS%" (
    set "VS_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist "%VS_VCVARS%" (
    echo Could not find vcvars64.bat
    exit /b 1
)
if not exist "%MNN_ROOT%\build_win_vulkan\MNN.lib" (
    echo Could not find MNN.lib at "%MNN_ROOT%\build_win_vulkan\MNN.lib"
    exit /b 1
)

call "%VS_VCVARS%"
if errorlevel 1 exit /b 1

if not exist "%REPO_ROOT%\artifacts\mnn_vulkan" mkdir "%REPO_ROOT%\artifacts\mnn_vulkan"

cl /nologo /EHsc /std:c++17 ^
    /I"%MNN_ROOT%\include" ^
    /I"%MNN_ROOT%\source" ^
    /Fo:"%REPO_ROOT%\artifacts\mnn_vulkan\mnn_opend4rt_video_infer.obj" ^
    "%REPO_ROOT%\scripts\mnn_opend4rt_video_infer.cpp" ^
    /Fe:"%REPO_ROOT%\artifacts\mnn_vulkan\mnn_opend4rt_video_infer.exe" ^
    /link /LIBPATH:"%MNN_ROOT%\build_win_vulkan" MNN.lib
if errorlevel 1 exit /b 1

copy /Y "%MNN_ROOT%\build_win_vulkan\MNN.dll" "%REPO_ROOT%\artifacts\mnn_vulkan\MNN.dll" >nul

echo Built "%REPO_ROOT%\artifacts\mnn_vulkan\mnn_opend4rt_video_infer.exe"
endlocal
