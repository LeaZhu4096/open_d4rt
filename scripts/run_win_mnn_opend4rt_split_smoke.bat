@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "MNN_ROOT=%REPO_ROOT%\..\d4rt-pytorch\third_party\MNN"
set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "ENCODER_PATH=%~1"
set "DECODER_PATH=%~2"
set "NUM_FRAMES=%~3"
set "HEIGHT=%~4"
set "WIDTH=%~5"
set "NUM_QUERIES=%~6"
set "MEMORY_TOKENS=%~7"
set "HIDDEN_DIM=%~8"
if "%ENCODER_PATH%"=="" set "ENCODER_PATH=%REPO_ROOT%\artifacts\mnn_vulkan\opend4rt_32clip_encoder_t2_32.mnn"
if "%DECODER_PATH%"=="" set "DECODER_PATH=%REPO_ROOT%\artifacts\mnn_vulkan\opend4rt_32clip_decoder_mem5_q8.mnn"
if "%NUM_FRAMES%"=="" set "NUM_FRAMES=2"
if "%HEIGHT%"=="" set "HEIGHT=32"
if "%WIDTH%"=="" set "WIDTH=32"
if "%NUM_QUERIES%"=="" set "NUM_QUERIES=8"
if "%MEMORY_TOKENS%"=="" set "MEMORY_TOKENS=5"
if "%HIDDEN_DIM%"=="" set "HIDDEN_DIM=1280"

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

if not exist "%REPO_ROOT%\artifacts\mnn_vulkan" (
    mkdir "%REPO_ROOT%\artifacts\mnn_vulkan"
)

cl /nologo /EHsc /std:c++17 ^
    /I"%MNN_ROOT%\include" ^
    /I"%MNN_ROOT%\source" ^
    /Fo:"%REPO_ROOT%\artifacts\mnn_vulkan\mnn_opend4rt_split_smoke.obj" ^
    "%REPO_ROOT%\scripts\mnn_opend4rt_split_smoke.cpp" ^
    /Fe:"%REPO_ROOT%\artifacts\mnn_vulkan\mnn_opend4rt_split_smoke.exe" ^
    /link /LIBPATH:"%MNN_ROOT%\build_win_vulkan" MNN.lib
if errorlevel 1 exit /b 1

set "PATH=%MNN_ROOT%\build_win_vulkan;%PATH%"

echo === OpenD4RT split CPU control ===
"%REPO_ROOT%\artifacts\mnn_vulkan\mnn_opend4rt_split_smoke.exe" "%ENCODER_PATH%" "%DECODER_PATH%" 0 %NUM_FRAMES% %HEIGHT% %WIDTH% %NUM_QUERIES% %MEMORY_TOKENS% %HIDDEN_DIM%
if errorlevel 1 exit /b 1

echo === OpenD4RT split Vulkan iGPU ===
"%REPO_ROOT%\artifacts\mnn_vulkan\mnn_opend4rt_split_smoke.exe" "%ENCODER_PATH%" "%DECODER_PATH%" 7 %NUM_FRAMES% %HEIGHT% %WIDTH% %NUM_QUERIES% %MEMORY_TOKENS% %HIDDEN_DIM%
if errorlevel 1 exit /b 1

endlocal