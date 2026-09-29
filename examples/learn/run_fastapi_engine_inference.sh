#!/usr/bin/env bash
# 第一行表示用 bash 执行本脚本；用法：在有 GPU 的机器上，从仓库根目录运行 bash examples/learn/run_fastapi_engine_inference.sh

# 任何一条命令失败，脚本就立刻停下，不再往下执行
set -e             
# 清掉机器上预设的 PYTHONPATH，免得其他目录里的包盖过虚拟环境里的包                  
unset PYTHONPATH
# 把 pip 的下载缓存（好几 GB）放到 /tmp                    
export PIP_CACHE_DIR=/tmp/pip-cache  

 # 在 /tmp/sgl-venv 建一个空的虚拟环境（系统 Python 缺 ensurepip，只能先不带 pip）
python3 -m venv --without-pip /tmp/sgl-venv       
# 用系统的 pip 给这个虚拟环境装上它自己的 pip
pip --python /tmp/sgl-venv/bin/python install pip
# 激活虚拟环境，之后的 python 和 pip 都指向它，不会动到系统 Python  
source /tmp/sgl-venv/bin/activate                  
# 按 docs/docs/get-started/install.mdx 的 CUDA 12 装法先装 CUDA 12.9 版的 torch（当前代码默认要 CUDA 13，即驱动 >= 580）
pip install torch==2.13.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129  
# 再装 CUDA 12.9 版的内核包，版本和仓库锁定的一致
pip install --no-deps sglang-kernel==0.4.6.post1 sgl-deep-gemm==0.1.5.post3 --index-url https://docs.sglang.ai/whl/cu129/ 
# 从源码可编辑安装当前仓库，改了代码立即生效，并跳过可选的 Rust 扩展 
SGLANG_BUILD_RUST_EXTS=none pip install -e python  

cd examples/runtime/engine  # 进入示例目录，因为示例启动服务时要从当前目录导入它自己

 # 运行示例；
 # 指定了模型
 # 指定了启动超时
python fastapi_engine_inference.py --model-path Qwen/Qwen2.5-0.5B-Instruct --startup-timeout 600 
