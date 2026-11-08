"""GPU 方案的转写流程（FSMN VAD + Paraformer + CT 标点、DiariZen、CAM++、条件 MossFormer2），要 NVIDIA GPU。

纯逻辑模块（sentences、assign、label、render、vec、gpu 选卡、models 的下载计划）只用标准库，可在任何 Python 3.10+ 上跑单测；
模型相关模块只在 engine/setup_gpu.sh 建的 Python 环境里导入。
"""
