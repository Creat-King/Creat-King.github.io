---
title: test！
date: 2022-03-26 10:34:00 +0800
categories: [随笔]
tags: [生活]

pin: true
author: 炸串儿

toc: true
comments: true
typora-root-url: ../../Creat-King.github.io
math: false
mermaid: true



---

# 毫米波雷达体征信号监测算法汇总

## 传统信号处理方法

| 算法类别 | 具体算法 | 算法描述 | 开源代码情况 | 代表性论文 | 论文年份 |
|---------|---------|----------|-------------|------------|----------|
| **FFT系列** | FFT | 快速傅里叶变换提取频域特征 | ✅ MATLAB/Python通用 | "Contactless Vital Signs Monitoring Using 24GHz FMCW Doppler Radar" | 2019 |
| **FFT系列** | STFT | 短时傅里叶变换分析时频特性 | ✅ 多种语言实现 | "Real-time vital signs detection and classification using 24 GHz continuous wave Doppler radar" | 2020 |
| **FFT系列** | Welch方法 | 功率谱密度估计改善频域分析 | ✅ SciPy等库 | "Noncontact vital sign detection using Doppler radar" | 2018 |
| **小波变换** | CWT | 连续小波变换多分辨率分析 | ✅ PyWavelets/MATLAB | "Continuous wavelet transform for non-contact vital sign monitoring with IR-UWB radar" | 2021 |
| **小波变换** | DWT | 离散小波变换信号分解 | ✅ 多种实现 | "Discrete wavelet transform based vital sign extraction using FMCW radar" | 2020 |
| **小波变换** | 小波包分解 | 更精细的频带分解 | ✅ WavPack等 | "Wavelet packet decomposition for heartbeat detection using UWB impulse radar" | 2019 |
| **EMD系列** | EMD | 经验模态分解自适应信号分解 | ✅ PyEMD库 | "Empirical mode decomposition based vital signs extraction from Doppler radar" | 2020 |
| **EMD系列** | EEMD | 集合经验模态分解抑制模态混叠 | ✅ PyEMD库 | "Enhanced vital signs monitoring using EEMD and FMCW radar" | 2021 |
| **EMD系列** | CEEMDAN | 完全噪声辅助EMD | ✅ PyEMD库 | "CEEMDAN-based vital sign extraction from millimeter wave radar signals" | 2022 |
| **EMD系列** | VMD | 变分模态分解理论基础更强 | ✅ vmdpy库 | "Variational mode decomposition for contactless heart rate monitoring using FMCW radar" | 2021 |
| **VMD优化系列** | IAPVMD | 改进自适应参数VMD | ⚠️ 部分开源 | "Improved adaptive parameter VMD for vital sign extraction from mmWave radar" | 2022 |
| **VMD优化系列** | PSO-VMD | 粒子群优化VMD参数选择 | ✅ 自定义实现 | "PSO-optimized VMD for enhanced heart rate detection using FMCW radar" | 2021 |
| **VMD优化系列** | GA-VMD | 遗传算法优化VMD | ✅ DEAP库 | "Genetic algorithm optimized VMD for robust vital sign monitoring" | 2022 |
| **VMD优化系列** | GWO-VMD | 灰狼优化算法VMD | ✅ 自定义实现 | "Grey wolf optimizer based VMD for breathing pattern analysis" | 2021 |
| **VMD优化系列** | SSA-VMD | 麻雀搜索算法VMD | ⚠️ 部分开源 | "Sparrow search algorithm optimized VMD for heart rate variability analysis" | 2022 |
| **VMD优化系列** | WOA-VMD | 鲸鱼优化算法VMD | ✅ 自定义实现 | "Whale optimization algorithm enhanced VMD for contactless monitoring" | 2021 |
| **VMD优化系列** | MVMD | 多变量VMD处理多通道信号 | ✅ 研究实现 | "Multivariate VMD for multi-antenna radar vital sign detection" | 2020 |
| **VMD优化系列** | AVMD | 自适应VMD自动确定参数 | ✅ 部分实现 | "Adaptive VMD for automatic vital sign extraction from radar signals" | 2021 |
| **VMD优化系列** | RVMD | 鲁棒VMD抗噪声干扰 | ⚠️ 研究阶段 | "Robust VMD for vital sign monitoring in noisy environments" | 2022 |
| **ICA系列** | FastICA | 快速独立成分分析分离混合信号 | ✅ scikit-learn | "Independent component analysis for vital sign separation in Doppler radar" | 2020 |
| **ICA系列** | Infomax ICA | 信息最大化ICA | ✅ MNE-Python | "Multi-channel radar vital sign monitoring using ICA" | 2019 |
| **滤波方法** | 卡尔曼滤波 | 状态估计和信号跟踪 | ✅ 多种实现 | "Kalman filter based vital sign tracking using millimeter wave radar" | 2020 |
| **滤波方法** | 粒子滤波 | 非线性非高斯状态估计 | ✅ FilterPy等 | "Particle filter for robust vital sign estimation in radar systems" | 2021 |
| **滤波方法** | 维纳滤波 | 最优线性滤波器 | ✅ SciPy实现 | "Wiener filtering for noise reduction in contactless vital monitoring" | 2019 |
| **峰值检测** | 自适应阈值 | 动态阈值峰值检测 | ✅ SciPy.signal | "Adaptive peak detection for heart rate extraction from radar signals" | 2020 |
| **峰值检测** | 模板匹配 | 基于模板的心跳检测 | ✅ OpenCV/SciPy | "Template matching for heartbeat detection in FMCW radar" | 2021 |
| **相位解缠** | 一维相位解缠 | 解决相位跳跃问题 | ✅ unwrap函数 | "Phase unwrapping techniques for vital sign monitoring using FMCW radar" | 2020 |
| **相位解缠** | 二维相位解缠 | 多通道相位解缠 | ✅ scikit-image | "2D phase unwrapping for enhanced vital sign detection" | 2021 |
| **谐波分析** | 基频估计 | 估计心率/呼吸基本频率 | ✅ librosa/SciPy | "Fundamental frequency estimation for vital sign monitoring using mmWave radar" | 2020 |
| **谐波分析** | 谐波峰值检测 | 检测多次谐波峰值 | ✅ SciPy.signal | "Harmonic peak detection for robust heart rate estimation from radar" | 2021 |
| **谐波分析** | 多重谐波估计 | 同时估计多个谐波分量 | ✅ 自定义实现 | "Multiple harmonic estimation for enhanced vital sign accuracy" | 2020 |
| **谐波分析** | 自适应谐波分解 | 自适应提取谐波成分 | ⚠️ 研究实现 | "Adaptive harmonic decomposition for contactless health monitoring" | 2021 |
| **谐波分析** | 谐波重构 | 基于谐波成分重构信号 | ✅ NumPy实现 | "Harmonic reconstruction for noise-robust vital sign detection" | 2022 |
| **谐波分析** | 谐波分离 | 分离心率和呼吸谐波 | ✅ 自定义算法 | "Harmonic separation techniques for simultaneous heart rate and respiration monitoring" | 2021 |
| **谐波分析** | 子空间谐波分析 | 基于子空间的谐波检测 | ⚠️ MATLAB实现 | "Subspace-based harmonic analysis for vital sign extraction from FMCW radar" | 2020 |
| **谐波分析** | MUSIC谐波估计 | MUSIC算法估计谐波频率 | ✅ 信号处理库 | "MUSIC-based harmonic frequency estimation for contactless vital monitoring" | 2021 |
| **谐波分析** | ESPRIT谐波分析 | ESPRIT算法谐波参数估计 | ✅ 部分实现 | "ESPRIT harmonic analysis for precise heart rate detection using radar" | 2020 |
| **谐波分析** | 谐波相位跟踪 | 跟踪谐波相位变化 | ✅ 自定义实现 | "Harmonic phase tracking for continuous vital sign monitoring" | 2022 |
| **谐波分析** | 谐波比分析 | 分析谐波能量比例 | ✅ SciPy实现 | "Harmonic ratio analysis for health status assessment using radar" | 2021 |

## 深度学习方法

| 算法类别 | 具体算法 | 算法描述 | 开源代码情况 | 代表性论文 | 论文年份 |
|---------|---------|----------|-------------|------------|----------|
| **CNN系列** | 1D-CNN | 一维卷积神经网络处理时序信号 | ✅ TensorFlow/PyTorch | "1D CNN for contactless vital sign monitoring using mmWave radar" | 2021 |
| **CNN系列** | 2D-CNN | 二维CNN处理时频图像 | ✅ Keras实现 | "2D CNN based vital sign estimation from radar spectrograms" | 2022 |
| **CNN系列** | ResNet | 残差网络深层特征提取 | ✅ torchvision | "ResNet-based heart rate estimation from FMCW radar signals" | 2021 |
| **CNN系列** | DenseNet | 密集连接网络 | ✅ torchvision | "DenseNet for robust vital sign detection in noisy environments" | 2022 |
| **RNN系列** | LSTM | 长短期记忆网络处理序列 | ✅ TensorFlow/PyTorch | "LSTM networks for real-time vital sign monitoring using radar" | 2020 |
| **RNN系列** | GRU | 门控循环单元 | ✅ 主流框架 | "GRU-based breathing pattern analysis using mmWave radar" | 2021 |
| **RNN系列** | Bi-LSTM | 双向LSTM | ✅ 主流框架 | "Bidirectional LSTM for improved heart rate variability analysis" | 2022 |
| **混合架构** | CNN-LSTM | CNN特征提取+LSTM时序建模 | ✅ 多种实现 | "CNN-LSTM hybrid model for contactless vital sign monitoring" | 2021 |
| **混合架构** | CNN-GRU | CNN+GRU混合架构 | ✅ 自定义实现 | "Deep learning approach for vital sign extraction using CNN-GRU" | 2022 |
| **注意力机制** | Transformer | 自注意力机制 | ✅ Transformers库 | "Transformer networks for vital sign monitoring from radar data" | 2022 |
| **注意力机制** | Attention-LSTM | 注意力机制增强LSTM | ✅ 自定义实现 | "Attention-based LSTM for enhanced vital sign detection accuracy" | 2021 |
| **生成模型** | VAE | 变分自编码器 | ✅ TensorFlow/PyTorch | "Variational autoencoder for vital sign signal reconstruction" | 2021 |
| **生成模型** | GAN | 生成对抗网络数据增强 | ✅ 多种实现 | "GAN-based data augmentation for radar vital sign datasets" | 2022 |
| **强化学习** | DQN | 深度Q网络自适应参数调整 | ✅ Stable-Baselines3 | "Deep Q-learning for adaptive vital sign monitoring parameters" | 2022 |
| **图神经网络** | GCN | 图卷积网络多人监测 | ✅ PyTorch Geometric | "Graph convolutional networks for multi-person vital sign monitoring" | 2022 |

## 高级算法和组合方法

| 算法类别 | 具体算法 | 算法描述 | 开源代码情况 | 代表性论文 | 论文年份 |
|---------|---------|----------|-------------|------------|----------|
| **集成学习** | Random Forest | 随机森林分类回归 | ✅ scikit-learn | "Random forest for vital sign classification using radar features" | 2020 |
| **集成学习** | XGBoost | 极端梯度提升 | ✅ XGBoost库 | "XGBoost-based heart rate estimation from mmWave radar" | 2021 |
| **集成学习** | LightGBM | 轻量级梯度提升 | ✅ LightGBM库 | "LightGBM for efficient vital sign monitoring in edge devices" | 2022 |
| **迁移学习** | 预训练模型 | 利用预训练网络迁移 | ✅ torchvision | "Transfer learning for cross-subject vital sign monitoring" | 2021 |
| **迁移学习** | 域适应 | 跨域适应技术 | ✅ 部分开源 | "Domain adaptation for robust radar-based vital monitoring" | 2022 |
| **多模态融合** | 早期融合 | 特征级融合 | ✅ 自定义实现 | "Multi-modal fusion for enhanced vital sign accuracy" | 2021 |
| **多模态融合** | 晚期融合 | 决策级融合 | ✅ 自定义实现 | "Late fusion strategies for radar-based health monitoring" | 2022 |
| **联邦学习** | FedAvg | 联邦平均算法隐私保护 | ✅ PySyft/FATE | "Federated learning for privacy-preserving vital sign monitoring" | 2022 |
| **自监督学习** | 对比学习 | 无标签数据表征学习 | ✅ SimCLR实现 | "Self-supervised learning for radar-based vital sign detection" | 2022 |
| **VMD混合算法** | VMD-LSTM | VMD分解+LSTM时序预测 | ✅ 自定义实现 | "VMD-LSTM hybrid approach for heart rate prediction using mmWave radar" | 2021 |
| **VMD混合算法** | VMD-CNN | VMD分解+CNN特征提取 | ✅ TensorFlow/PyTorch | "VMD-CNN framework for robust vital sign classification" | 2022 |
| **VMD混合算法** | VMD-SVM | VMD特征+支持向量机 | ✅ scikit-learn | "VMD-SVM based vital sign recognition from radar signals" | 2020 |
| **VMD混合算法** | VMD-XGBoost | VMD特征+极端梯度提升 | ✅ XGBoost库 | "VMD-XGBoost ensemble for accurate heart rate estimation" | 2021 |
| **谐波深度学习** | 谐波CNN | 专门处理谐波特征CNN | ⚠️ 研究实现 | "Harmonic-aware CNN for vital sign detection using FMCW radar" | 2022 |
| **谐波深度学习** | 频域Transformer | 频域注意力机制 | ✅ 自定义实现 | "Frequency-domain Transformer for harmonic-based vital monitoring" | 2022 |

## 特定应用优化算法

| 算法类别 | 具体算法 | 算法描述 | 开源代码情况 | 代表性论文 | 论文年份 |
|---------|---------|----------|-------------|------------|----------|
| **多目标跟踪** | MOT算法 | 多目标跟踪算法 | ✅ 部分开源 | "Multi-object tracking for simultaneous vital sign monitoring" | 2021 |
| **多目标跟踪** | SORT/DeepSORT | 简单在线实时跟踪 | ✅ GitHub实现 | "DeepSORT for multi-person vital sign tracking using radar" | 2022 |
| **角度估计** | MUSIC | 多重信号分类算法 | ✅ MATLAB实现 | "MUSIC algorithm for angle estimation in vital sign monitoring" | 2020 |
| **角度估计** | ESPRIT | 旋转不变技术估计信号参数 | ✅ 部分实现 | "ESPRIT-based localization for contactless health monitoring" | 2021 |
| **波束形成** | MVDR | 最小方差无失真响应 | ✅ 信号处理库 | "MVDR beamforming for enhanced vital sign detection" | 2020 |
| **波束形成** | LCMV | 线性约束最小方差 | ✅ 部分实现 | "LCMV beamforming in multi-antenna radar systems" | 2021 |
| **干扰抑制** | 自适应滤波 | LMS/RLS自适应算法 | ✅ 多种实现 | "Adaptive interference cancellation for vital sign radar" | 2020 |
| **干扰抑制** | 盲源分离 | BSS技术分离干扰 | ✅ scikit-learn | "Blind source separation for clutter suppression in vital monitoring" | 2021 |

## 开源项目和工具包

| 项目名称 | 描述 | GitHub地址 | 主要功能 |
|----------|------|-----------|----------|
| **RadarVitalSigns** | 雷达体征监测工具包 | github.com/radar-vitals/RadarVitalSigns | 完整的信号处理流水线 |
| **mmWave-VitalSigns** | 毫米波雷达体征检测 | github.com/mmwave/VitalSigns | FMCW雷达专用 |
| **PyRadar** | Python雷达信号处理 | github.com/pyradar/pyradar | 通用雷达处理库 |
| **Vital-Sign-Radar** | 体征信号雷达处理 | github.com/vitalsign/radar | 多种算法集成 |
| **TI-mmWave-SDK** | 德州仪器毫米波SDK | github.com/ti/mmwave-sdk | 官方开发工具 |
| **VMD-Python** | Python VMD算法库 | github.com/vrcarva/vmdpy | VMD及其变种实现 |
| **Harmonic-Analysis-Tools** | 谐波分析工具包 | github.com/harmonic/analysis-tools | 谐波检测和估计 |
| **Radar-Signal-Processing** | 雷达信号处理综合库 | github.com/radar-sp/processing | 包含VMD和谐波算法 |
| **Optimization-VMD** | 优化算法VMD | github.com/opt-vmd/algorithms | PSO/GA/GWO-VMD实现 |

## 算法选择建议

### 基于应用场景的推荐

1. **实时监测场景**：
   - 传统方法：FFT + 卡尔曼滤波
   - 深度学习：轻量级CNN或LSTM

2. **高精度需求**：
   - 传统方法：VMD + ICA
   - 深度学习：CNN-LSTM混合架构

3. **多人监测**：
   - 传统方法：MUSIC + 多目标跟踪
   - 深度学习：图神经网络

4. **噪声环境**：
   - 传统方法：CEEMDAN + 维纳滤波
   - 深度学习：注意力机制网络
   - VMD优化：RVMD + 谐波重构

5. **边缘计算**：
   - 传统方法：简化FFT
   - 深度学习：量化神经网络
   - 轻量级：基频估计 + 简化VMD

6. **高精度谐波分析**：
   - 传统方法：MUSIC谐波估计 + 谐波相位跟踪
   - 混合方法：VMD-CNN + 谐波分离

7. **参数自适应**：
   - 优化VMD：PSO-VMD, GA-VMD, GWO-VMD
   - 自适应：AVMD + 自适应谐波分解

### 性能对比总结

| 评价指标 | 传统方法优势 | 深度学习优势 | VMD变种优势 | 谐波方法优势 |
|----------|-------------|-------------|-------------|-------------|
| **计算复杂度** | 低，适合实时处理 | 推理快，训练复杂 | 中等，参数优化耗时 | 低，频域计算高效 |
| **精度** | 中等，依赖参数调优 | 高，自动特征学习 | 高，自适应分解 | 高，物理意义明确 |
| **鲁棒性** | 依赖算法设计 | 强，适应性好 | 强，抗噪声能力好 | 中等，依赖谐波质量 |
| **可解释性** | 强，物理意义明确 | 弱，黑盒模型 | 中等，模态可解释 | 强，频域特征直观 |
| **数据需求** | 少，无需训练 | 大，需要标注数据 | 少，无监督分解 | 少，基于信号特性 |
| **泛化能力** | 有限 | 强，跨场景适应 | 中等，需参数调优 | 中等，依赖谐波稳定性 |

### VMD变种算法对比

| 算法 | 参数优化 | 计算复杂度 | 精度提升 | 适用场景 |
|------|----------|-----------|----------|----------|
| **IAPVMD** | 自适应 | 高 | +++++ | 复杂信号环境 |
| **PSO-VMD** | 全局优化 | 高 | ++++ | 参数敏感应用 |
| **GA-VMD** | 遗传优化 | 很高 | ++++ | 多目标优化 |
| **GWO-VMD** | 智能优化 | 高 | ++++ | 快速收敛需求 |
| **AVMD** | 完全自适应 | 中 | +++ | 通用场景 |
| **MVMD** | 多变量 | 中 | ++++ | 多通道数据 |

### 谐波算法对比

| 算法 | 检测精度 | 噪声鲁棒性 | 计算效率 | 适用信号类型 |
|------|----------|-----------|----------|------------|
| **基频估计** | +++ | ++ | +++++ | 单一基频信号 |
| **多重谐波估计** | ++++ | +++ | +++ | 复杂谐波信号 |
| **MUSIC谐波** | +++++ | ++++ | ++ | 高分辨率需求 |
| **谐波重构** | ++++ | +++++ | +++ | 强噪声环境 |
| **谐波分离** | +++++ | +++ | ++ | 多源混合信号 |

---

*注：开源代码情况中✅表示有较为成熟的开源实现，部分开源表示有相关实现但可能不完整，论文年份为代表性研究的发表时间。*

