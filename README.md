 这个项目是我用花半年时间用手机写出来的，简单介绍一下我和项目的背景：小学的时候就沉迷Minecraft，初中的时候开始在YT看洋人的视频自学python，可惜时间有限，只能学到一些杂碎的知识，并且个人更偏向喜欢折腾系统，以至于在初中和python接触的时间非常少，后来电脑还被折腾坏了，只能在手机上勉强自学编程，termux对python的兼容太雷人，我只能用colab和kaggle运行python环境，后来就慢慢爱上了机器学习。你可以点我头像看我的linuxdo主页，你会看到我去年发布了一些白嫖云电脑部署开源模型的教程，比如flux.1和cogvideo还有videoretalking，但是这些都是已经训练好的模型，后来我玩mc的时候心血来潮，想训练一个自己的mc模型，实现一句话让ai生成我的世界建筑物。但是想要自己一个人从0开始训练一个ai模型，哪是一件容易的事，模型训练又不是请客吃饭。我只能到网上到处征集资料，我阅读了大量的计算机科学论文，想着有没有成功案例可以参考，不幸的是一个也没有找到，网上只有文本到3d点云的ai模型，而训练一个我的世界生成建筑物的模型，应该要训练一个文本到体素模型的模型，和网上的ai生成3d，根本就不是同一个领域，于是我就请教了gpt4o和claude，它们结合我的想法帮我写一个不完整的框架，我就复制粘贴到colab测试了起来，结果遇到各种报错，我就把报错的内容复制粘贴发给gpt，有很多错误ai都不能解决（比如很多库都更新了，ai还在用旧版库的模块，还反过来说我的库太旧了让我更新，或者是因为模型幻觉乱定义函数，或者就是colab环境本身自带很多库，如果我安装新的库会导致和一些旧的库冲突等等），困扰了我很久，长夜难眠，但是这没阻止我要写好项目的决心，我榨干了自己的每一分每一秒时间投了到了这上面，一个一个错误被解决了，一个月过去了，自己的项目终于可以训练了，这个时候，自己才想起来，自己压根没有数据集。怎么办？自己去mc造10万个房子拿去训练吗，想想都觉得不现实，那就让ai生成建筑物，然后去训练呗？问题是ai会生成建筑物，我还需要训练ai模型干嘛，于是项目到这里就碰壁了，我重新整机资料，让gemini新发布的thinking模型和linuxdo佬友"ky"的deepseek-r1-lite模型帮我想了一个新思路，不过当时deepseek-r1还没发布正式版，lite版本的上下文非常有限，gemini的thinking模型的上下文也只有30ktokens左右，而我准备的资料（比如自己在github整理的关心ai和mc相关的论文和代码以及一些冷门的py库的源代码，比如nbt）的tokens占用远远超过这个，我只能疯狂压缩资料的内容，或者问ai一些零零散散的问题，最终ai建议我去微调closeAI在23年发布的shap-e模型，这是一个轻量的文本生成3d模型的模型，15g显存就可以运行，虽然质量比较堪忧，但是最终还是要转换成体素模型，所以不必在乎太多质量。首先冻结大部分层，保留原本的nlp，然后拿自己的数据集微调训练，既然是微调，数据集少一点也没关系，我就拿自己昔日的mc建筑地图里的建筑物导出成了schematic文件，用nbt库解析成了文本，然后用groq生成描述，一顿操作猛如虎，又是一个多月过去了，数据集有了，首先要把描述文本转成token张量，也就是微调nlp的tokenids，结果一上来就遇到了形状不匹配还是什么报错去了，我忘了，我记得shape用的nlp好像是clip，对于微调文本处理的报错，我又研究了几天，终端终于没有出现报错日志，我都快感动哭了，接下来就是测试微调我的世界schematic文件了，一上来，终端就送了我一份报错大礼，它说形状和结构不匹配，它说我的结构不统一，然后我问了ai，ai建议我把我的mc建筑物结构都裁剪成16x16x16，于是我只好重新处理数据集，又浪费了一天时间，重新处理好数据集后终端又报错，说我的数据集形状与shap-e模型的形状不匹配，到这一步，ai也不知道怎么办了，gemini2.0flash thinking, deepseek r1-lite，o1-preview，3-5天都没帮我解决，项目再一次碰壁了，过了几天，我发现网上就有一个开源的ai生成我的世界schematic的模型，但是没有集成nlp，也就没法用文本生成schematic，而是ai自己发挥，通过空间随机生成，我要做的是给他安装一个文本处理nlp就可以实现文本生成我的世界schematic了，乍一听还挺容易，结果我才发现这个模型是11×11×11的建筑结构训练来的，而且零样本推理的质量非常差，一个月过去了，各种错误和维度不匹配，形状不匹配，全部被我解决了，终于是硬着头皮写完了微调训练脚本，感觉就要迎来胜利的曙光，结果到了推理的时候，发现根本就不听从文本的指示，生成出来的东西完全没法看，不知道是数据集不够，还是推理脚本有问题，还是训练脚本有问题，自己打算放弃了。过了几天，我重新整理了一下思路，发现自己把事情想得太复杂了，其实这种东西，只要把普通的3d模型转换成体素模型，然后把每一个体素模型的方块都换成mc的方块不就好了吗。
 然而，想象是美好的，现实是残酷的，才怪，因为网上有很多现成的开源文本到3d的模型，比如hunyuan3d，可以生成非常高质量的模型，它的原理就会首先通过ai生成一张图片，然后根据图片生成3d模型，最后给3d模型生成纹理。我要做的就是用python写一个映射软件，并且使用ai的function calling来实现ai智能映射，gemini的多模态功能非常强悍，识图功能遥遥领先，而且ai studio的Gemini还免费，用来ai映射实在是太合适不过了。
 但是这一条龙下来，少说都得24g显存，考虑到也不是人人都有大显存的显卡，我决定弄出一个长期免费白嫖的方案，比如在免费的云电脑上面运行。但是colab只有16g不到，而kaggle也只有16g显存，hunyuan3d-2的官方文档说至少要25g显存，于是我决定部署一个优化版本的hunyuan3d，它的原理其实就是把显存的压力转移到了内存上，只要内存足够就可以了，而colab只有12g内存，比显存还少，于是我首先写了一个启智社区版本的项目，后续会上传更多免费的平台方案。
 
简洁的简介：通过自然语言生成Minecraft建筑！ 本工具链实现从文本描述→AI生成3D模型→自动体素化→Minecraft Schematic文件的全流程自动化。

 核心亮点
- 多模态AI集成：融合Hunyuan3D-2图像生成与Gemini语言模型
- 工业级优化：支持Octree体素化（最高32x32x32分辨率）
- 智能材质映射：基于python的高级颜色匹配算法
- 多端兼容：输出标准Schematic格式，兼容Java版我的世界
白嫖云端显卡一键部署（推荐）
1.启智社区免费算力平台（支持手机访问）
没有账号先注册：
（带aff）
注册地址：https://openi.pcl.ac.cn/user/sign_up?sharedUser=NewestAI
#AFF 
每天至少可以免费用5个小时，如果运行不超过30分钟就关机不计时长，相当于无限白嫖，并且数据可以保留，比colab强太多。
首先访问我的项目地址：
https://openi.pcl.ac.cn/NewestAI/Mine-Builder
然后下滑找到一键运行云脑任务模板
![IMG_20250322_235033|690x247](upload://t0blvp4Lq3i7yFUgIoLYFf3QLrg.jpeg)
点击运行
下滑找到新建任务
![Screenshot_2025-03-22-23-46-43-130_com.android.chrome|224x500](upload://spbMUAzVCWjgPqd918Od40qrRbn.jpeg)
单击新建任务后等待机器开机就可以调试了
![IMG_20250323_103856|690x344](upload://tlDtyn3vI8FRyIHwFKorAnmXssS.jpeg)
在调试界面打开terminal
![IMG_20250323_110801|542x500](upload://byjCwOA97KOXQntQotyfiOfNDFJ.jpeg)
输入source install.sh并回车就可以一键自动部署了，第一次部署耗时比较久，之后就很快了，如果途中终端莫名其妙消失了，重新打开一次终端再运行就好了。
本地部署（不推荐）
bash
git clone https://github.com/nianxi666/Mine-Builder
cd Mine-Builder
pip install -r requirements.txt
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
使用教程


 功能特性
 核心流程
1. 文本到图像生成  
   - 支持HunyuanDiT/Stable Diffusion（暂不）多模型切换

2. 3D建模优化  
   python
    包含的优化处理器
   - FloaterRemover()   浮点消除
   - DegenerateFaceRemover()   畸形面修复
   - FaceReducer()   多边形精简

3. 智能材质系统
   json
   "block_mapping": {
     "stone": "minecraft:stone", "minecraft:cobblestone",
     "glass": "minecraft:stained_glass:15", "minecraft:glass"
   }

 本地安装（不推荐）
 系统要求
- NVIDIA GPU (推荐RTX 2060)
- CUDA 11.8
- RAM ≥25GB(windows)
- 云端部署（推荐）
- 
