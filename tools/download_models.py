import os
from pathlib import Path
import requests

# 定义从 Hugging Face 下载模型的基础链接
RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/"

# 获取脚本所在目录的父目录作为基础路径
BASE_DIR = Path(__file__).resolve().parent.parent


# 下载模型的函数
def dl_model(link, model_name, dir_name):
    """
    通过 HTTP 请求下载模型，并保存到指定目录
    :param link: 模型文件的基础下载链接
    :param model_name: 模型文件的名字
    :param dir_name: 模型保存的目录
    """
    # 发送 GET 请求下载模型文件
    with requests.get(f"{link}{model_name}") as r:
        print("Downloading start1...")
        # 如果响应状态码不是 200，会抛出异常
        r.raise_for_status()

        print("Downloading start2...")
        # 创建目标文件夹（如果不存在的话）
        os.makedirs(os.path.dirname(dir_name / model_name), exist_ok=True)

        print("Downloading start3...")
        # 打开本地文件并将内容写入
        with open(dir_name / model_name, "wb") as f:
            # 按块读取内容并写入文件
            print("Downloading start11...")
            for chunk in r.iter_content(chunk_size=8192):
                print("Downloading start222...")
                f.write(chunk)
                print("Downloading start333...")
    print("Downloading 4444...")


if __name__ == "__main__":
    # 下载 hubert_base.pt 模型
    print("Downloading hubert_base.pt...")
    dl_model(RVC_DOWNLOAD_LINK, "hubert_base.pt", BASE_DIR / "assets/hubert")

    # 下载 rmvpe.pt 模型
    print("Downloading rmvpe.pt...")
    dl_model(RVC_DOWNLOAD_LINK, "rmvpe.pt", BASE_DIR / "assets/rmvpe")

    # 下载 vocals.onnx 模型（uvr5_weights 相关）
    print("Downloading vocals.onnx...")
    dl_model(
        RVC_DOWNLOAD_LINK + "uvr5_weights/onnx_dereverb_By_FoxJoy/",  # 完整的下载路径
        "vocals.onnx",  # 模型文件名
        BASE_DIR / "assets/uvr5_weights/onnx_dereverb_By_FoxJoy",  # 保存路径
    )

    # 定义预训练模型的保存目录
    rvc_models_dir = BASE_DIR / "assets/pretrained"

    # 下载一组预训练模型（D32k, D40k, G32k 等）
    print("Downloading pretrained models:")
    model_names = [
        "D32k.pth",
        "D40k.pth",
        "D48k.pth",
        "G32k.pth",
        "G40k.pth",
        "G48k.pth",
        "f0D32k.pth",
        "f0D40k.pth",
        "f0D48k.pth",
        "f0G32k.pth",
        "f0G40k.pth",
        "f0G48k.pth",
    ]
    # 遍历模型名并下载
    for model in model_names:
        print(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK + "pretrained/", model, rvc_models_dir)

    # 下载 v2 版本的预训练模型
    rvc_models_dir = BASE_DIR / "assets/pretrained_v2"
    print("Downloading pretrained models v2:")
    # 遍历并下载模型
    for model in model_names:
        print(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK + "pretrained_v2/", model, rvc_models_dir)

    # 下载 uvr5_weights 相关的模型
    print("Downloading uvr5_weights:")
    rvc_models_dir = BASE_DIR / "assets/uvr5_weights"
    # 定义 uvr5_weights 相关的模型名列表
    model_names = [
        "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ]
    # 遍历并下载 uvr5_weights 相关的所有模型
    for model in model_names:
        print(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK + "uvr5_weights/", model, rvc_models_dir)

    # 下载完成所有模型
    print("All models downloaded!")
