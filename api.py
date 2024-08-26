import os
import sys
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

from infer.modules.vc.modules import VC
from configs.config import Config
import torch
import warnings

import shutil
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.io.wavfile import write
import numpy as np
import io


logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from scipy.io.wavfile import read, write
import numpy as np
import io

app = FastAPI()

def convert_audio(input_audio_path):
    # 示例变声函数，简单反转音频数据
    log, converted = vc.vc_single(0, # 说话人ID
                input_audio_path, # 待转换音频路径
                0, # 变调，升八度-12，降八度12
                "", # F0曲线文件
                "rmvpe", # 音高提取算法
                "logs/lutao/added_IVF2435_Flat_nprobe_1_lutao_v2.index", # 特征检索文件路径
                "null", # 不使用
                0.75, # 检索特征占比
                3, # 滤波半径
                0, # 不进行重采样
                0.25, # 输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络
                0.33, # 保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果
                )
    print(log)
    return converted

@app.post("/convert_audio")
async def convert_audio_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    input_audio_path = '/tmp/input.wav'
    
    # 将上传的文件保存到 /tmp/input.wav
    with open(input_audio_path, 'wb') as f:
        f.write(await file.read())
    
    sample_rate, converted_audio_data = convert_audio(input_audio_path)

    # 将变声后的音频数据写入一个内存字节流
    buffer = io.BytesIO()
    write(buffer, sample_rate, converted_audio_data)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"})

if __name__ == '__main__':
    import uvicorn
    config = Config()
    vc = VC(config)
    vc.get_vc('lutao.pth', 0.33, 0.33)
    convert_audio('/app/dataset/test/output.wav') # 提前载入索引
    uvicorn.run(app, host='0.0.0.0', port=5000)
