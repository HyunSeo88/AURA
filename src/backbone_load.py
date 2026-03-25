import torch
from speechbrain.inference.separation import SepformerSeparation

def sepformer_load(source="speechbrain/sepformer-wsj02mix", savedir="pretrained_models/sepformer-wsj02mix"):
    """
    Hugging Face/SpeechBrain에서 Pre-trained SepFormer 모델을 로드
    
    Args:
        source (str): HuggingFace 모델 레포지토리 주소
        savedir (str): 모델 저장 경로
        
    Returns:
        model (SepformerSeparation): 로드된 전체 모델 객체
    """
    print(f"Loading SepFormer backbone from {source}...")
    
    model = SepformerSeparation.from_hparams(
        source=source,
        savedir=savedir
    )
    
    # 모델을 Training Mode로 전환 (Fine-tuning을 위해)
    model.train()
    
    print("SepFormer backbone loaded successfully.")
    return model

if __name__ == "__main__":
    # 테스트 코드
    model = sepformer_load()
    print(model)