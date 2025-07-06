# SLM Forest - 세법 Q&A 파인튜닝 프로젝트
이 프로젝트는 Small Language Models(SLM)을 세법 Q&A 데이터셋으로 파인튜닝하여 실용적인 세법 질의응답 시스템을 구축하는 프로젝트입니다.

## 🎯 프로젝트 개요

- **현재 모델**: Microsoft Phi-2 (2.7B 파라미터) - 한글 성능 제한으로 인한 차세대 모델 검토 중
- **다음 단계**: Google Gemma3:12b 파인튜닝 예정 (향상된 한글 지원)
- **데이터셋**: 실용적인 세법 Q&A 데이터 (OpenAPI + 판례 데이터 기반)
- **훈련 환경**: RunPod (GPU 클라우드)
- **추론 환경**: 로컬 환경 (M4 MacBook Pro 성능 제한 확인됨)

## 📁 프로젝트 구조

```
slm-forest/
├── 📄 README.md                          # 프로젝트 메인 문서
├── 📄 .gitignore                         # Git 제외 파일 설정
│
├── 📁 slm_datagen/                       # 데이터 생성 모듈
│   ├── 📄 gen_finetunningds.py          # 파인튜닝 데이터셋 생성 스크립트 (20KB)
│   ├── 📄 setup_m4_mac.sh               # M4 Mac 환경 설정 스크립트

├── 📁 fine-tunning/                     # 모델 훈련 모듈
│   └── 📁 phi2/                         # Phi-2 훈련 설정
│       ├── 📄 finetuning-phi2.py       # Phi-2 파인튜닝 스크립트 
│       ├── 📄 train_config.yaml        # 훈련 설정 파일
│       ├── 📄 runpod_train.sh          # RunPod 훈련 실행 스크립트
│       └── 📄 requirements-finetunning.txt # 훈련 환경 의존성
│
└── 📁 model_serving/                    # 모델 서빙 모듈 (GitHub에서 제외됨)
    ├── 📄 phi-2-inference.py           # Phi-2 추론 스크립트
    ├── 📄 phi-2-fintunning-inference.py # 파인튜닝된 Phi-2 추론 스크립트 (12KB)
    ├── 📄 mistral-inference.py         # Mistral 추론 스크립트
    └── 📄 requirements-inference.txt    # 추론 환경 의존성
```

## 🚀 빠른 시작

### 1. 데이터셋 생성

#### 환경 설정
```bash
cd slm_datagen
pip install -r requirements.txt
```

#### API 설정
`tax-law-gen-raw-data/config.yaml` 파일에서 API 키 설정:
```yaml
langmodel:
  API:
    OpenAI:
      apikey: "your-openai-api-key"
      chat_model: "gpt-4o"
    Claude:
      apikey: "your-claude-api-key"
      chat_model: "claude-3-sonnet-20240229"
```

#### 데이터셋 생성 실행
```bash
python gen_finetunningds.py
```

### 2. RunPod에서 훈련하기

#### RunPod 설정
1. RunPod에서 GPU 인스턴스 생성 (RTX 4090, A100 등 권장)
2. 프로젝트 파일들을 RunPod에 업로드
3. 터미널에서 다음 명령어 실행:

```bash
cd fine-tunning/phi2

# 실행 권한 부여
chmod +x runpod_train.sh

# 훈련 시작
./runpod_train.sh
```

#### 훈련 설정 조정
`train_config.yaml` 파일에서 다음 설정을 조정할 수 있습니다:

```yaml
training_args:
  num_train_epochs: 3                    # 훈련 에포크
  per_device_train_batch_size: 4         # 배치 크기
  learning_rate: 2e-5                    # 학습률
  gradient_accumulation_steps: 4         # 그래디언트 누적

lora_config:
  r: 16                                  # LoRA 랭크
  lora_alpha: 32                         # LoRA 알파
  lora_dropout: 0.1                      # LoRA 드롭아웃
```

### 3. 모델 추론 (현재 제한사항 있음)

#### 환경 설정
```bash
cd model_serving
pip install -r requirements-inference.txt
```

#### 추론 실행
```bash
# 기본 Phi-2 추론
python phi-2-inference.py --question "부가가치세 신고는 어떻게 하나요?"

# 파인튜닝된 모델 추론
python phi-2-fintunning-inference.py --interactive
```

## 📊 데이터셋 정보

### 원시 데이터 소스
- **OpenAPI 데이터**: 12개 세법 분야별 구조화된 데이터
- **판례 데이터**: 총 417MB의 세법 관련 판례 및 해석 데이터
- **최종 데이터셋**: 5MB의 Q&A 형태로 정제된 데이터

### 데이터 생성 파이프라인
1. **원시 데이터 수집**: OpenAPI 및 판례 데이터 크롤링
2. **데이터 정제**: 중복 제거, 형식 통일
3. **Q&A 변환**: LLM을 활용한 질문-답변 쌍 생성
4. **품질 검증**: 자동화된 품질 체크 및 수동 검토

### 데이터셋 예시
```json
{
  "question": "부모님께 20억원 증여받았는데 증여세는 얼마나 되나요?",
  "answer": "20억원 증여에 대한 증여세는 다음과 같이 계산됩니다:\n\n【증여세 계산】\n- 증여재산가액: 20억원\n- 공제: 없음 (부모님 증여는 공제 대상 아님)\n- 과세가액: 20억원\n\n【세율 적용】\n- 1억원 이하: 10%\n- 1억원 초과 5억원 이하: 20%\n- 5억원 초과 10억원 이하: 30%\n\n【세액 계산】\n- 1억원 × 10% = 1,000만원\n- 4억원 × 20% = 8,000만원\n- 총 증여세: 9,000만원",
  "type": "student",
  "source": "practical"
}
```

**현재 상태**: Phi-2 모델의 한글 처리 한계로 인해 Gemma3:12b 모델로의 업그레이드를 계획 중.
