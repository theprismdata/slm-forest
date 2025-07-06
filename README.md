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
│   └── 📁 tax-law-gen-raw-data/         # 원시 데이터 (GitHub에서 제외됨)
│       ├── 📄 config.yaml               # API 설정 파일
│       ├── 📁 tax-law-openapi/          # 세법 OpenAPI 데이터 (12개 분야)
│       ├── 📁 tax-law-judgment/         # 세법 판례 데이터 (총 417MB)
│       └── 📁 fine-tunning-ds/          # 파인튜닝 데이터셋
│           └── 📄 distillation_legal_qa_dataset.json # 최종 Q&A 데이터셋 (5MB)
│
├── 📁 fine-tunning/                     # 모델 훈련 모듈
│   └── 📁 phi2/                         # Phi-2 훈련 설정
│       ├── 📄 finetuning-phi2.py       # Phi-2 파인튜닝 스크립트 (19KB)
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

## 🔧 기술적 세부사항

### 현재 모델 (Phi-2)
- **베이스 모델**: Microsoft Phi-2 (2.7B 파라미터)
- **파인튜닝 방법**: LoRA (Low-Rank Adaptation)
- **최적화**: FP16 훈련, 그래디언트 누적
- **⚠️ 한계점**: 
  - 한글 처리 성능 제한
  - M4 MacBook Pro에서 느린 추론 속도 (초당 5-10 토큰)
  - 복잡한 세법 질의에 대한 이해도 부족

### 차세대 모델 (Gemma3:12b) - 계획 중
- **베이스 모델**: Google Gemma3:12b (12B 파라미터)
- **예상 장점**:
  - 향상된 한글 지원
  - 더 나은 추론 능력
  - 복잡한 법률 논리 처리 개선
- **훈련 계획**: 동일한 데이터셋으로 LoRA 파인튜닝
- **인프라**: RunPod A100 GPU 활용

### 훈련 설정
- **최대 시퀀스 길이**: 2,048 토큰
- **프롬프트 템플릿**: 
  ```
  ### 질문: {question}
  
  ### 답변: {answer}
  ```
- **검증 분할**: 10%
- **조기 종료**: 3 에포크 연속 개선 없을 시

## 💰 비용 추정

### 데이터 생성 비용
- **OpenAI API**: 약 $50-100 (GPT-4o 사용)
- **Claude API**: 약 $30-50 (Claude-3-Sonnet 사용)

### RunPod 훈련 비용
- **Phi-2 (RTX 4090)**: 약 $1.2 - $4.8
- **Gemma3:12b (A100 40GB)**: 약 $10 - $20 (예상)

### 추론 비용
- **로컬 실행**: 무료 (성능 제한 있음)
- **클라우드 추론**: 검토 중

## 🛠️ 문제 해결

### 현재 알려진 문제들

**1. Phi-2 한글 성능 문제**
- 복잡한 한글 문장 이해 부족
- 법률 용어 처리 정확도 낮음
- **해결 방안**: Gemma3:12b로 모델 업그레이드

**2. M4 MacBook Pro 성능 문제**
- 추론 속도 느림 (초당 5-10 토큰)
- 메모리 사용량 높음 (12-16GB)
- **해결 방안**: 클라우드 추론 환경 검토

**3. 일반적인 기술적 문제**
```bash
# CUDA 메모리 부족
per_device_train_batch_size: 2
gradient_accumulation_steps: 8

# MPS 관련 오류
python inference.py --device cpu

# 모델 로딩 실패
rm -rf ~/.cache/huggingface/
```

## 📈 성능 모니터링

### 현재 성능 (Phi-2)
- **응답 시간**: 평균 10-20초 (M4 Mac)
- **토큰 생성 속도**: 초당 5-10 토큰
- **메모리 사용량**: 12-16GB
- **한글 정확도**: 60-70% (주관적 평가)

### 목표 성능 (Gemma3:12b)
- **응답 시간**: 평균 5-10초 (클라우드 환경)
- **토큰 생성 속도**: 초당 20-30 토큰
- **메모리 사용량**: 24-32GB
- **한글 정확도**: 80-90% (목표)

## 🔄 개발 로드맵

### Phase 1: 완료 ✅
- [x] 세법 데이터 수집 및 정제
- [x] Phi-2 파인튜닝 파이프라인 구축
- [x] 기본 추론 환경 설정

### Phase 2: 진행 중 🔄
- [x] 현재 모델 한계점 분석
- [ ] Gemma3:12b 파인튜닝 환경 준비
- [ ] 향상된 데이터셋 생성

### Phase 3: 계획 중 📋
- [ ] Gemma3:12b 파인튜닝 실행
- [ ] 성능 비교 분석
- [ ] 클라우드 추론 환경 구축
- [ ] 웹 인터페이스 개발


**현재 상태**: Phi-2 모델의 한글 처리 한계로 인해 Gemma3:12b 모델로의 업그레이드를 계획 중입니다.
