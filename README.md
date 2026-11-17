# 거짓 전제 실습: LLM을 거짓 전제로 속여 보기

의료 질문에 틀린 전제를 섞었을 때 LLM이 그 전제를 받아들이는지 직접 재 보는 실습입니다. 설치할 것은 없습니다. 구글 계정만 있으면 됩니다.

**수업 전에 아래 1~2단계를 끝내고 `✅ 연결 성공`이 나오는지 확인해 주세요.** 5분 정도 걸립니다.

## 1. Gemini API 키 받기 (무료)

1. https://aistudio.google.com 에 구글 계정으로 로그인합니다.
2. **Get API key** → **Create API key**를 누르고 키를 복사합니다.

키는 비밀번호와 같습니다. 다른 사람과 나누지 마세요.

> 무료 티어로 보낸 내용은 구글이 제품 개선에 쓸 수 있습니다. 실제 건강 정보나 개인정보를 넣지 마세요.

## 2. Colab에서 열고 연결 확인

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7897nikey/prism-practicum/blob/main/premise_lab.ipynb)

1. 위 버튼을 눌러 노트북을 엽니다.
2. **"0. 준비" 아래 셀만** 실행합니다(`Shift + Enter`). "Google에서 작성하지 않은 노트북" 경고가 뜨면 **무시하고 실행**을 누릅니다.
3. `키:` 칸에 1단계에서 복사한 키를 붙여 넣고 Enter를 누릅니다. 글자가 보이지 않는 게 정상입니다.
4. `✅ 연결 성공`이 나오면 준비 끝입니다. 나머지 셀은 수업 시간에 함께 실행합니다.

Colab은 창을 닫으면 저장한 키가 사라집니다. 수업 날 다시 열면 키를 한 번 더 붙여 넣으면 됩니다.

## 내 컴퓨터에서 돌리려면 (선택)

```bash
conda env create -f environment.yml
conda activate premise-lab
jupyter notebook premise_lab.ipynb
```

## 파일

| 파일 | 내용 |
|---|---|
| `premise_lab.ipynb` | 실습 노트북 |
| `lab_utils.py` | API 호출, 자동 판정, 집계 |
| `questions.jsonl` | 실습용 문항 8개. 실제 환자 질문이 아니라 실습을 위해 직접 쓴 문항입니다 |
| `test_lab.py` | API 없이 도는 점검 (`python test_lab.py`) |
