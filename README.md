# Bybit VWAP Strategy Monitor 📊

실시간으로 Bybit OI 상위 20개 USDT 무기한 선물을 모니터링하고 VWAP 평균회귀 전략 신호를 텔레그램으로 전송합니다.

## 🚀 설치 방법

### 1. Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 텔레그램 봇 설정

#### 봇 생성

1. 텔레그램에서 [@BotFather](https://t.me/botfather) 검색
2. `/newbot` 명령어 입력
3. 봇 이름 설정
4. 받은 **Bot Token** 저장

#### Chat ID 확인

1. [@userinfobot](https://t.me/userinfobot)에게 메시지 전송
2. 받은 숫자가 **Chat ID**

### 3. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 수정:

```bash
cp .env.example .env
nano .env
```

`.env` 파일 내용:

```env
# Bybit API (선택사항 - 데이터 조회만 하므로 비워둬도 됨)
BYBIT_API_KEY=
BYBIT_API_SECRET=

# 텔레그램 (필수)
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

## 📱 실행

```bash
python bybit_vwap_monitor.py
```

## ⚙️ 전략 설정

스크립트 상단에서 수정 가능:

```python
# 모니터링 설정
TIMEFRAMES = ['3m', '5m', '15m']  # 체크할 타임프레임
TOP_OI_COUNT = 20                  # 상위 OI 개수
CHECK_INTERVAL = 60                # 체크 주기 (초)

# 전략 파라미터
BAND_ENTRY_MULT = 2.0              # 엔트리 밴드 배수
MIN_STRENGTH = 0.7                 # 최소 신호 강도
MIN_VOL_RATIO = 0.25              # 최소 변동성 비율
STOP_POINTS = 20.0                # 손절 버퍼 (포인트)
```

## 📊 전략 로직

### 롱 신호 (H1/H2 패턴)

- 시가가 하단밴드 아래에서 시작
- 종가가 하단밴드 위로 돌파
- 신호 강도 ≥ 0.7

### 숏 신호 (L1/L2 패턴)

- 시가가 상단밴드 위에서 시작
- 종가가 상단밴드 아래로 하락
- 신호 강도 ≥ 0.7

### 필터링

- 변동성 필터: stdev / ATR ≥ 0.25
- 일봉 세션별 VWAP 계산

## 📲 텔레그램 알림 예시

```
🟢 LONG SIGNAL 🟢

Symbol: BTCUSDT
Timeframe: 5m
Entry: $43250.50
Stop Loss: $43220.30
Target (VWAP): $43280.75

Signal Strength: 85%
Vol Ratio: 0.42

Risk/Reward: 2.15

Time: 2025-12-13 10:30:45 UTC
```

## 🔧 문제 해결

### Rate Limit 에러

```python
CHECK_INTERVAL = 120  # 60초 → 120초로 증가
```

### 텔레그램 메시지 안 옴

1. Bot Token이 올바른지 확인
2. Chat ID가 올바른지 확인
3. 봇에게 먼저 메시지를 보냈는지 확인 (봇 활성화)

### 데이터 없음

- Bybit API가 정상인지 확인
- 네트워크 연결 확인
- VPN 사용 시 Bybit 접속 가능한지 확인

## ⚠️ 주의사항

1. **실전 매매 아님**: 이 스크립트는 신호만 제공합니다
2. **API Rate Limit**: 너무 자주 체크하면 Bybit에서 차단될 수 있음
3. **백테스트 필수**: 실제 돈 투자 전 충분히 테스트하세요
4. **자기책임**: 투자 손실에 대한 책임은 본인에게 있습니다

## 📝 로그 확인

실행 중 콘솔에서 다음 정보 확인:

- 상위 20개 OI 종목 목록
- 신호 발생 시 상세 정보
- 에러 메시지

## 🛑 종료

`Ctrl + C` 누르면 안전하게 종료됩니다.
