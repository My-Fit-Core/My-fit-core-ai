"""분석 결과를 백엔드 callback_url로 보낸다 (HMAC 서명, #101).

서명 규칙은 백엔드와 세부 합의 전인 AI 파트 제안이다(02-specs.md §1).
- 키: 환경변수 AI_CALLBACK_HMAC_SECRET (11/25 백엔드 답변)
- X-AI-Timestamp: 보낸 시각(유닉스 초)
- X-AI-Signature: "sha256=" + hex(HMAC-SHA256(키, f"{timestamp}.{본문}"))
- 본문은 서명한 바이트 그대로 보낸다. 백엔드는 같은 방식으로 계산해 비교하고, 시각이 5분 넘게 차이 나면 거절한다.
"""

import hashlib
import hmac
import logging
import os
import time

import httpx

from app.schemas import AnalyzeResult

log = logging.getLogger(__name__)

TIMESTAMP_HEADER = "X-AI-Timestamp"
SIGNATURE_HEADER = "X-AI-Signature"
# ponytail: 재시도 3번, 간격 1·2초 고정. 백엔드가 오래 내려가 있으면 큐·영속 재시도가 필요하다
RETRY_DELAYS = (1.0, 2.0)
TIMEOUT_S = 10.0


def sign(body: bytes, secret: str, timestamp: int) -> dict[str, str]:
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return {
        TIMESTAMP_HEADER: str(timestamp),
        SIGNATURE_HEADER: f"sha256={mac.hexdigest()}",
    }


def verify(body: bytes, headers: dict[str, str], secret: str, now: int, max_age_s: int = 300) -> bool:
    """백엔드 쪽 검증과 같은 규칙. 테스트와 백엔드 참고용."""
    ts = int(headers[TIMESTAMP_HEADER])
    expected = sign(body, secret, ts)[SIGNATURE_HEADER]
    return abs(now - ts) <= max_age_s and hmac.compare_digest(expected, headers[SIGNATURE_HEADER])


def send_result(url: str, result: AnalyzeResult, client: httpx.Client, sleep=time.sleep) -> bool:
    """서명해서 POST한다. 2xx면 True. 실패하면 재시도 후 로그만 남긴다(알릴 다른 통로가 없다)."""
    secret = os.getenv("AI_CALLBACK_HMAC_SECRET", "")
    if not secret:
        log.warning("AI_CALLBACK_HMAC_SECRET 없음: 서명 없이 보낸다 (개발용)")
    body = result.model_dump_json().encode()
    for attempt in range(len(RETRY_DELAYS) + 1):
        headers = {"Content-Type": "application/json"}
        if secret:
            headers |= sign(body, secret, int(time.time()))
        try:
            res = client.post(url, content=body, headers=headers, timeout=TIMEOUT_S)
            if res.is_success:
                return True
            log.warning("callback %s: HTTP %s (시도 %d)", result.job_id, res.status_code, attempt + 1)
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            log.warning("callback %s: %s (시도 %d)", result.job_id, e, attempt + 1)
        if attempt < len(RETRY_DELAYS):
            sleep(RETRY_DELAYS[attempt])
    log.error("callback 전달 실패: job_id=%s url=%s", result.job_id, url)
    return False
