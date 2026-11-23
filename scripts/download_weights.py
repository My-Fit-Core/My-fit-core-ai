"""검출 가중치(deepfashion2_yolov8s-seg.pt)를 내려받는다. 이미 있으면 건너뛴다.

사용: python scripts/download_weights.py
저장 위치: 환경변수 DETECTOR_WEIGHTS, 없으면 weights/ (커밋 금지, .gitignore).
출처·라이선스: docs/reports/261108-0305-01-detection-segmentation-poc.md (비상업 연구용)
"""

import hashlib
import os
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt"
# Hugging Face LFS sha256. .pt는 pickle이라 로드 시 코드가 실행될 수 있어 꼭 확인한다.
SHA256 = "c319c15e86443ea61f70b40620909d737c47ff2a39381536503ac5c54772f8ac"
ROOT = Path(__file__).resolve().parents[1]
# 빈 값(.env.example 복사)도 기본 경로로
DEST = Path(
    os.getenv("DETECTOR_WEIGHTS") or ROOT / "weights" / "deepfashion2_yolov8s-seg.pt"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if DEST.exists():
        print(f"이미 있음: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    if sha256(tmp) != SHA256:
        tmp.unlink()
        raise SystemExit("sha256 불일치: 받은 파일을 지웠다")
    tmp.rename(DEST)
    print(f"저장: {DEST}")


if __name__ == "__main__":
    main()
