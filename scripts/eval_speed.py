"""추론 속도 측정 (T13 partial) — 워밍 후 detection 파이프라인 타이밍.

가중치 파일이 없으면 안내 메시지 후 종료한다.
T1(실사 Flat Lay 사진) 없이도 합성 이미지로 기본 타이밍을 잴 수 있다.

사용법:
    python scripts/eval_speed.py                     # 합성 이미지 (640×480)
    python scripts/eval_speed.py --image path/to/img.jpg
    python scripts/eval_speed.py --runs 20
"""

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from app.pipeline.detection import WEIGHTS_PATH, detect, load_detector
    from app.pipeline.image_input import validate_image
except ImportError as e:
    print(f"[skip] ML 패키지 없음 ({e}). requirements-ml.txt 를 먼저 설치하세요.")
    sys.exit(0)


def _make_synthetic(width: int = 640, height: int = 480):
    """흰 배경에 단색 사각형을 그린 합성 Flat Lay 이미지."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (240, 240, 235))
    draw = ImageDraw.Draw(img)
    # 옷처럼 보이는 직사각형 2개
    draw.rectangle([80, 60, 280, 380], fill=(50, 80, 120))
    draw.rectangle([320, 80, 560, 400], fill=(180, 100, 60))
    return img


def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def main():
    parser = argparse.ArgumentParser(description="Detection 파이프라인 추론 속도 측정")
    parser.add_argument("--image", type=Path, default=None, help="테스트 이미지 경로 (없으면 합성)")
    parser.add_argument("--runs", type=int, default=10, help="타이밍 반복 횟수 (워밍 제외)")
    args = parser.parse_args()

    # ── 가중치 체크 ──────────────────────────────────────────────
    if not WEIGHTS_PATH.exists():
        print(f"[skip] 가중치 파일 없음: {WEIGHTS_PATH}")
        print("       python scripts/download_weights.py 로 먼저 다운로드하세요.")
        sys.exit(0)

    # ── 이미지 준비 ──────────────────────────────────────────────
    if args.image:
        from PIL import Image
        raw = Image.open(args.image)
        label = args.image.name
    else:
        raw = _make_synthetic()
        label = "synthetic 640×480"

    img, warnings = validate_image(raw)
    if warnings:
        print(f"[warn] {warnings}")
    print(f"이미지: {label}  ({img.width}×{img.height})")
    print(f"반복:   {args.runs}회 (워밍 1회 별도)")

    # ── 워밍 ─────────────────────────────────────────────────────
    print("\n워밍 중...", end=" ", flush=True)
    t0 = time.perf_counter()
    load_detector()          # 모델 로드
    detect(img)              # 첫 추론
    warmup = time.perf_counter() - t0
    print(f"{_fmt(warmup)}  (모델 로드 + 첫 추론 합산)")

    # ── 타이밍 ───────────────────────────────────────────────────
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        results = detect(img)
        times.append(time.perf_counter() - t0)

    n_det = len(results)
    print(f"\n검출 결과: {n_det}벌")
    print(f"{'mean':>6}: {_fmt(mean(times))}")
    print(f"{'min':>6}: {_fmt(min(times))}")
    print(f"{'max':>6}: {_fmt(max(times))}")
    if args.runs >= 2:
        print(f"{'stdev':>6}: {_fmt(stdev(times))}")

    target = 2.0
    avg = mean(times)
    if avg <= target:
        print(f"\n✓ 목표 {target}s 이내 ({_fmt(avg)})")
    else:
        print(f"\n✗ 목표 {target}s 초과 ({_fmt(avg)}) — 최적화 필요")


if __name__ == "__main__":
    main()
