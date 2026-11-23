"""마스크 안 픽셀 k-means로 옷의 대표색을 뽑는다 (F-203, 02-specs.md §2).

색으로 추구미 점수를 보정하지 않는다(#50). 패턴은 방법·값 목록이 미정이라 여기서 다루지 않는다.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ponytail: 근거 문서 없는 기본값. 앱에 보여 줄 색 개수 기준이 생기면 configs/로 옮긴다
K = 3
MAX_PIXELS = 10_000


def extract_colors(
    image: Image.Image,
    mask: np.ndarray,
    k: int = K,
    max_pixels: int = MAX_PIXELS,
    seed: int = 0,
) -> list[tuple[str, float]]:
    """("#RRGGBB", ratio) 목록, ratio 내림차순·합 1. 마스크가 비면 []."""
    pixels = np.asarray(image.convert("RGB"))
    if mask.shape != pixels.shape[:2]:
        raise ValueError(f"mask {mask.shape} != image {pixels.shape[:2]}")
    pixels = pixels[mask.astype(bool)]
    if len(pixels) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(pixels) > max_pixels:
        pixels = pixels[rng.choice(len(pixels), max_pixels, replace=False)]

    n_clusters = min(k, len(np.unique(pixels, axis=0)))
    # n_init=1: 실제 사진 2벌에서 n_init=4와 같은 결과, 시간은 약 1/10 (Report 참고)
    km = KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit(pixels)
    counts = np.bincount(km.labels_, minlength=n_clusters)
    order = np.argsort(-counts)
    total = counts.sum()
    return [
        (
            "#{:02X}{:02X}{:02X}".format(
                *np.clip(np.rint(km.cluster_centers_[i]), 0, 255).astype(int)
            ),
            float(counts[i] / total),
        )
        for i in order
    ]
