
import os
import re
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

# =========================
# 工具：自然排序
# =========================
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(folder: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    fs = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def list_subfolders(folder: str):
    subs = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    subs.sort(key=natural_key)
    return subs


# =========================
# 可视化颜色
# =========================
def _color_from_id(k: int):
    rng = np.random.default_rng(12345 + k * 97)
    c = rng.integers(50, 256, size=3, dtype=np.uint8)
    return int(c[0]), int(c[1]), int(c[2])  # B,G,R


# =========================
# 命名匹配：当前 mask 和 skeleton 完全同名
# =========================
def skel_to_mask_name(skel_name: str) -> str:
    return skel_name


# =========================
# 读 skeleton（0/1）
# =========================
def read_skeleton01(skel_path: str, binarize_mode="otsu", fixed_thresh=127):
    raw = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"无法读取 skeleton: {skel_path}")

    if binarize_mode == "fixed":
        _, bw = cv2.threshold(raw, fixed_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, bw = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    skel01 = (bw > 0).astype(np.uint8)
    return raw, skel01


# =========================
# 读 mask（二值化后用于 distanceTransform）
# =========================
def read_mask01(mask_path: str, binarize_mode="otsu", fixed_thresh=127,
                close_ks: int = 7, open_ks: int = 0):
    raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"无法读取 mask: {mask_path}")

    if binarize_mode == "fixed":
        _, bw = cv2.threshold(raw, fixed_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, bw = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if close_ks and close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    if open_ks and open_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    mask01 = (bw > 0).astype(np.uint8)
    return raw, mask01

def distance_transform(mask01: np.ndarray) -> np.ndarray:
    dt = cv2.distanceTransform(mask01.astype(np.uint8), cv2.DIST_L2, 5)
    return dt.astype(np.float32)


# =========================
# BaseSeed：优先限制在 x_center 附近，再在其中尽量选靠下点
# =========================
def pick_base_seed(
    skel01: np.ndarray,
    x_center: float,
    x_tol: Optional[int] = None,
    x_tol_ratio: float = 0.06
):
    """
    优先在 x_center 附近选 BaseSeed，不再强调“必须在最底部 y_band 内”。

    规则：
    1) 在全图骨架中，仅保留 |x - x_center| <= x_tol 的点
    2) 在这些点中优先选 y 最大（更靠下）的点
    3) 若同样靠下有多个点，再选离 x_center 最近的点
    4) 若 x_center 附近没有骨架点，则退回全图：先尽量靠下，再尽量靠近 x_center
    """
    ys, xs = np.where(skel01 == 1)
    if ys.size == 0:
        return None

    H, W = skel01.shape
    if x_tol is None:
        x_tol = max(25, int(round(W * x_tol_ratio)))

    dx = np.abs(xs.astype(np.float32) - float(x_center))
    near_mask = dx <= float(x_tol)

    # 优先：限制在 x_center 附近
    if np.any(near_mask):
        xs_near = xs[near_mask]
        ys_near = ys[near_mask]

        # 先找最靠下
        y_max = ys_near.max()
        low_mask = (ys_near == y_max)

        xs_low = xs_near[low_mask]
        ys_low = ys_near[low_mask]

        # 再找最靠近 x_center
        j = int(np.argmin(np.abs(xs_low.astype(np.float32) - float(x_center))))
        return (int(xs_low[j]), int(ys_low[j]))

    # 退回：若附近没有骨架点，则全图中优先靠下，再靠近 x_center
    y_max = ys.max()
    low_mask = (ys >= y_max - 3)  # 给最底部少量冗余
    xs_low = xs[low_mask]
    ys_low = ys[low_mask]

    if xs_low.size > 0:
        j = int(np.argmin(np.abs(xs_low.astype(np.float32) - float(x_center))))
        return (int(xs_low[j]), int(ys_low[j]))

    j = int(np.argmin(np.abs(xs.astype(np.float32) - float(x_center))))
    return (int(xs[j]), int(ys[j]))


def geodesic_dist_on_skeleton(skel01: np.ndarray, seed_xy):
    """BFS 计算骨架像素到 seed 的测地距离；非骨架为 inf"""
    H, W = skel01.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    if seed_xy is None:
        return dist
    sx, sy = seed_xy
    if not (0 <= sx < W and 0 <= sy < H) or skel01[sy, sx] == 0:
        return dist

    q = deque()
    dist[sy, sx] = 0.0
    q.append((sx, sy))

    nbrs = [(-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)]

    while q:
        x, y = q.popleft()
        d0 = dist[y, x]
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and skel01[ny, nx] == 1:
                nd = d0 + 1.0
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    q.append((nx, ny))
    return dist


@dataclass
class ViewResult:
    name: str
    skel_path: str
    mask_path: str
    H: int
    W: int
    y_top: int
    y_bottom: int
    usable_y_bottom: int
    roi_y0: int
    roi_y1: int
    n_layers: int
    layer_bounds: List[Tuple[int, int]]
    counts: List[int]
    out_subdir: str


# ============================================================
# 第一遍：每张 skeleton 分层统计CC +（可选）保存每层可视化
# ============================================================
def process_one_image_and_save_layer_visuals(
    skel_path: str,
    out_subdir: str,
    n_layers: int = 15,
    bottom_frac: float = 1/3,
    base_cut_frac: float = 0.15,
    connectivity: int = 8,
    remove_short: bool = True,
    save_layer_visuals: bool = True,
):
    os.makedirs(out_subdir, exist_ok=True)

    raw, skel01 = read_skeleton01(skel_path, binarize_mode="otsu")
    H, W = skel01.shape

    ys_all = np.where(skel01 == 1)[0]
    if ys_all.size == 0:
        raise ValueError(f"骨架像素为空: {skel_path}")

    y_top = int(ys_all.min())
    y_bottom = int(ys_all.max())
    span = y_bottom - y_top + 1

    cut_h = int(np.ceil(span * base_cut_frac))
    usable_y_bottom = max(y_top, y_bottom - cut_h)
    usable_span = usable_y_bottom - y_top + 1
    if usable_span <= 0:
        raise ValueError(f"剔除基部后 usable_span<=0，请减小 base_cut_frac: {skel_path}")

    roi_h = int(np.ceil(usable_span * bottom_frac))
    roi_h = max(1, min(roi_h, usable_span))

    roi_y0 = usable_y_bottom - roi_h + 1
    roi_y1 = usable_y_bottom + 1

    roi = skel01[roi_y0:roi_y1, :]
    roi_h = roi.shape[0]
    if roi_h <= 0:
        raise ValueError(f"ROI高度为0: {skel_path}")

    edges = np.linspace(0, roi_h, n_layers + 1)
    edges = np.round(edges).astype(int)
    edges[0] = 0
    edges[-1] = roi_h

    counts: List[int] = []
    layer_bounds: List[Tuple[int, int]] = []

    for li in range(n_layers):
        y0, y1 = int(edges[li]), int(edges[li + 1])
        abs_y0, abs_y1 = roi_y0 + y0, roi_y0 + y1
        layer_bounds.append((abs_y0, abs_y1))

        if y1 <= y0:
            counts.append(0)
            continue

        layer = roi[y0:y1, :]
        layer_h = layer.shape[0]

        if layer.sum() == 0:
            counts.append(0)
            if save_layer_visuals:
                vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                cv2.line(vis, (0, abs_y0), (W - 1, abs_y0), (0, 255, 0), 1)
                cv2.line(vis, (0, abs_y1 - 1), (W - 1, abs_y1 - 1), (0, 255, 0), 1)
                cv2.putText(vis, f"Layer {li+1:02d} CC=0", (10, min(H - 10, abs_y0 + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                outp = os.path.join(out_subdir, f"layer_{li+1:02d}_y[{abs_y0},{abs_y1})_CC0.png")
                cv2.imwrite(outp, vis)
            continue

        layer_u8 = (layer * 255).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(layer_u8, connectivity=connectivity)

        if remove_short and num_labels > 1:
            threshold = layer_h
            keep = np.zeros(num_labels, dtype=bool)
            for cid in range(1, num_labels):
                ys, xs = np.where(labels == cid)
                if ys.size == 0:
                    continue
                len_y = int(ys.max() - ys.min() + 1)
                if len_y >= threshold:
                    keep[cid] = True
            layer_keep = keep[labels]
            layer_filt = np.where(layer_keep, 255, 0).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(layer_filt, connectivity=connectivity)

        comp = int(num_labels - 1)
        counts.append(comp)

        if save_layer_visuals:
            vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            vis_roi = vis[abs_y0:abs_y1, :]
            for cid in range(1, num_labels):
                m = (labels == cid)
                if np.any(m):
                    vis_roi[m] = _color_from_id(cid)

            cv2.line(vis, (0, abs_y0), (W - 1, abs_y0), (0, 255, 0), 1)
            cv2.line(vis, (0, abs_y1 - 1), (W - 1, abs_y1 - 1), (0, 255, 0), 1)
            cv2.putText(vis, f"Layer {li+1:02d} CC={comp}", (10, min(H - 10, abs_y0 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            outp = os.path.join(out_subdir, f"layer_{li+1:02d}_y[{abs_y0},{abs_y1})_CC{comp}.png")
            cv2.imwrite(outp, vis)

    return raw, skel01, ViewResult(
        name=os.path.splitext(skel_to_mask_name(os.path.basename(skel_path)))[0],
        skel_path=skel_path,
        mask_path="",
        H=H, W=W,
        y_top=y_top, y_bottom=y_bottom,
        usable_y_bottom=usable_y_bottom,
        roi_y0=roi_y0, roi_y1=roi_y1,
        n_layers=n_layers,
        layer_bounds=layer_bounds,
        counts=counts,
        out_subdir=out_subdir
    )


# ============================================================
# 统一高度带 + 每视角选层
# ============================================================
def choose_unified_band_and_per_view_layer(
    view_results: List[ViewResult],
    band_len: int = 5,
    avoid_bottom_layers: int = 1,
    avoid_top_layers: int = 0,
):
    n_layers = view_results[0].n_layers
    M = len(view_results)

    counts_mat = np.array([vr.counts for vr in view_results], dtype=float)
    counts_mat[counts_mat <= 0] = np.nan

    max_start = n_layers - band_len - avoid_bottom_layers + 1
    min_start = 1 + avoid_top_layers
    if max_start < min_start:
        raise ValueError("band_len/avoid_bottom_layers 设置不合理，导致无候选带")

    best = None
    best_pick_layers = None

    for start in range(min_start, max_start + 1):
        band_idx = np.arange(start - 1, start - 1 + band_len)
        band_vals = counts_mat[:, band_idx].reshape(-1)
        band_vals = band_vals[~np.isnan(band_vals)]
        if band_vals.size < max(5, M):
            continue

        cc_star = np.nanmedian(band_vals)

        pick_layers = []
        pick_ccs = []
        pick_devs = []

        for i in range(M):
            row = counts_mat[i, band_idx]
            if np.all(np.isnan(row)):
                pick_layers.append(np.nan)
                pick_ccs.append(np.nan)
                pick_devs.append(np.nan)
                continue

            best_l, best_cost, best_cc = None, None, None
            for j in range(band_len):
                cc = row[j]
                if np.isnan(cc):
                    continue

                j0 = max(0, j - 1)
                j1 = min(band_len, j + 2)
                win = row[j0:j1]
                win = win[~np.isnan(win)]
                local_var = float(np.var(win)) if win.size >= 2 else 0.0

                dev = abs(cc - cc_star)
                cost = dev + 0.25 * local_var

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_l = (start + j)
                    best_cc = cc

            pick_layers.append(best_l)
            pick_ccs.append(best_cc)
            pick_devs.append(abs(best_cc - cc_star) if best_cc is not None else np.nan)

        pick_ccs_arr = np.array(pick_ccs, dtype=float)
        valid = ~np.isnan(pick_ccs_arr)
        if valid.sum() < max(3, int(0.6 * M)):
            continue

        mean_dev = float(np.nanmean(pick_devs))
        std_cc = float(np.nanstd(pick_ccs_arr))
        mean_cc = float(np.nanmean(pick_ccs_arr))

        score = mean_dev + 0.35 * std_cc - 0.05 * mean_cc
        if best is None or score < best[0]:
            best = (score, start, cc_star)
            best_pick_layers = pick_layers

    if best is None:
        start = 6 if n_layers >= 10 else 1
        cc_star = np.nanmedian(counts_mat)
        best = (999.0, start, cc_star)
        best_pick_layers = [start + band_len // 2] * M

    _, band_start, cc_star = best
    band_end = band_start + band_len - 1
    return band_start, band_end, float(cc_star), best_pick_layers


# ============================================================
# 计算某张图的“选定层labels”
# ============================================================
def compute_layer_labels_for_one_layer_from_skel01(
    skel01: np.ndarray,
    layer_index_1based: int,
    n_layers: int,
    roi_y0: int,
    roi_y1: int,
    connectivity: int = 8,
    remove_short: bool = True,
):
    roi = skel01[roi_y0:roi_y1, :]
    roi_h = roi.shape[0]

    edges = np.linspace(0, roi_h, n_layers + 1)
    edges = np.round(edges).astype(int)
    edges[0] = 0
    edges[-1] = roi_h

    li = layer_index_1based - 1
    y0, y1 = int(edges[li]), int(edges[li + 1])
    abs_y0, abs_y1 = roi_y0 + y0, roi_y0 + y1

    layer = roi[y0:y1, :]
    if layer.size == 0 or layer.sum() == 0:
        return (abs_y0, abs_y1), 0, None

    layer_u8 = (layer * 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(layer_u8, connectivity=connectivity)

    if remove_short and num_labels > 1:
        threshold = layer.shape[0]
        keep = np.zeros(num_labels, dtype=bool)
        for cid in range(1, num_labels):
            ys, xs = np.where(labels == cid)
            if ys.size == 0:
                continue
            len_y = int(ys.max() - ys.min() + 1)
            if len_y >= threshold:
                keep[cid] = True
        layer_keep = keep[labels]
        layer_filt = np.where(layer_keep, 255, 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(layer_filt, connectivity=connectivity)

    comp = int(num_labels - 1)
    return (abs_y0, abs_y1), comp, labels


# ============================================================
# 候选分蘖点：每个连通域质心
# ============================================================
def extract_component_centroids(labels: np.ndarray, abs_y0: int) -> List[Tuple[int, float, float]]:
    pts = []
    if labels is None:
        return pts
    num_labels = int(labels.max() + 1)
    for cid in range(1, num_labels):
        ys, xs = np.where(labels == cid)
        if ys.size == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean() + abs_y0)
        pts.append((cid, cx, cy))
    return pts


# ============================================================

# ============================================================
def pick_main_component_in_selected_layer(
    labels: np.ndarray,
    abs_y0: int,
    dt: np.ndarray,
    x_center: float,
    W: int,
    skel_geodist: np.ndarray,
    w_center: float = 13.0,
    w_base: float = 8.0
):
    if labels is None:
        return None

    num_labels = int(labels.max() + 1)
    if num_labels <= 1:
        return None

    finite_all = skel_geodist[np.isfinite(skel_geodist)]
    gmax = float(finite_all.max()) if finite_all.size else 1.0
    gmax = max(1.0, gmax)

    best = None
    for cid in range(1, num_labels):
        ys, xs = np.where(labels == cid)
        if ys.size == 0:
            continue

        y_global = ys + abs_y0
        x_global = xs

        cx = float(xs.mean())
        center_pen = abs(cx - x_center) / max(1.0, W)

        dvals = dt[y_global, x_global]
        mean_dt = float(np.mean(dvals)) if dvals.size else 0.0

        g = skel_geodist[y_global, x_global]
        g = g[np.isfinite(g)]
        if g.size == 0:
            continue
        base_dist_med = float(np.median(g))
        base_dist_norm = base_dist_med / gmax

        yspan = float(ys.max() - ys.min() + 1)
        area = float(xs.size)

        score = (2.5 * mean_dt) + (0.8 * yspan) + (0.002 * area) \
                - (w_center * center_pen) \
                - (w_base * base_dist_norm)

        seed_x = int(x_global[0])
        seed_y = int(y_global[0])

        if best is None or score > best[0]:
            best = (score, int(cid), seed_x, seed_y)

    return best


# ============================================================

# ============================================================
def main_stem_point_by_min_geodist(
    labels: np.ndarray,
    abs_y0: int,
    main_cid: int,
    skel_geodist: np.ndarray,
    x_ref: float,
    slack: int = 3
) -> Optional[Tuple[float, float]]:
    if labels is None or main_cid is None or main_cid <= 0:
        return None

    ys, xs = np.where(labels == main_cid)
    if ys.size == 0:
        return None

    yg = ys + abs_y0
    xg = xs
    g = skel_geodist[yg, xg]
    ok = np.isfinite(g)
    if not np.any(ok):
        return None

    yg = yg[ok]
    xg = xg[ok]
    g = g[ok]

    gmin = float(np.min(g))
    keep = (g <= gmin + float(slack))
    yg2 = yg[keep]
    xg2 = xg[keep]
    if yg2.size == 0:
        return None

    j = int(np.argmin(np.abs(xg2.astype(np.float32) - float(x_ref))))
    return (float(xg2[j]), float(yg2[j]))


# ============================================================
# 单个子文件夹处理
# ============================================================
def run_pipeline(
    mask_folder: str,
    skel_folder: str,
    out_root: str,
    n_layers: int = 15,
    bottom_frac: float = 1/3,
    base_cut_frac: float = 0.15,
    connectivity: int = 8,
    remove_short: bool = True,
    band_len: int = 5,
    save_layer_visuals: bool = True,
    w_center: float = 13.0,
    w_base: float = 8.0,
    base_seed_x_tol: Optional[int] = None,
    base_seed_x_tol_ratio: float = 0.06
):
    os.makedirs(out_root, exist_ok=True)

    skel_files = list_images(skel_folder)
    if not skel_files:
        raise FileNotFoundError(f"skeleton 文件夹无图像: {skel_folder}")

    view_name_to_idx = {}
    valid_view_counter = 0

    view_results: List[ViewResult] = []
    cache_skel = {}  # name -> (raw, skel01)

    for sf in skel_files:
        skel_path = os.path.join(skel_folder, sf)
        mask_name = skel_to_mask_name(sf)
        mask_path = os.path.join(mask_folder, mask_name)

        if not os.path.exists(mask_path):
            print(f"[跳过] 找不到对应mask: {sf} -> {mask_name}")
            continue

        name = os.path.splitext(mask_name)[0]
        view_name_to_idx[name] = valid_view_counter
        valid_view_counter += 1

        out_subdir = os.path.join(out_root, name)

        raw, skel01, vr = process_one_image_and_save_layer_visuals(
            skel_path=skel_path,
            out_subdir=out_subdir,
            n_layers=n_layers,
            bottom_frac=bottom_frac,
            base_cut_frac=base_cut_frac,
            connectivity=connectivity,
            remove_short=remove_short,
            save_layer_visuals=save_layer_visuals
        )
        vr.mask_path = mask_path
        view_results.append(vr)
        cache_skel[name] = (raw, skel01)

    if not view_results:
        raise RuntimeError("没有成功配对的mask+skeleton，请检查命名与文件夹。")

    band_start, band_end, cc_star, per_view_layer = choose_unified_band_and_per_view_layer(
        view_results=view_results,
        band_len=band_len,
        avoid_bottom_layers=1,
        avoid_top_layers=0
    )

    summary_rows = []
    points_rows = []
    stem_rows = []

    for vr, sel_layer in zip(view_results, per_view_layer):
        name = vr.name
        out_subdir = vr.out_subdir
        raw_skel, skel01 = cache_skel[name]

        _, mask01 = read_mask01(vr.mask_path, binarize_mode="otsu", close_ks=7, open_ks=0)
        dt = distance_transform(mask01)

        (abs_y0, abs_y1), comp, labels = compute_layer_labels_for_one_layer_from_skel01(
            skel01=skel01,
            layer_index_1based=int(sel_layer),
            n_layers=n_layers,
            roi_y0=vr.roi_y0,
            roi_y1=vr.roi_y1,
            connectivity=connectivity,
            remove_short=remove_short
        )

        band = mask01[abs_y0:abs_y1, :]
        ys_b, xs_b = np.where(band == 1)
        x_center = float(xs_b.mean()) if xs_b.size else (vr.W / 2.0)

        base_seed = pick_base_seed(
            skel01=skel01,
            x_center=x_center,
            x_tol=base_seed_x_tol,
            x_tol_ratio=base_seed_x_tol_ratio
        )
        skel_geodist = geodesic_dist_on_skeleton(skel01, base_seed)

        centroids = extract_component_centroids(labels, abs_y0=abs_y0)

        pick = pick_main_component_in_selected_layer(
            labels=labels,
            abs_y0=abs_y0,
            dt=dt,
            x_center=x_center,
            W=vr.W,
            skel_geodist=skel_geodist,
            w_center=w_center,
            w_base=w_base
        )

        main_cid = -1
        stem_pt = None
        if pick is not None:
            _, main_cid, _, _ = pick
            stem_pt = main_stem_point_by_min_geodist(
                labels=labels,
                abs_y0=abs_y0,
                main_cid=main_cid,
                skel_geodist=skel_geodist,
                x_ref=x_center,
                slack=3
            )

        if main_cid > 0:
            centroids = [p for p in centroids if p[0] != main_cid]

        summary_rows.append({
            "view": name,
            "view_idx": int(view_name_to_idx[name]),
            "skeleton_path": vr.skel_path,
            "mask_path": vr.mask_path,
            "band_start": band_start,
            "band_end": band_end,
            "cc_star": cc_star,
            "selected_layer": int(sel_layer),
            "selected_layer_y0": int(abs_y0),
            "selected_layer_y1": int(abs_y1),
            "selected_CC": int(comp),
            "roi_y0": int(vr.roi_y0),
            "roi_y1": int(vr.roi_y1),
        })

        for (cid, x, y) in centroids:
            points_rows.append({
                "view": name,
                "view_idx": int(view_name_to_idx[name]),
                "component_id": int(cid),
                "x": float(x),
                "y": float(y),
                "selected_layer": int(sel_layer),
                "selected_CC": int(comp),
            })

        stem_rows.append({
            "view": name,
            "view_idx": int(view_name_to_idx[name]),
            "main_cid_in_layer": int(main_cid),
            "x_center": float(x_center),
            "base_seed_x": float(base_seed[0]) if base_seed is not None else np.nan,
            "base_seed_y": float(base_seed[1]) if base_seed is not None else np.nan,
            "stem_x": float(stem_pt[0]) if stem_pt is not None else np.nan,
            "stem_y": float(stem_pt[1]) if stem_pt is not None else np.nan,
            "selected_layer": int(sel_layer),
            "selected_CC": int(comp),
        })

        vis = cv2.cvtColor(raw_skel, cv2.COLOR_GRAY2BGR)

        cv2.line(vis, (0, abs_y0), (vr.W - 1, abs_y0), (0, 255, 0), 2)
        cv2.line(vis, (0, abs_y1 - 1), (vr.W - 1, abs_y1 - 1), (0, 255, 0), 2)

        if base_seed is not None:
            cv2.circle(vis, (int(base_seed[0]), int(base_seed[1])), 7, (255, 255, 0), -1)
            cv2.putText(vis, "BaseSeed", (int(base_seed[0]) + 8, int(base_seed[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if labels is not None and main_cid > 0:
            ys, xs = np.where(labels == main_cid)
            if ys.size:
                yy = ys + abs_y0
                xx = xs
                vis[yy, xx] = (180, 180, 180)

        for (cid, x, y) in centroids:
            cv2.circle(vis, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1)
            cv2.putText(vis, f"{cid}", (int(round(x)) + 6, int(round(y)) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if stem_pt is not None:
            sx, sy = stem_pt
            cv2.circle(vis, (int(round(sx)), int(round(sy))), 9, (255, 0, 0), -1)
            cv2.putText(vis, "MainStem", (int(round(sx)) + 8, int(round(sy)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(
            vis,
            f"Band L{band_start}-{band_end} | pick L{int(sel_layer)} CC={comp} | mainCID={main_cid} | wC={w_center} wB={w_base}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        out_img = os.path.join(out_subdir, f"_selected_layer_L{int(sel_layer):02d}_CC{comp}_cands_and_mainstem.png")
        cv2.imwrite(out_img, vis)

    df_summary = pd.DataFrame(summary_rows)
    df_points = pd.DataFrame(points_rows)
    df_stem = pd.DataFrame(stem_rows)

    summary_csv = os.path.join(out_root, "selection_summary.csv")
    points_csv = os.path.join(out_root, "candidate_tiller_points.csv")
    stem_csv = os.path.join(out_root, "main_stem_points.csv")

    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    df_points.to_csv(points_csv, index=False, encoding="utf-8-sig")
    df_stem.to_csv(stem_csv, index=False, encoding="utf-8-sig")

    print("\n=== 完成 ===")
    print(f"统一高度带：Layer {band_start} ~ {band_end}（CC*={cc_star:.2f}）")
    print(f"汇总选择结果：{summary_csv}")
    print(f"候选分蘖点坐标：{points_csv}")
    print(f"主茎点坐标：{stem_csv}")
    print(f"每张图输出：{out_root}\\<图名>\\_selected_layer_*_cands_and_mainstem.png")


# ============================================================
# 总文件夹批处理
# ============================================================
def run_batch(
    mask_root: str,
    skel_root: str,
    out_root: str,
    n_layers: int = 15,
    bottom_frac: float = 1/3,
    base_cut_frac: float = 0.15,
    connectivity: int = 8,
    remove_short: bool = True,
    band_len: int = 5,
    save_layer_visuals: bool = True,
    w_center: float = 13.0,
    w_base: float = 8.0,
    base_seed_x_tol: Optional[int] = None,
    base_seed_x_tol_ratio: float = 0.06
):
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"mask_root 不存在: {mask_root}")
    if not os.path.isdir(skel_root):
        raise FileNotFoundError(f"skel_root 不存在: {skel_root}")
    os.makedirs(out_root, exist_ok=True)

    skel_subfolders = list_subfolders(skel_root)
    if not skel_subfolders:
        raise RuntimeError(f"skeleton 总文件夹下没有子文件夹: {skel_root}")

    print(f"共发现 {len(skel_subfolders)} 个 skeleton 子文件夹，开始批处理...")

    for sub_name in skel_subfolders:
        cur_mask_folder = os.path.join(mask_root, sub_name)
        cur_skel_folder = os.path.join(skel_root, sub_name)
        cur_out_folder = os.path.join(out_root, sub_name)

        if not os.path.isdir(cur_mask_folder):
            print(f"[跳过] mask 子文件夹不存在，无法与 skeleton 对应: {cur_mask_folder}")
            continue

        print("\n" + "=" * 80)
        print(f"[开始处理] 子文件夹: {sub_name}")
        print(f"mask_folder = {cur_mask_folder}")
        print(f"skel_folder = {cur_skel_folder}")
        print(f"out_folder  = {cur_out_folder}")

        try:
            run_pipeline(
                mask_folder=cur_mask_folder,
                skel_folder=cur_skel_folder,
                out_root=cur_out_folder,
                n_layers=n_layers,
                bottom_frac=bottom_frac,
                base_cut_frac=base_cut_frac,
                connectivity=connectivity,
                remove_short=remove_short,
                band_len=band_len,
                save_layer_visuals=save_layer_visuals,
                w_center=w_center,
                w_base=w_base,
                base_seed_x_tol=base_seed_x_tol,
                base_seed_x_tol_ratio=base_seed_x_tol_ratio
            )
            print(f"[处理完成] {sub_name}")
        except Exception as e:
            print(f"[处理失败] {sub_name}")
            print(f"原因: {e}")


# =========================
# 入口
# =========================
if __name__ == "__main__":
    mask_root = r"D:\fennie\yuantufennie-bajie-yunsui\quanburoimask\yunsui"
    skel_root = r"D:\fennie\yuantufennie-bajie-yunsui\quanburoimaskgujiawugaijin\yunsui"
    out_root = r"D:\fennie\yuantufennie-bajie-yunsui\quanburoimaskoutgujiawugaijin\yunsui"

    run_batch(
        mask_root=mask_root,
        skel_root=skel_root,
        out_root=out_root,
        n_layers=15,
        bottom_frac=1/3,
        base_cut_frac=0.15,
        connectivity=8,
        remove_short=True,
        band_len=5,
        save_layer_visuals=True,
        w_center=13.0,
        w_base=8.0,
        base_seed_x_tol=None,          # 可不填，自动按图像宽度比例
        base_seed_x_tol_ratio=0.06     # BaseSeed 限制在 x_center 附近
    )


