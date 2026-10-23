
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# -------------------------
# IO utils
# -------------------------
def get_image_width_from_path(img_path: str) -> int:
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Cannot read image to get width: {img_path}")
    return int(im.shape[1])


def _resolve_theta_column(df_ang: pd.DataFrame) -> str:
    """兼容 view_angles.csv 的角度列名"""
    cand = ["theta_deg", "theta", "angle_deg", "deg"]
    for c in cand:
        if c in df_ang.columns:
            return c
    raise ValueError(f"view_angles.csv must contain one of {cand}, got {df_ang.columns.tolist()}")


def _safe_read_csv(p: str) -> pd.DataFrame:
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p)


# -------------------------
# 按比例自动设置 min_samples
# -------------------------
def recommend_min_samples(n_views_total: int, ratio: float = 0.25, min_floor: int = 2) -> int:
    return max(min_floor, int(round(ratio * n_views_total)))


# -------------------------
# Union-Find
# -------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
            return rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
            return ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
            return ra


# -------------------------
# View distance
# -------------------------
def circular_view_dist(a: int, b: int, n_views: int) -> int:
    d = abs(a - b)
    return min(d, n_views - d)


# -------------------------
# Feature building
# -------------------------
def build_Z_dataframe(
    out_root: str,
    selection_summary_csv: str,
    candidate_points_csv: str,
    main_stem_csv: str,
    view_angles_csv: str,
    lambda_scale: float = 5.0,
):
    df_sum = _safe_read_csv(selection_summary_csv)
    df_pts = _safe_read_csv(candidate_points_csv)
    df_stm = _safe_read_csv(main_stem_csv)
    df_ang = _safe_read_csv(view_angles_csv)

    # 必需列检查
    need_cols = {"view", "skeleton_path"}
    if not need_cols.issubset(df_sum.columns):
        raise ValueError(f"selection_summary.csv must contain {need_cols}, got {df_sum.columns.tolist()}")

    if "view" not in df_ang.columns:
        raise ValueError("view_angles.csv must contain column: view")

    theta_col = _resolve_theta_column(df_ang)

    # 每个 view 获取图像宽度 W
    df_w = df_sum[["view", "skeleton_path"]].drop_duplicates().copy()
    widths = []
    for _, r in df_w.iterrows():
        sp = str(r["skeleton_path"])
        if not os.path.exists(sp):
            sp2 = os.path.join(out_root, os.path.basename(sp))
            if os.path.exists(sp2):
                sp = sp2
        W = get_image_width_from_path(sp)
        widths.append(W)
    df_w["W"] = widths


    keep_cols_stm = ["view", "stem_x"]
    if "view_idx" in df_stm.columns:
        keep_cols_stm.append("view_idx")

    df = (
        df_pts.merge(df_stm[keep_cols_stm], on="view", how="left", suffixes=("", "_stm"))
              .merge(df_w[["view", "W"]], on="view", how="left")
              .merge(df_ang[["view", theta_col]], on="view", how="left")
    )


    if "view_idx" not in df.columns:
        views_sorted = sorted(df["view"].dropna().unique().tolist())
        view_to_idx = {v: i for i, v in enumerate(views_sorted)}
        df["view_idx"] = df["view"].map(view_to_idx).astype(int)
    else:
        df["view_idx"] = df["view_idx"].astype(int)


    df = df.dropna(subset=["stem_x", "W", theta_col, "x"]).copy()
    df["W"] = df["W"].astype(float)
    df["theta_deg"] = df[theta_col].astype(float)


    df["X_tilde"] = (df["x"].astype(float) - df["stem_x"].astype(float)) / df["W"]

    # 计算 Z = [cosθ, sinθ, λX~]
    theta = np.deg2rad(df["theta_deg"].to_numpy())
    df["Zx"] = np.cos(theta)
    df["Zy"] = np.sin(theta)
    df["Zz"] = lambda_scale * df["X_tilde"].astype(float)

    return df


# -------------------------
# Scaling / weighting
# -------------------------
def make_feature_matrix(
    df: pd.DataFrame,
    scaler_mode: str = "pre",
    w_xy: float = 1.0,
    w_z: float = 1.0,
):

    Z = df[["Zx", "Zy", "Zz"]].to_numpy(dtype=float)

    if scaler_mode == "none":
        Z[:, 0] *= w_xy
        Z[:, 1] *= w_xy
        Z[:, 2] *= w_z
        return Z

    if scaler_mode == "pre":
        scaler = StandardScaler()
        Zn = scaler.fit_transform(Z)
        Zn[:, 0] *= w_xy
        Zn[:, 1] *= w_xy
        Zn[:, 2] *= w_z
        return Zn

    if scaler_mode == "post":
        Z[:, 0] *= w_xy
        Z[:, 1] *= w_xy
        Z[:, 2] *= w_z
        scaler = StandardScaler()
        return scaler.fit_transform(Z)

    raise ValueError("scaler_mode must be one of: none, pre, post")


# -------------------------
# k-distance helper
# -------------------------
def plot_k_distance(X: np.ndarray, k: int, save_path: str = None, show: bool = True):
    if X.shape[0] < k:
        print(f"[WARN] Not enough points for k-distance: n={X.shape[0]} < k={k}")
        return None

    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    dists, _ = nbrs.kneighbors(X)
    kdist = np.sort(dists[:, -1])

    plt.figure(figsize=(7, 4))
    plt.plot(kdist)
    plt.title(f"k-distance curve (k={k})")
    plt.xlabel("Points sorted")
    plt.ylabel(f"{k}-NN distance")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved k-distance plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    q90 = float(np.quantile(kdist, 0.90))
    q95 = float(np.quantile(kdist, 0.95))
    return {"q90": q90, "q95": q95}


# -------------------------
# Constrained DBSCAN-like clustering
# -------------------------
def constrained_dbscan_cluster(
    df: pd.DataFrame,
    X: np.ndarray,
    eps: float,
    min_samples: int,
    max_view_gap: int = None,
):

    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    if "view_idx" not in df.columns:
        raise ValueError("df must contain column 'view_idx' for constrained clustering.")

    view_idx = df["view_idx"].to_numpy(dtype=int)
    n_views = int(df["view_idx"].max()) + 1

    # 1) 构建邻接表：只允许跨视角建立边
    neighbors = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if view_idx[i] == view_idx[j]:
                continue  # 同视角点绝不直接连边

            if max_view_gap is not None:
                if circular_view_dist(view_idx[i], view_idx[j], n_views) > max_view_gap:
                    continue

            d = np.linalg.norm(X[i] - X[j])
            if d <= eps:
                neighbors[i].append(j)
                neighbors[j].append(i)

    # 2) 核心点判定：按不同视角支持数判定
    is_core = np.zeros(n, dtype=bool)
    for i in range(n):
        support_views = {view_idx[i]}
        for j in neighbors[i]:
            support_views.add(view_idx[j])
        if len(support_views) >= min_samples:
            is_core[i] = True

    # 3) 核心点先聚成簇，簇合并时检查“视角冲突”
    uf = UnionFind(n)
    cluster_used_views = {i: {view_idx[i]} for i in range(n)}

    core_edges = []
    for i in range(n):
        if not is_core[i]:
            continue
        for j in neighbors[i]:
            if j <= i or (not is_core[j]):
                continue
            d = np.linalg.norm(X[i] - X[j])
            core_edges.append((d, i, j))

    core_edges.sort(key=lambda x: x[0])

    for d, i, j in core_edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue

        used_i = cluster_used_views[ri]
        used_j = cluster_used_views[rj]

        # 两个簇中若已有相同视角，则禁止合并
        if len(used_i & used_j) > 0:
            continue

        new_root = uf.union(ri, rj)
        if new_root == ri:
            cluster_used_views[ri] = used_i | used_j
            cluster_used_views.pop(rj, None)
        else:
            cluster_used_views[rj] = used_i | used_j
            cluster_used_views.pop(ri, None)


    labels = np.full(n, -1, dtype=int)
    root_used_views = {}

    for i in range(n):
        if is_core[i]:
            r = uf.find(i)
            labels[i] = r
            root_used_views.setdefault(r, set()).add(view_idx[i])


    for i in range(n):
        if is_core[i]:
            continue

        candidates = []
        for j in neighbors[i]:
            if not is_core[j]:
                continue

            r = uf.find(j)

            # 该簇中已有同视角点，则不能加入
            if view_idx[i] in root_used_views[r]:
                continue

            d = np.linalg.norm(X[i] - X[j])
            candidates.append((d, r))

        if len(candidates) == 0:
            continue

        candidates.sort(key=lambda t: t[0])
        best_root = candidates[0][1]
        labels[i] = best_root
        root_used_views[best_root].add(view_idx[i])


    root_to_members = {}
    for i, lb in enumerate(labels):
        if lb == -1:
            continue
        root_to_members.setdefault(lb, []).append(i)

    root_to_new = {}
    new_id = 0
    for r, members in root_to_members.items():
        used_views = {view_idx[m] for m in members}
        if len(used_views) >= min_samples:
            root_to_new[r] = new_id
            new_id += 1

    final_labels = np.full(n, -1, dtype=int)
    for i, lb in enumerate(labels):
        if lb in root_to_new:
            final_labels[i] = root_to_new[lb]

    return final_labels


def dbscan_cluster(df: pd.DataFrame, X: np.ndarray, eps: float, min_samples: int, max_view_gap: int = None):
    labels = constrained_dbscan_cluster(
        df=df,
        X=X,
        eps=eps,
        min_samples=min_samples,
        max_view_gap=max_view_gap
    )
    return labels


# -------------------------
# Diagnostics
# -------------------------
def cluster_stats_and_diagnose(df: pd.DataFrame, labels: np.ndarray, theta_step_deg: float = None):
    df2 = df.copy()
    df2["cluster"] = labels

    stat = (
        df2[df2["cluster"] >= 0]
        .groupby("cluster")
        .agg(
            n_points=("cluster", "size"),
            n_views=("view", "nunique"),
            theta_min=("theta_deg", "min"),
            theta_max=("theta_deg", "max"),
            z_mean=("Zz", "mean"),
            z_std=("Zz", "std"),
        )
        .reset_index()
        .sort_values(["n_views", "n_points"], ascending=False)
    )

    if theta_step_deg is not None and theta_step_deg > 0 and len(stat) > 0:
        gap_flags = []
        for cid in stat["cluster"].tolist():
            thetas = np.sort(df2.loc[df2["cluster"] == cid, "theta_deg"].unique())
            if thetas.size <= 2:
                gap_flags.append(False)
                continue
            gaps = np.diff(thetas)
            has_gap = bool(np.any(gaps > 1.5 * theta_step_deg))
            gap_flags.append(has_gap)
        stat["has_theta_gap"] = gap_flags
    else:
        stat["has_theta_gap"] = np.nan

    return df2, stat


def validate_view_exclusive_constraint(df_labeled: pd.DataFrame):
    bad_rows = []
    valid = df_labeled[df_labeled["cluster"] >= 0].copy()

    for cid, sub in valid.groupby("cluster"):
        vc = sub["view_idx"].value_counts()
        repeated = vc[vc > 1]
        if len(repeated) > 0:
            bad_rows.append({
                "cluster": int(cid),
                "bad_views": ",".join(map(str, repeated.index.tolist())),
                "counts": ",".join(map(str, repeated.tolist()))
            })

    if len(bad_rows) == 0:
        print("[OK] 视角互斥约束满足：每个簇内每个视角至多一个点。")
        return pd.DataFrame(columns=["cluster", "bad_views", "counts"])

    df_bad = pd.DataFrame(bad_rows)
    print("[WARN] 发现违反视角互斥约束的簇：")
    print(df_bad)
    return df_bad


# -------------------------
# ✅ 高区分度离散配色：tab20 + tab20b + tab20c，不够再用固定seed的HSV补足
# -------------------------
def _get_distinct_colors(n: int, seed: int = 123):
    """
    返回 n 个“尽量差异大”的RGBA颜色，不重复。
    1) 优先使用离散调色板 tab20/tab20b/tab20c (最多 60)
    2) 超过 60 后，用固定 seed 的 HSV 随机打散补足
    """
    if n <= 0:
        return []

    colors = []

    base_maps = ["tab20", "tab20b", "tab20c"]
    for name in base_maps:
        cmap = plt.cm.get_cmap(name)
        for i in range(cmap.N):
            colors.append(cmap(i))

    if n <= len(colors):
        return colors[:n]

    rng = np.random.default_rng(seed)
    need = n - len(colors)

    hs = np.linspace(0, 1, max(need * 3, 180), endpoint=False)
    rng.shuffle(hs)

    extra = []
    for h in hs:
        rgba = plt.cm.hsv(h)
        extra.append(rgba)
        if len(extra) >= need:
            break

    colors.extend(extra)
    return colors[:n]


def plot_clusters_3d(df: pd.DataFrame, title: str, save_path: str = None, show: bool = True):
    labels = df["cluster"].to_numpy()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    uniq = np.unique(labels)

    cluster_labels = sorted([lb for lb in uniq.tolist() if lb != -1])
    n_clusters = len(cluster_labels)

    palette = _get_distinct_colors(n_clusters, seed=123)
    color_map = {lb: palette[i] for i, lb in enumerate(cluster_labels)}

    if -1 in uniq:
        m = (labels == -1)
        ax.scatter(
            df.loc[m, "Zx"], df.loc[m, "Zy"], df.loc[m, "Zz"],
            s=10, alpha=0.25, c="#bdbdbd", label="noise"
        )

    for lb in cluster_labels:
        m = (labels == lb)
        ax.scatter(
            df.loc[m, "Zx"], df.loc[m, "Zy"], df.loc[m, "Zz"],
            s=18, alpha=0.9, color=color_map[lb], label=f"C{lb}"
        )

    ax.set_xlabel("cos(theta)")
    ax.set_ylabel("sin(theta)")
    ax.set_zlabel("lambda * X_tilde")
    ax.set_title(title)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    handles, texts = ax.get_legend_handles_labels()
    ax.legend(handles, texts, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved cluster plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -------------------------
# eps sweep (auto)
# -------------------------
def eps_sweep(
    df: pd.DataFrame,
    X: np.ndarray,
    eps_list: list,
    min_samples: int,
    theta_step_deg: float = None,
    save_csv: str = None,
    max_view_gap: int = None,
):
    rows = []
    for eps in eps_list:
        labels = dbscan_cluster(df, X, eps=eps, min_samples=min_samples, max_view_gap=max_view_gap)
        n = labels.size
        n_noise = int(np.sum(labels == -1))
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))

        df2, stat = cluster_stats_and_diagnose(df, labels, theta_step_deg=theta_step_deg)
        mean_views = float(stat["n_views"].mean()) if len(stat) else 0.0
        max_views = int(stat["n_views"].max()) if len(stat) else 0
        mean_pts = float(stat["n_points"].mean()) if len(stat) else 0.0
        max_pts = int(stat["n_points"].max()) if len(stat) else 0
        gap_ratio = float(np.mean(stat["has_theta_gap"].fillna(False))) if len(stat) else 0.0

        rows.append({
            "eps": float(eps),
            "min_samples": int(min_samples),
            "n_points_total": int(n),
            "n_clusters": int(n_clusters),
            "noise_ratio": float(n_noise / max(1, n)),
            "mean_cluster_views": float(mean_views),
            "max_cluster_views": int(max_views),
            "mean_cluster_points": float(mean_pts),
            "max_cluster_points": int(max_pts),
            "theta_gap_ratio": float(gap_ratio),
        })

    df_sweep = pd.DataFrame(rows).sort_values("eps")

    if save_csv is not None:
        df_sweep.to_csv(save_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved eps sweep to: {save_csv}")

    return df_sweep


# -------------------------
# AUTO eps selection (scoring)
# -------------------------
def select_best_eps_from_sweep(
    df_sweep: pd.DataFrame,
    n_views_total: int,
    min_clusters: int = 6,
    max_clusters: int = 25,

    w_under: float = 3.0,
    w_over: float = 0.6,
    w_noise: float = 2.0,
    w_gap: float = 1.0,
    w_big: float = 0.8,
):
    df = df_sweep.copy()
    scores = []

    for _, r in df.iterrows():
        k = float(r["n_clusters"])
        noise = float(r["noise_ratio"])
        gap = float(r["theta_gap_ratio"])
        max_views = float(r["max_cluster_views"])

        under_pen = 0.0
        over_pen = 0.0
        if k < min_clusters:
            under_pen = (min_clusters - k) / max(1.0, min_clusters)
        if k > max_clusters:
            over_pen = (k - max_clusters) / max(1.0, max_clusters)

        big_pen = 0.0
        if n_views_total > 0:
            big_pen = max(0.0, (max_views / n_views_total) - 0.85)

        score = (
            w_under * under_pen +
            w_over * over_pen +
            w_noise * noise +
            w_gap * gap +
            w_big * big_pen
        )
        scores.append(float(score))

    df["score"] = scores

    df_sorted = df.sort_values(
        ["score", "noise_ratio", "theta_gap_ratio", "eps"],
        ascending=[True, True, True, True]
    )
    best_eps = float(df_sorted.iloc[0]["eps"])
    return best_eps, df_sorted


# -------------------------
# eps_list auto builder
# -------------------------
def build_eps_list_auto(kd=None, fallback=(0.90, 1.60), step=0.05):
    if kd is not None and ("q90" in kd) and ("q95" in kd):
        lo = max(0.05, 0.50 * float(kd["q90"]))
        hi = 1.15 * float(kd["q95"])
    else:
        lo, hi = float(fallback[0]), float(fallback[1])

    if hi <= lo:
        hi = lo + step

    arr = np.arange(lo, hi + 1e-9, step)
    eps_list = [float(f"{x:.2f}") for x in arr]
    eps_list = list(dict.fromkeys(eps_list))
    return eps_list


# -------------------------
# Main pipeline
# -------------------------
def run_all(
    out_root: str,
    selection_summary_csv: str,
    candidate_points_csv: str,
    main_stem_csv: str,
    view_angles_csv: str,

    lambda_scale: float = 5.0,

    eps: float = None,
    min_samples: int = 4,

    scaler_mode: str = "pre",
    w_xy: float = 1.0,
    w_z: float = 4.0,

    do_eps_sweep: bool = True,
    eps_list: list = None,

    do_kdist: bool = True,

    theta_step_deg: float = 30.0,

    auto_select_eps: bool = True,
    min_clusters: int = 6,
    max_clusters: int = 25,

    max_view_gap: int = None,

    show: bool = True
):
    os.makedirs(out_root, exist_ok=True)

    df = build_Z_dataframe(
        out_root=out_root,
        selection_summary_csv=selection_summary_csv,
        candidate_points_csv=candidate_points_csv,
        main_stem_csv=main_stem_csv,
        view_angles_csv=view_angles_csv,
        lambda_scale=lambda_scale
    )

    n_views_total = int(df["view"].nunique())
    print("unique views in points:", n_views_total)
    print("unique theta_deg:", df["theta_deg"].nunique())
    print("theta_deg unique values (sorted head):", np.sort(df["theta_deg"].unique())[:20])


    min_samples = recommend_min_samples(n_views_total, ratio=0.25, min_floor=2)
    print(f"[AUTO] min_samples = max(2, round(0.25 * {n_views_total})) = {min_samples}")

    z_csv = os.path.join(out_root, f"tiller_points_with_Z_lambda{lambda_scale}.csv")
    df.to_csv(z_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved Z points to: {z_csv}")

    X = make_feature_matrix(df, scaler_mode=scaler_mode, w_xy=w_xy, w_z=w_z)

    kd = None
    if do_kdist:
        kdist_path = os.path.join(out_root, f"kdist_k{min_samples}_mode{scaler_mode}_wz{w_z}.png")
        kd = plot_k_distance(X, k=min_samples, save_path=kdist_path, show=show)
        if kd is not None:
            print(f"[INFO] k-distance quantiles: q90={kd['q90']:.3f}, q95={kd['q95']:.3f}")
            print("[HINT] eps 可以围绕 q90~q95 选一个扫描区间。")

    df_sweep_scored = None
    best_eps = None

    need_sweep = do_eps_sweep or (auto_select_eps and eps is None)

    if need_sweep:
        if eps_list is None:
            eps_list = build_eps_list_auto(
                kd,
                fallback=(0.60, 1.60),
                step=0.05
            )
            print(f"[AUTO] eps_list generated: [{eps_list[0]} ... {eps_list[-1]}], n={len(eps_list)}")

        sweep_csv = os.path.join(out_root, f"eps_sweep_ms{min_samples}_mode{scaler_mode}_wz{w_z}.csv")
        df_sweep = eps_sweep(
            df=df,
            X=X,
            eps_list=eps_list,
            min_samples=min_samples,
            theta_step_deg=theta_step_deg,
            save_csv=sweep_csv,
            max_view_gap=max_view_gap
        )

        best_eps, df_sweep_scored = select_best_eps_from_sweep(
            df_sweep,
            n_views_total=n_views_total,
            min_clusters=min_clusters,
            max_clusters=max_clusters
        )

        scored_csv = os.path.join(out_root, f"eps_sweep_scored_ms{min_samples}_mode{scaler_mode}_wz{w_z}.csv")
        df_sweep_scored.to_csv(scored_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved scored sweep to: {scored_csv}")

        print("\n[EPS SWEEP SCORED TOP 10] (score 越小越好)")
        print(df_sweep_scored.head(10).to_string(index=False))
        print(f"\n[AUTO] best_eps = {best_eps:.3f} (min_clusters={min_clusters}, max_clusters={max_clusters})")

    if (eps is not None) and (not auto_select_eps):
        best_eps = float(eps)

    if best_eps is None:
        if eps is not None:
            best_eps = float(eps)
        elif kd is not None:
            best_eps = float(kd["q90"])
        else:
            best_eps = 0.65

    labels = dbscan_cluster(df, X, eps=best_eps, min_samples=min_samples, max_view_gap=max_view_gap)
    df_labeled, stat = cluster_stats_and_diagnose(df, labels, theta_step_deg=theta_step_deg)

    labeled_csv = os.path.join(out_root, f"tiller_points_with_labels_eps{best_eps:.3f}_ms{min_samples}.csv")
    stat_csv = os.path.join(out_root, f"cluster_stats_eps{best_eps:.3f}_ms{min_samples}.csv")
    df_labeled.to_csv(labeled_csv, index=False, encoding="utf-8-sig")
    stat.to_csv(stat_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved labeled points: {labeled_csv}")
    print(f"[OK] Saved cluster stats:  {stat_csv}")

    df_bad = validate_view_exclusive_constraint(df_labeled)
    bad_csv = os.path.join(out_root, f"constraint_check_eps{best_eps:.3f}_ms{min_samples}.csv")
    df_bad.to_csv(bad_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved constraint check: {bad_csv}")

    plot_path = os.path.join(out_root, f"cluster_plot_eps{best_eps:.3f}_ms{min_samples}.png")
    title = (
        f"Constrained DBSCAN in Z space "
        f"(lambda={lambda_scale}, eps={best_eps:.3f}, min_samples={min_samples}, "
        f"mode={scaler_mode}, w_z={w_z})"
    )
    plot_clusters_3d(df_labeled, title=title, save_path=plot_path, show=show)

    return df_labeled, stat, best_eps, df_sweep_scored


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    out_root = r"D:\fennie\yuantufennie-bajie-yunsui\xinzengbajie\yuantu\xiandaoroimaskgujiahua-out\VID_20250727_164427-12shijiao_roi"

    selection_summary_csv = os.path.join(out_root, "selection_summary.csv")
    candidate_points_csv  = os.path.join(out_root, "candidate_tiller_points.csv")
    main_stem_csv         = os.path.join(out_root, "main_stem_points.csv")
    view_angles_csv       = os.path.join(out_root, "view_angles.csv")

    run_all(
        out_root=out_root,
        selection_summary_csv=selection_summary_csv,
        candidate_points_csv=candidate_points_csv,
        main_stem_csv=main_stem_csv,
        view_angles_csv=view_angles_csv,

        lambda_scale=5.0,

        eps=None,
        min_samples=4,

        scaler_mode="pre",
        w_xy=1.0,
        w_z=4.0,

        do_kdist=True,
        do_eps_sweep=True,
        eps_list=None,

        theta_step_deg=30.0,

        auto_select_eps=True,
        min_clusters=6,
        max_clusters=25,

        max_view_gap=None,

        show=True
    )
