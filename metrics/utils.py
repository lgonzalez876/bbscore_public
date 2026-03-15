from jepa.src.models.attentive_pooler import AttentiveClassifier
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added for F.relu
from typing import Dict, Callable, List, Optional, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

import math
import numpy as np
import torch


def pearson_correlation_scorer(y_true, y_pred, eps=1e-8):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # if either has (almost) zero variance, define a default:
    if var_true < eps or var_pred < eps:
        # perfectly matching constant signals → perfect score
        if np.allclose(y_true, y_pred, atol=eps):
            return 1.0
        # otherwise no meaningful correlation
        return 0.0

    # otherwise safe to compute Pearson
    r, _ = pearsonr(y_true, y_pred)
    # then apply your transform
    return (2 * r) / (1 + r) if r != 1 else 1.0


def run_kfold_cv(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    scoring_funcs: Dict[str, Callable],
    n_splits: int = 10,
    random_state: int = 42,
    stratify_on: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if stratify_on is not None:
        if len(stratify_on) != X.shape[0]:
            raise ValueError(
                "Length of stratify_on must match X and y for StratifiedKFold.")
        print(f"Using Stratified K-Fold with {n_splits} splits.")
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)
        split_iterator = kf.split(X, stratify_on)  # Pass stratification target
    else:
        print(f"Using Standard K-Fold with {n_splits} splits.")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iterator = kf.split(X)  # Standard split

    scores = {name: [] for name in scoring_funcs}
    # scores['preds'] = []
    # scores['gt'] = []

    # Wrap the fold loop with tqdm progress bar
    for fold_idx, (train_idx, val_idx) in enumerate(tqdm(split_iterator, total=n_splits, desc="Folds")):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_factory()
        if model is not None:  # for metrics like onetoone that don't use any model
            model.fit(X_train, y_train)

            fold_preds = model.predict(X_val)
            # If fold_preds is a torch.Tensor, convert to numpy array
            if hasattr(fold_preds, 'cpu'):
                fold_preds = fold_preds.cpu().numpy()
        else:
            fold_preds = X_val  # in cases where no model is used.

        # scores['preds'].append(fold_preds)
        # scores['gt'].append(y_val)

        for name, scoring_func in scoring_funcs.items():
            if model is not None:
                fold_score = scoring_func(y_val, fold_preds)
            else:
                fold_score = scoring_func(X_val, y_val)

            if isinstance(fold_score, (float, int, np.number)):
                scores[name].append(fold_score)
            else:
                scores[name].append(np.array(scoring_func(y_val, fold_preds)))

        # Free model after each fold to avoid accumulation
        del model

    return {name: np.array(score_list, dtype=object) for name, score_list in scores.items()}


def run_kfold_cv_chunked(
    model_factory: List[Callable],
    X: np.ndarray,
    y: np.ndarray,
    scoring_funcs: Dict[str, Callable],
    chunk_size: int = 4000,  # Number of output features processed at a time
    n_splits: int = 10,
    random_state: int = 42,
    stratify_on: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:

    if stratify_on is not None:
        if len(stratify_on) != X.shape[0]:
            raise ValueError(
                "Length of stratify_on must match X and y for StratifiedKFold.")
        print(f"Using Stratified K-Fold (Chunked) with {n_splits} splits.")
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)
        split_iterator = kf.split(X, stratify_on)  # Pass stratification target
    else:
        print(f"Using Standard K-Fold (Chunked) with {n_splits} splits.")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iterator = kf.split(X)  # Standard split

    scores = {name: [] for name in scoring_funcs}
    # scores['preds'] = []
    # scores['gt'] = []

    # Compute number of output chunks
    num_chunks = math.ceil(y.shape[1] / chunk_size)

    # Wrap the fold loop with tqdm progress bar
    for train_idx, val_idx in tqdm(split_iterator, total=n_splits, desc="Folds"):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Initialize prediction storage
        y_pred_chunks = np.zeros_like(y_val)

        # Process the output in chunks
        for i in tqdm(range(num_chunks), total=num_chunks, desc="Chunks", leave=False):
            start_col = i * chunk_size
            # Ensure not exceeding bounds
            end_col = min((i + 1) * chunk_size, y.shape[1])
            y_train_chunk = y_train[:, start_col:end_col]
            # y_val_chunk = y_val[:, start_col:end_col] # Not needed for prediction step

            model = model_factory()
            if model is not None:
                model.fit(X_train, y_train_chunk)
                y_pred_chunks[:, start_col:end_col] = model.predict(X_val)
            else:
                # No model case
                y_pred_chunks[:, start_col:end_col] = X_val

        # scores['preds'].append(y_pred_chunks)
        # scores['gt'].append(y_val)

        # Compute scores for each metric
        for name, scoring_func in scoring_funcs.items():
            if model is not None:
                fold_score = scoring_func(y_val, y_pred_chunks)
            else:
                fold_score = scoring_func(X_val, y_val)

            if isinstance(fold_score, (float, int, np.number)):
                scores[name].append(fold_score)
            else:
                scores[name].append(
                    np.array(scoring_func(y_val, y_pred_chunks)))

    return {name: np.array(score_list, dtype=object) for name, score_list in scores.items()}


def run_eval(
    model_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scoring_funcs: Dict[str, Callable],
    save_weights: bool = True,
) -> Dict[str, np.ndarray]:
    scores = {name: [] for name in scoring_funcs}

    # Create and train model
    model = model_factory()
    if model is not None:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    else:
        # No model case
        y_pred = X_val

    # scores['preds'] = y_pred
    # scores['gt'] = y_val

    # Save weights if present
    if save_weights and model is not None and hasattr(model, "coef_"):
        coef = model.coef_
        intercept = getattr(model, "intercept_", None)

        # Handle Torch tensors vs NumPy arrays
        is_torch = isinstance(coef, torch.Tensor)

        if is_torch:
            coef = coef.detach().cpu().numpy()
            if intercept is not None and isinstance(intercept, torch.Tensor):
                intercept = intercept.detach().cpu().numpy()

        scores["coef"] = coef
        if intercept is not None:
            scores["intercept"] = intercept

        # Optional: also store the chosen alpha, handling both APIs
        alpha = getattr(model, "alpha_", None)  # sklearn RidgeCV
        if alpha is None:
            alpha = getattr(model, "best_alpha", None)  # your TorchRidgeCV
        if alpha is not None:
            scores["alpha"] = np.array(alpha)

    # Compute scores for each metric
    for name, scoring_func in scoring_funcs.items():
        if model is not None:
            fold_score = scoring_func(y_val, y_pred)
        else:
            fold_score = scoring_func(X_val, y_val)

        if isinstance(fold_score, (float, int, np.number)):
            scores[name].append(fold_score)
        else:
            scores[name].append(np.array(fold_score))

    # Convert only "metric" entries (lists) to arrays; leave arrays as-is
    out = {}
    for name, value in scores.items():
        if isinstance(value, list):
            out[name] = np.array(value)
        else:
            out[name] = value

    return out


def run_eval_chunked(
    model_factory: List[Callable],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scoring_funcs: Dict[str, Callable],
    chunk_size: int = 4000,  # Number of output features processed at a time
) -> Dict[str, np.ndarray]:

    scores = {name: [] for name in scoring_funcs}

    # Compute number of output chunks
    num_chunks = math.ceil(y_train.shape[1] / chunk_size)

    # Initialize prediction storage
    y_pred_chunks = np.zeros_like(y_val)

    # Process the output in chunks
    for i in tqdm(range(num_chunks), total=num_chunks, desc="Chunks", leave=False):
        y_train_chunk = y_train[:, i * chunk_size: (i + 1) * chunk_size]
        y_val_chunk = y_val[:, i * chunk_size: (i + 1) * chunk_size]

        model = model_factory()
        if model is not None:
            model.fit(X_train, y_train_chunk)
            y_pred_chunks[:, i *
                          chunk_size: (i + 1) * chunk_size] = model.predict(X_val)
        else:
            # No model case
            y_pred_chunks[:, i * chunk_size: (i + 1) * chunk_size] = X_val

    # scores['preds'] = [y_pred_chunks]
    # scores['gt'] = [y_val]

    # Compute scores for each metric
    for name, scoring_func in scoring_funcs.items():
        if model is not None:
            fold_score = scoring_func(y_val, y_pred_chunks)
        else:
            fold_score = scoring_func(X_val, y_val)

        if isinstance(fold_score, (float, int, np.number)):
            scores[name].append(fold_score)
        else:
            scores[name].append(np.array(scoring_func(y_val, y_pred_chunks)))

    return {name: np.array(score_list) for name, score_list in scores.items()}


class TorchRidge:
    def __init__(self, alpha=1.0, solver="cholesky", fit_intercept=True):
        if solver not in ["cholesky", "lsqr"]:
            raise ValueError("Solver must be 'cholesky' or 'lsqr'.")

        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        X, y = (
            X.to(self.device).clone().detach().float(),
            y.to(self.device).clone().detach().float(),
        )

        if self.fit_intercept:
            X_mean = X.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)
            X = X - X_mean
            y = y - y_mean
        else:
            X_mean, y_mean = 0, 0

        n_samples, n_features = X.shape
        I = torch.eye(n_features, dtype=torch.float32, device=self.device)
        ridge_term = self.alpha * I

        XtX = X.T @ X
        XtY = X.T @ y
        XtX_ridge = XtX + ridge_term

        if self.solver == "cholesky":
            self.coef_ = torch.linalg.solve(XtX_ridge, XtY)
        elif self.solver == "lsqr":
            self.coef_, _ = torch.linalg.lstsq(XtX_ridge, XtY)[:2]

        self.intercept_ = y_mean - X_mean @ self.coef_

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        if isinstance(X, np.ndarray):
            # Convert to torch.float32
            X = torch.from_numpy(X).to(torch.float32)

        X = X.to(self.device)
        return X @ self.coef_ + self.intercept_


class TorchRidgeCV:
    """
    Optimized for 40GB+ GPUs (A40/A100).
    Performs internal validation split to select alpha, then refits on full data.

    Automatic Switching:
    - If Samples >= Features (Tall): Uses Primal Solver (X.T @ X)
    - If Features > Samples (Fat):  Uses Dual Solver (X @ X.T)
    - Handles massive weight matrices by offloading to CPU with Dynamic Chunking.
    """

    def __init__(self, alphas: List[float], device="cuda", val_fraction=0.1):
        self.alphas = alphas
        self.best_alpha = None
        self.coef_ = None
        self.intercept_ = None
        self.device = device
        self.val_fraction = val_fraction

    def _get_safe_chunk_size(self, n_features, dtype_size=4, safety_margin=0.8):
        """
        Dynamically calculate max chunk size based on free GPU memory.
        Result Matrix Size = (n_features * chunk_size * dtype_size).
        """
        if self.device == "cpu":
            return 1024  # Default for CPU

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            # Reserve memory for workspace and safety
            usable_bytes = free_bytes * safety_margin

            bytes_per_column = n_features * dtype_size

            if bytes_per_column == 0:
                return 1024

            max_chunk = int(usable_bytes / bytes_per_column)
            # Ensure at least 1, cap at reasonable number to avoid other limits
            return max(1, min(max_chunk, 8192))
        except Exception:
            return 512  # Fallback

    def fit(self, X, y):
        # Expect X, y to be Numpy arrays or Tensors
        n_samples = X.shape[0]

        print(
            f"   [TorchRidge] Moving full data ({n_samples} samples) to {self.device} (Float32)...")

        # 1. Move Full Data to GPU (Float32)
        X_full = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_full = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        # 2. In-Place Centering
        X_mean = X_full.mean(dim=0, keepdim=True)
        y_mean = y_full.mean(dim=0, keepdim=True)

        print("   [TorchRidge] Centering data in-place...")
        X_full.sub_(X_mean)
        y_full.sub_(y_mean)

        n_features = X_full.shape[1]

        # —————————————————————————————————————————————————————————————————————
        # BRANCH A: DUAL SOLVER (Fat Matrix: Features > Samples)
        # —————————————————————————————————————————————————————————————————————
        if n_features > n_samples:
            print(
                f"   [TorchRidge] Matrix is Fat (D={n_features} > N={n_samples}). Using DUAL solver (Kernel).")

            # Validation Split
            n_val = int(n_samples * self.val_fraction)
            perm = torch.randperm(n_samples, device=self.device)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            X_train = X_full[train_idx]
            y_train = y_full[train_idx]
            X_val = X_full[val_idx]
            y_val = y_full[val_idx]

            print("   [TorchRidge] Computing Kernel (K_train)...")
            # K is tiny (N x N)
            K_train = X_train @ X_train.T
            K_cross = X_val @ X_train.T

            n_tr = K_train.shape[0]
            I = torch.eye(n_tr, dtype=torch.float32, device=self.device)

            best_score = -float('inf')
            self.best_alpha = self.alphas[-1]

            print(f"   [TorchRidge] Scanning {len(self.alphas)} alphas...")
            for alpha in self.alphas:
                # Solve: (K + alpha*I) alpha_vec = y
                solver_matrix = K_train + alpha * I
                try:
                    dual_coef = torch.linalg.solve(solver_matrix, y_train)
                except RuntimeError:
                    dual_coef = torch.linalg.lstsq(
                        solver_matrix, y_train).solution

                # Predict: y_pred = K_cross @ dual_coef
                preds_val = K_cross @ dual_coef

                # Score R2
                ss_res = ((y_val - preds_val) ** 2).sum(0)
                ss_tot = (y_val ** 2).sum(0)
                r2 = (1 - ss_res / (ss_tot + 1e-6)).mean().item()

                if r2 > best_score:
                    best_score = r2
                    self.best_alpha = alpha

            print(
                f"   [TorchRidge] Best Alpha: {self.best_alpha} (Val R2: {best_score:.4f})")

            # Refit on Full Data
            print("   [TorchRidge] Refitting on FULL data (Dual)...")
            K_full = X_full @ X_full.T
            I_full = torch.eye(
                n_samples, dtype=torch.float32, device=self.device)

            solver_matrix = K_full + self.best_alpha * I_full
            try:
                final_dual_coef = torch.linalg.solve(solver_matrix, y_full)
            except RuntimeError:
                final_dual_coef = torch.linalg.lstsq(
                    solver_matrix, y_full).solution

            # —————————————————————————————————————————————————————————————
            # RECOVER PRIMAL WEIGHTS (Dynamic Chunking)
            # w = X.T @ dual_coef
            # —————————————————————————————————————————————————————————————
            print("   [TorchRidge] Recovering primal weights (Chunked to CPU)...")
            n_targets = y_full.shape[1]

            # Initialize CPU tensor for weights
            self.coef_ = torch.zeros(
                (n_features, n_targets), dtype=torch.float32, device='cpu')

            # Dynamically find chunk size based on remaining VRAM
            # We need to store (Features x Chunk) float32 matrix
            chunk_size = self._get_safe_chunk_size(n_features)
            print(
                f"   [TorchRidge] Dynamic Chunk Size: {chunk_size} targets per batch")

            for i in range(0, n_targets, chunk_size):
                end = min(i + chunk_size, n_targets)

                # Slice dual coeffs for this batch (N x Chunk)
                dual_chunk = final_dual_coef[:, i:end]

                # Compute (D, N) @ (N, Chunk) -> (D, Chunk)
                # This operation happens on GPU
                w_chunk = X_full.T @ dual_chunk

                # Move to CPU immediately
                self.coef_[:, i:end] = w_chunk.cpu()

                del w_chunk, dual_chunk
                torch.cuda.empty_cache()

            del K_train, K_cross, K_full, X_train, X_val, y_train, y_val, final_dual_coef

        # —————————————————————————————————————————————————————————————————————
        # BRANCH B: PRIMAL SOLVER (Tall Matrix: Samples >= Features)
        # —————————————————————————————————————————————————————————————————————
        else:
            print(
                f"   [TorchRidge] Matrix is Tall (N={n_samples} >= D={n_features}). Using PRIMAL solver.")

            n_val = int(n_samples * self.val_fraction)
            perm = torch.randperm(n_samples, device=self.device)
            val_idx = perm[:n_val]

            X_val = X_full[val_idx]
            y_val = y_full[val_idx]

            XtX_full = X_full.T @ X_full
            XtY_full = X_full.T @ y_full

            XtX_val = X_val.T @ X_val
            XtY_val = X_val.T @ y_val

            XtX_train = XtX_full - XtX_val
            XtY_train = XtY_full - XtY_val

            I = torch.eye(n_features, dtype=torch.float32, device=self.device)
            best_score = -float('inf')
            self.best_alpha = self.alphas[-1]

            print(f"   [TorchRidge] Scanning {len(self.alphas)} alphas...")
            for alpha in self.alphas:
                ridge_term = alpha * I
                solver_matrix = XtX_train + ridge_term
                try:
                    coef = torch.linalg.solve(solver_matrix, XtY_train)
                except RuntimeError:
                    coef = torch.linalg.lstsq(
                        solver_matrix, XtY_train).solution

                preds_val = X_val @ coef

                ss_res = ((y_val - preds_val) ** 2).sum(0)
                ss_tot = (y_val ** 2).sum(0)
                r2 = (1 - ss_res / (ss_tot + 1e-6)).mean().item()

                if r2 > best_score:
                    best_score = r2
                    self.best_alpha = alpha

            print(
                f"   [TorchRidge] Best Alpha: {self.best_alpha} (Val R2: {best_score:.4f})")

            print("   [TorchRidge] Refitting on FULL data (Primal)...")
            solver_matrix = XtX_full + (self.best_alpha * I)
            try:
                self.coef_ = torch.linalg.solve(solver_matrix, XtY_full)
            except RuntimeError:
                self.coef_ = torch.linalg.lstsq(
                    solver_matrix, XtY_full).solution

            # Move coef to CPU
            self.coef_ = self.coef_.cpu()

            del XtX_full, XtY_full, XtX_train, XtY_train, XtX_val, XtY_val, X_val, y_val

        # —————————————————————————————————————————————————————————————————————
        # COMMON: Intercept & Cleanup
        # —————————————————————————————————————————————————————————————————————

        X_mean_cpu = X_mean.cpu()
        y_mean_cpu = y_mean.cpu()
        self.intercept_ = y_mean_cpu - X_mean_cpu @ self.coef_

        del X_full, y_full
        torch.cuda.empty_cache()

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        # Predict on CPU because weights are huge and on CPU
        X = X.to(dtype=torch.float32, device='cpu')
        return (X @ self.coef_ + self.intercept_).numpy()


class TorchElasticNetCV:
    """
    Production-Grade Solver for Tall Matrices (Samples >> Features).

    Optimizations:
    1. Precomputes Gram Matrices (XtX) in Chunks (Low VRAM).
    2. Uses Float64 (Double) Precision for all Algebra (High Stability).
    3. Fused Kernels + Constant Pre-calculation (High Speed).
    4. Amplitude Rescaling (Fixes Lasso Shrinkage Bias).
    """

    def __init__(self, alphas: List[float], l1_ratio=1.0, max_iter=5000, tol=1e-4,
                 device="cuda", val_fraction=0.1, chunk_size=50000):
        self.alphas = sorted(alphas, reverse=True)
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.val_fraction = val_fraction
        self.chunk_size = chunk_size

        self.best_alpha = None
        self.coef_ = None
        self.intercept_ = None

    def _soft_threshold(self, x, lambda_val):
        return torch.sign(x) * torch.relu(torch.abs(x) - lambda_val)

    def _fista_solve_cov(self, XtX, Xty, n_samples, alpha, L_data, w_init=None):
        """
        Solves ElasticNet using Mixed Precision.

        Optimization:
        1. Inputs (XtX, Xty) are Float64 (High Precision Construction).
        2. We cast them to Float32 for the solver loop (High Speed on A40).
        3. This runs ~30x faster than the Float64 solver.
        """
        # --- CAST TO FLOAT32 FOR SPEED ---
        # The matrix was built in Double to preserve summation accuracy.
        # We solve in Float to use A40 Tensor Cores.
        XtX_f32 = XtX.float()
        Xty_f32 = Xty.float()

        n_features, n_targets = Xty_f32.shape

        # Penalties
        lambda_1 = alpha * self.l1_ratio
        lambda_2 = alpha * (1.0 - self.l1_ratio)

        # Initialize w in Float32
        if w_init is not None:
            w = w_init.clone().float()
        else:
            w = torch.zeros((n_features, n_targets),
                            dtype=torch.float32, device=self.device)

        y_k = w.clone()
        t_k = 1.0

        # --- FIX 1: Normalize Lipschitz by N ---
        L_grad = L_data / n_samples
        L_total = L_grad + lambda_2
        step_size = 1.0 / L_total

        # --- FIX 2: Pre-calculate constants (Float32) ---
        inv_N = 1.0 / n_samples

        # Bias
        bias = Xty_f32 * (step_size * inv_N)

        # Scalars
        beta_val = 1.0 - (step_size * lambda_2)
        alpha_val = -(step_size * inv_N)
        import time
        t = time.time()
        for k in range(self.max_iter):
            w_old = w.clone()

            # --- FUSED KERNEL (FLOAT32) ---
            # This will now run on Tensor Cores (~0.04s instead of 1.25s)
            z = torch.addmm(y_k, XtX_f32, y_k, beta=beta_val, alpha=alpha_val)
            z.add_(bias)

            # Proximal Step
            w = self._soft_threshold(z, lambda_1 * step_size)

            # Check Convergence
            w_norm = torch.norm(w)
            diff = torch.norm(w - w_old) / (w_norm + 1e-10)

            if diff < self.tol:
                break

            # Momentum
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
            momentum = (t_k - 1.0) / t_next
            y_k = w + momentum * (w - w_old)
            t_k = t_next

        print(
            f"   [TorchFast] Alpha {alpha:.2e} converged in {k+1} iterations in {time.time() - t:.2e} seconds.")
        # Return as Float32 (compatible with cache)
        return w

    def fit(self, X, y):
        # 1. Inputs to CPU (Float32 storage)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if X.dtype != torch.float32:
            X = X.float()
        if y.dtype != torch.float32:
            y = y.float()

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        print("[TorchFast] Computing means...")
        X_mean = X.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)

        # 2. Split Indices
        n_val = int(n_samples * self.val_fraction)
        perm = torch.randperm(n_samples, device='cpu')
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        is_val = torch.zeros(n_samples, dtype=torch.bool)
        is_val[val_idx] = True

        # 3. Build Gram Matrices (HYBRID PRECISION)
        print(
            f"[TorchFast] Building Gram matrices (Chunk size {self.chunk_size})...")

        # Accumulators are Float64 (Stable Storage)
        XtX_train = torch.zeros((n_features, n_features),
                                dtype=torch.float64, device=self.device)
        Xty_train = torch.zeros((n_features, n_targets),
                                dtype=torch.float64, device=self.device)

        val_X_gpu_list = []
        val_y_gpu_list = []

        for i in range(0, n_samples, self.chunk_size):
            end = min(i + self.chunk_size, n_samples)

            # Load as Float32
            X_chunk = X[i:end].to(self.device, non_blocking=True)
            y_chunk = y[i:end].to(self.device, non_blocking=True)

            X_chunk.sub_(X_mean.to(self.device))
            y_chunk.sub_(y_mean.to(self.device))

            mask = is_val[i:end].to(self.device)
            X_tr = X_chunk[~mask]
            y_tr = y_chunk[~mask]

            if len(X_tr) > 0:
                # --- HYBRID TRICK ---
                # 1. Matmul happens in Float32 (Fast, utilizes A40 Tensor Cores)
                term_XtX = X_tr.T @ X_tr
                term_Xty = X_tr.T @ y_tr

                # 2. Addition happens in Float64 (Preserves summation precision)
                # PyTorch handles the cast automatically here
                XtX_train.add_(term_XtX)
                Xty_train.add_(term_Xty)

            X_v = X_chunk[mask]
            y_v = y_chunk[mask]
            if len(X_v) > 0:
                val_X_gpu_list.append(X_v)
                val_y_gpu_list.append(y_v)

            del X_chunk, y_chunk, X_tr, y_tr, X_v, y_v
            torch.cuda.empty_cache()

        X_val_gpu = torch.cat(val_X_gpu_list, dim=0)
        y_val_gpu = torch.cat(val_y_gpu_list, dim=0)
        del val_X_gpu_list, val_y_gpu_list

        # Estimate L (Hybrid)
        print("[TorchFast] Estimating L...")
        # L estimation on the summary matrix is fast regardless of precision
        v = torch.randn(n_features, 1, dtype=torch.float64, device=self.device)
        for _ in range(15):
            v = XtX_train @ v
            v = v / (torch.norm(v) + 1e-8)
        L_data = torch.norm(XtX_train @ v).item()

        # 4. Validation Loop (Solver in Float64)
        # Solving on 17k x 17k is okay in Float64 (it's small compared to building)
        print("[TorchFast] Scanning alphas...")
        best_score = -float('inf')
        self.best_alpha = self.alphas[0]
        best_w_cache = None

        w_current = torch.zeros((n_features, n_targets),
                                dtype=torch.float64, device=self.device)
        n_train = len(train_idx)

        for alpha in self.alphas:
            w_current = self._fista_solve_cov(
                XtX_train, Xty_train, n_train, alpha, L_data, w_init=w_current)

            # Predict in Float32 (Faster scoring)
            preds = X_val_gpu @ w_current.float()
            ss_res = ((y_val_gpu - preds) ** 2).sum(0)
            ss_tot = (y_val_gpu ** 2).sum(0)
            r2 = (1 - ss_res / (ss_tot + 1e-6)).mean().item()

            if r2 > best_score:
                best_score = r2
                self.best_alpha = alpha
                best_w_cache = w_current.clone()

        print(
            f"[TorchFast] Best Alpha: {self.best_alpha} (Val R2: {best_score:.4f})")

        # 5. Refit on FULL Data
        print("[TorchFast] Refitting on FULL data...")

        # Hybrid Build for Validation part
        XtX_val = (X_val_gpu.T @ X_val_gpu).double()  # Float32 compute -> cast
        Xty_val = (X_val_gpu.T @ y_val_gpu).double()

        XtX_full = XtX_train + XtX_val
        Xty_full = Xty_train + Xty_val

        del X_val_gpu, y_val_gpu, XtX_train, Xty_train, XtX_val, Xty_val
        torch.cuda.empty_cache()

        # Re-estimate L
        v = torch.randn(n_features, 1, dtype=torch.float64, device=self.device)
        for _ in range(15):
            v = XtX_full @ v
            v = v / (torch.norm(v) + 1e-8)
        L_full = torch.norm(XtX_full @ v).item()

        # Final Solve
        self.coef_ = self._fista_solve_cov(
            XtX_full, Xty_full, n_samples, self.best_alpha, L_full, w_init=best_w_cache)

        # 6. Rescaling (Relaxed Lasso)
        print("[TorchFast] Calculating algebraic rescaling...")
        with torch.no_grad():
            # CAST TO DOUBLE for the calculation to match XtX_full
            coef_64 = self.coef_.double()

            # Numerator: w.T @ Xty
            numerator = (coef_64 * Xty_full).sum(dim=0)

            # Denominator: w.T @ XtX @ w
            # Now both are Double (Float64), so this @ works
            temp = XtX_full @ coef_64
            denominator = (coef_64 * temp).sum(dim=0) + 1e-10

            # Calculate scalars
            scalars = numerator / denominator

            # Apply scalars (Cast back to float to update the model weights)
            self.coef_ *= scalars.float()

            print(
                f"[TorchFast] Rescaling complete. Avg scalar: {scalars.mean().item():.2e}")

        # Finalize
        self.coef_ = self.coef_.float().cpu()
        self.intercept_ = y_mean - X_mean @ self.coef_

        del XtX_full, Xty_full
        torch.cuda.empty_cache()

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(dtype=torch.float32, device='cpu')
        return (X @ self.coef_ + self.intercept_).numpy()


class TorchElasticNetCV_float32:
    """
    High-Performance Solver for Tall Matrices (Samples >> Features).
    Optimized for A40/A100.

    Strategy:
    - Precomputes Covariance Matrices (XtX, Xty) on GPU in chunks.
    - FISTA solver runs on the small (Feature x Feature) covariance matrix.
    - Memory Usage: Constant ~1.5GB VRAM regardless of dataset size (N).
    - Speed: 10x-50x faster than standard iterative solvers.

    Supports: Lasso (l1_ratio=1.0) and ElasticNet (0 < l1_ratio < 1.0).
    """

    def __init__(self, alphas: List[float], l1_ratio=1.0, max_iter=1000, tol=1e-4,
                 device="cuda", val_fraction=0.1, chunk_size=10000):
        # High to Low for Warm Start
        self.alphas = sorted(alphas, reverse=True)
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.val_fraction = val_fraction
        self.chunk_size = chunk_size

        self.best_alpha = None
        self.coef_ = None
        self.intercept_ = None

    def _soft_threshold(self, x, lambda_val):
        return torch.sign(x) * torch.relu(torch.abs(x) - lambda_val)

    def _estimate_lipschitz_from_cov(self, XtX):
        """
        Estimates Lipschitz constant L using Power Iteration on XtX.
        Since we already have XtX, this is instant.
        """
        n_features = XtX.shape[0]
        v = torch.randn(n_features, 1, device=self.device)

        # Power iteration to find max eigenvalue of XtX
        for _ in range(15):
            v = XtX @ v
            v = v / (torch.norm(v) + 1e-8)

        # Rayleigh quotient: (v.T @ XtX @ v) / (v.T @ v)
        # Since v is normalized, just norm(XtX @ v)
        L = torch.norm(XtX @ v).item()
        return L

    def _fista_solve_cov(self, XtX, Xty, n_samples, alpha, L_data, w_init=None):
        """
        Solves ElasticNet using precomputed Covariance Matrices.
        Fused kernel optimization + Constant pre-calculation.
        FIXED: Step size normalized by n_samples.
        """
        n_features, n_targets = Xty.shape

        # Penalties
        lambda_1 = alpha * self.l1_ratio
        lambda_2 = alpha * (1.0 - self.l1_ratio)

        if w_init is not None:
            w = w_init.clone()
        else:
            w = torch.zeros((n_features, n_targets),
                            dtype=XtX.dtype, device=self.device)

        y_k = w.clone()
        t_k = 1.0

        # --- CRITICAL FIX START ---
        # The Lipschitz constant of the Gradient of MSE (1/N * ||Xw-y||^2)
        # is (1/N) * SpectralNorm(XtX).
        # We must divide the raw L_data (SpectralNorm) by n_samples.
        L_grad = L_data / n_samples

        # Total Lipschitz = Smooth part of data + Smooth L2 penalty
        L_total = L_grad + lambda_2

        step_size = 1.0 / L_total

        # --- PRE-CALCULATION ---
        inv_N = 1.0 / n_samples

        # 1. Bias term: (step / N) * Xty
        bias = Xty * (step_size * inv_N)

        # 2. Scalars for addmm
        # z = y_k - step * [ (XtX@y_k)/N + lambda_2*y_k ] + bias
        # z = y_k * (1 - step*lambda_2) - (XtX@y_k)*(step/N) + bias

        beta_val = 1.0 - (step_size * lambda_2)
        alpha_val = -(step_size * inv_N)
        # -----------------------
        import time
        t = time.time()
        for k in range(self.max_iter):
            w_old = w.clone()

            # 1. Fused Gradient Step
            # z = beta * y_k + alpha * (XtX @ y_k) + bias
            z = torch.addmm(y_k, XtX, y_k, beta=beta_val, alpha=alpha_val)
            z.add_(bias)

            # 2. Proximal Step
            w = self._soft_threshold(z, lambda_1 * step_size)

            # 3. Check Convergence
            w_norm = torch.norm(w)
            diff = torch.norm(w - w_old) / (w_norm + 1e-10)

            if diff < self.tol:
                break

            # 4. Momentum
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
            momentum = (t_k - 1.0) / t_next
            y_k = w + momentum * (w - w_old)
            t_k = t_next

        # Print k+1 so "0 iterations" becomes "1 iteration"
        print(
            f"   [TorchFast] Alpha {alpha:.2e} converged in {k+1} iterations in {time.time() - t} seconds.")
        return w

    def _fista_solve_cov_(self, XtX, Xty, n_samples, alpha, L_data, w_init=None):
        """
        Solves ElasticNet using precomputed Covariance Matrices.
        Gradient: (1/N) * (XtX @ w - Xty) + lambda_2 * w
        """
        n_features, n_targets = Xty.shape

        # Penalties
        lambda_1 = alpha * self.l1_ratio
        lambda_2 = alpha * (1.0 - self.l1_ratio)

        if w_init is not None:
            w = w_init.clone()
        else:
            w = torch.zeros((n_features, n_targets),
                            dtype=torch.float32, device=self.device)

        y_k = w.clone()
        t_k = 1.0

        # Adjust step size for L2 penalty
        L_total = L_data + lambda_2
        step_size = 1.0 / L_total

        for k in range(self.max_iter):
            w_old = w.clone()

            # 1. Gradient Step (Using Covariance)
            # residuals = Xw - y  --> Not computed directly
            # gradient = X.T @ residuals / N  --> (XtX @ w - Xty) / N
            grad_mse = (XtX @ y_k - Xty) / n_samples
            grad_l2 = lambda_2 * y_k

            z = y_k - step_size * (grad_mse + grad_l2)

            # 2. Proximal Step (Soft Thresholding for L1)
            w = self._soft_threshold(z, lambda_1 * step_size)

            # 3. Check Convergence
            diff = torch.norm(w - w_old) / (torch.norm(w) + 1e-10)
            if diff < self.tol:
                break

            # 4. Momentum
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
            momentum = (t_k - 1.0) / t_next
            y_k = w + momentum * (w - w_old)
            t_k = t_next

        return w

    def fit(self, X, y):
        # Handle Inputs (Keep on CPU initially)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if X.dtype != torch.float32:
            X = X.float()
        if y.dtype != torch.float32:
            y = y.float()

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        # 1. Calculate Global Means (CPU is fast enough for linear scan)
        print("[TorchFast] Computing means...")
        X_mean = X.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)

        # 2. Prepare Indices
        n_val = int(n_samples * self.val_fraction)
        perm = torch.randperm(n_samples, device='cpu')
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        # Create a boolean mask for fast filtering
        is_val = torch.zeros(n_samples, dtype=torch.bool)
        is_val[val_idx] = True

        # 3. Build Covariance Matrices (Chunked to GPU)
        # We need:
        #   XtX_train, Xty_train (for Solver)
        #   X_val, y_val         (for Validation scoring)

        print(
            f"[TorchFast] Building Gram matrices (Chunk size {self.chunk_size})...")

        XtX_train = torch.zeros((n_features, n_features), device=self.device)
        Xty_train = torch.zeros((n_features, n_targets), device=self.device)

        # We store validation data on GPU for fast scoring.
        # (Assuming val set ~10% fits in memory. If not, score in chunks too.)
        val_X_gpu_list = []
        val_y_gpu_list = []

        # Process in chunks
        for i in range(0, n_samples, self.chunk_size):
            end = min(i + self.chunk_size, n_samples)

            # Load chunk
            X_chunk = X[i:end].to(self.device, non_blocking=True)
            y_chunk = y[i:end].to(self.device, non_blocking=True)

            # Center chunk
            X_chunk.sub_(X_mean.to(self.device))
            y_chunk.sub_(y_mean.to(self.device))

            # Mask
            chunk_mask = is_val[i:end].to(self.device)

            # Train Part
            X_tr = X_chunk[~chunk_mask]
            y_tr = y_chunk[~chunk_mask]

            if len(X_tr) > 0:
                XtX_train.add_(X_tr.T @ X_tr)
                Xty_train.add_(X_tr.T @ y_tr)

            # Val Part (Save for scoring)
            X_v = X_chunk[chunk_mask]
            y_v = y_chunk[chunk_mask]

            if len(X_v) > 0:
                val_X_gpu_list.append(X_v)
                val_y_gpu_list.append(y_v)

            del X_chunk, y_chunk, X_tr, y_tr, X_v, y_v
            torch.cuda.empty_cache()

        # Concatenate Validation Set
        X_val_gpu = torch.cat(val_X_gpu_list, dim=0)
        y_val_gpu = torch.cat(val_y_gpu_list, dim=0)
        del val_X_gpu_list, val_y_gpu_list

        # 4. Validation Loop (Solver runs on 1.2GB matrix -> Super Fast)
        print("[TorchFast] Scanning alphas...")
        L_data = self._estimate_lipschitz_from_cov(XtX_train)

        best_score = -float('inf')
        self.best_alpha = self.alphas[0]
        best_w_cache = None
        w_current = torch.zeros((n_features, n_targets), device=self.device)

        n_train_samples = len(train_idx)

        for alpha in self.alphas:
            # Solve using covariance matrices
            w_current = self._fista_solve_cov(
                XtX_train, Xty_train, n_train_samples, alpha, L_data, w_init=w_current)

            # Score
            preds = X_val_gpu @ w_current
            ss_res = ((y_val_gpu - preds) ** 2).sum(0)
            ss_tot = (y_val_gpu ** 2).sum(0)
            r2 = (1 - ss_res / (ss_tot + 1e-6)).mean().item()

            if r2 > best_score:
                best_score = r2
                self.best_alpha = alpha
                best_w_cache = w_current.clone()

        print(
            f"[TorchFast] Best Alpha: {self.best_alpha} (Val R2: {best_score:.4f})")

        # 5. Refit on FULL Data
        # We can construct XtX_full = XtX_train + XtX_val
        # This avoids reloading the data!
        print("[TorchFast] Refitting on FULL data...")

        XtX_val = X_val_gpu.T @ X_val_gpu
        Xty_val = X_val_gpu.T @ y_val_gpu

        XtX_full = XtX_train + XtX_val
        Xty_full = Xty_train + Xty_val

        # Clear Val buffers
        del X_val_gpu, y_val_gpu, XtX_train, Xty_train, XtX_val, Xty_val
        torch.cuda.empty_cache()

        L_full = self._estimate_lipschitz_from_cov(XtX_full)
        self.coef_ = self._fista_solve_cov(
            XtX_full, Xty_full, n_samples, self.best_alpha, L_full, w_init=best_w_cache)

        # 6. Amplitude Rescaling (Algebraic)
        # We calculate the optimal scalar k without reloading X!
        # k = (w.T @ X.T @ y) / (w.T @ X.T @ X @ w)
        #   = (w.T @ Xty_full) / (w.T @ XtX_full @ w)
        # Computed per target.

        print("[TorchFast] Calculating algebraic rescaling...")
        with torch.no_grad():
            # Numerator: diag(w.T @ Xty_full)
            # Efficient: sum(w * Xty_full, dim=0)
            numerator = (self.coef_ * Xty_full).sum(dim=0)

            # Denominator: diag(w.T @ XtX_full @ w)
            # Temp = XtX_full @ w
            # Denom = sum(w * Temp, dim=0)
            temp = XtX_full @ self.coef_
            denominator = (self.coef_ * temp).sum(dim=0) + 1e-10

            scalars = numerator / denominator
            self.coef_ *= scalars
            print(
                f"[TorchFast] Rescaling complete. Avg scalar: {scalars.mean().item():.2e}")

        # 7. Finalize
        self.coef_ = self.coef_.cpu()

        # Intercept = y_mean - X_mean @ coef
        # (calculated on CPU)
        self.intercept_ = y_mean - X_mean @ self.coef_

        # Cleanup
        del XtX_full, Xty_full
        torch.cuda.empty_cache()

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(dtype=torch.float32, device='cpu')
        return (X @ self.coef_ + self.intercept_).numpy()


class TorchConstrainedRidgeCV:
    """
    Exact ridge with hard constraint:
      For each block b and target i:
        W[b*block_size + i, i] = 0

    Assumes:
      - y has shape (N, block_size)  (e.g., 1024 targets)
      - X has shape (N, D) where D is multiple of block_size: D = n_blocks * block_size

    Does:
      - Single validation split (like your original code)
      - Chooses alpha by constrained validation R2 (mean over targets)
      - Refits on full data with the best alpha
      - Returns coef_ on CPU, intercept_ on CPU
    """

    def __init__(
        self,
        alphas: List[float],
        device: str = "cuda",
        val_fraction: float = 0.1,
        block_size: int = 1024,
        smw_target_chunk: int = 16,   # targets per chunk for SMW
        jitter: float = 1e-6,
        seed: Optional[int] = None,
    ):
        self.alphas = list(alphas)
        self.device = device
        self.val_fraction = float(val_fraction)
        self.block_size = int(block_size)
        self.smw_target_chunk = int(smw_target_chunk)
        self.jitter = float(jitter)
        self.seed = seed

        self.best_alpha: Optional[float] = None
        self.coef_: Optional[torch.Tensor] = None      # (D,T) on CPU
        self.intercept_: Optional[torch.Tensor] = None  # (1,T) on CPU

    # ---------- indexing for constraints ----------

    def _forbidden_index_matrix(self, n_features: int, n_targets: int, device) -> torch.Tensor:
        """
        forb_idx[b, i] = b*block_size + i
        shape: (n_blocks, block_size)
        """
        bs = self.block_size
        if n_targets != bs:
            raise ValueError(
                f"Expected y.shape[1] == block_size == {bs}, got {n_targets}.")
        if n_features % bs != 0:
            raise ValueError(
                f"Expected n_features multiple of {bs}, got {n_features}.")
        n_blocks = n_features // bs
        b = torch.arange(n_blocks, device=device).unsqueeze(1)  # (B,1)
        i = torch.arange(bs, device=device).unsqueeze(0)        # (1,bs)
        return b * bs + i                                       # (B,bs)

    # ---------- dual SMW chunk (fat) ----------

    def _dual_constrained_chunk(
        self,
        # Cholesky factor of A = (K + alpha I)  (Ntr,Ntr)
        L: torch.Tensor,
        X_tr: torch.Tensor,              # (Ntr,D)
        Y_tr: torch.Tensor,              # (Ntr,T)
        X_va: Optional[torch.Tensor],    # (Nva,D) or None
        K_cross: Optional[torch.Tensor],  # (Nva,Ntr) or None
        forb_idx: torch.Tensor,          # (B,T)
        tgt_idx: torch.Tensor,           # (c,)
    ):
        """
        Exact solve: (A - U U^T) a = y  using SMW (rank B downdate), batched over c targets.

        Returns:
          preds (Nva,c) or None
          a     (Ntr,c)
        """
        device = X_tr.device
        Ntr = X_tr.shape[0]
        B = forb_idx.shape[0]
        c = tgt_idx.numel()

        # p = A^{-1} y
        P = torch.cholesky_solve(Y_tr[:, tgt_idx], L)  # (Ntr,c)

        # U for these targets: columns are forbidden features for each target
        cols = forb_idx[:, tgt_idx].contiguous().reshape(-1)  # (B*c,)
        U_tr_flat = X_tr[:, cols]                              # (Ntr,B*c)

        # Q = A^{-1} U
        Q_flat = torch.cholesky_solve(U_tr_flat, L)            # (Ntr,B*c)

        # reshape to (Ntr,B,c)
        U_tr = U_tr_flat.view(Ntr, B, c)
        Q = Q_flat.view(Ntr, B, c)

        # batch shapes (c,B,Ntr), (c,Ntr,B), (c,Ntr,1)
        U_t = U_tr.permute(2, 1, 0).contiguous()   # (c,B,Ntr)
        Q_t = Q.permute(2, 0, 1).contiguous()      # (c,Ntr,B)
        P_t = P.T.contiguous().unsqueeze(-1)       # (c,Ntr,1)

        # S = U^T A^{-1} U  (c,B,B)
        S = torch.bmm(U_t, Q_t)                    # (c,B,B)
        # rhs = U^T A^{-1} y (c,B,1)
        rhs = torch.bmm(U_t, P_t)                  # (c,B,1)

        I = torch.eye(B, device=device, dtype=S.dtype).unsqueeze(0)   # (1,B,B)
        Inner = (I - S) + self.jitter * \
            I                               # (c,B,B)

        # (c,B,1)
        v = torch.linalg.solve(Inner, rhs)

        # a = p + A^{-1} U v
        # (c,Ntr,1)
        a_t = P_t + torch.bmm(Q_t, v)
        # (Ntr,c)
        a = a_t.squeeze(-1).T.contiguous()

        if X_va is None:
            return None, a

        # preds = (K_cross - U_va U_tr^T) a = K_cross a - U_va (U_tr^T a)
        # (Nva,c)
        term1 = K_cross @ a

        Nva = X_va.shape[0]
        # (Nva,B*c)
        U_va_flat = X_va[:, cols]
        U_va = U_va_flat.view(Nva, B, c).permute(
            2, 0, 1).contiguous()   # (c,Nva,B)

        # (c,Ntr,1)
        a_t2 = a.T.unsqueeze(-1)
        # (c,B,1) = U_tr^T a
        s = torch.bmm(U_t, a_t2)

        term2 = torch.bmm(
            U_va, s).squeeze(-1).T.contiguous()            # (Nva,c)
        preds = term1 - term2
        return preds, a

    # ---------- primal constrained chunk (tall) ----------

    def _primal_constrained_chunk(
        self,
        L: torch.Tensor,            # Cholesky of A = (XtX + alpha I) (D,D)
        XtY: torch.Tensor,          # (D,T) training crossprod
        forb_idx: torch.Tensor,     # (B,T)
        tgt_idx: torch.Tensor,      # (c,)
        n_features: int,
    ) -> torch.Tensor:
        """
        Exact constrained primal solution with equality constraints w[S]=0:
          w_c = w - Z * solve(Z_S, w_S)
        where:
          w   = A^{-1} XtY[:,t]
          Z   = A^{-1}[:,S]
          Z_S = A^{-1}[S,S]
        Batched over c targets (still loops over c internally, but c is small).
        """
        device = XtY.device
        B = forb_idx.shape[0]
        c = tgt_idx.numel()

        # Unconstrained weights for these targets: W_unc = A^{-1} XtY
        W_unc = torch.cholesky_solve(XtY[:, tgt_idx], L)  # (D,c)

        # Build RHS E selecting basis vectors at forbidden indices for this chunk
        cols = forb_idx[:, tgt_idx].contiguous().reshape(-1)  # (B*c,)
        m = cols.numel()

        E = torch.zeros((n_features, m), device=device, dtype=W_unc.dtype)
        E[cols, torch.arange(m, device=device)] = 1.0

        Z_flat = torch.cholesky_solve(E, L)  # (D, B*c)
        # (D,B,c) with columns ordered by (block, target_in_chunk)
        Z = Z_flat.view(n_features, B, c)

        Wc = W_unc.clone()
        # small per-target solves on (B x B)
        for j in range(c):
            idxS = forb_idx[:, tgt_idx[j]]          # (B,)
            Zj = Z[:, :, j]                         # (D,B)
            M_SS = Zj[idxS, :]                      # (B,B) = A^{-1}[S,S]
            wS = W_unc[idxS, j]                     # (B,)

            lam = torch.linalg.solve(
                M_SS + self.jitter * torch.eye(B, device=device, dtype=W_unc.dtype), wS)
            Wc[:, j] = W_unc[:, j] - (Zj @ lam)

            # exact enforce
            Wc[idxS, j] = 0.0

        return Wc  # (D,c)

    # ---------- scoring ----------

    @staticmethod
    def _mean_r2_centered(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        y_true and y_pred are centered already.
        Returns mean R2 over targets.
        """
        ss_res = ((y_true - y_pred) ** 2).sum(0)
        ss_tot = (y_true ** 2).sum(0)
        r2 = (1.0 - ss_res / (ss_tot + 1e-6)).mean()
        return float(r2.item())

    # ---------- fit/predict ----------

    def fit(self, X, y):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        X_full = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_full = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        if y_full.ndim == 1:
            y_full = y_full.unsqueeze(1)

        N, D = X_full.shape
        T = y_full.shape[1]
        bs = self.block_size

        if T != bs:
            raise ValueError(f"Expected y to have {bs} targets, got {T}.")
        if D % bs != 0:
            raise ValueError(f"Expected D multiple of {bs}, got D={D}.")

        # center
        X_mean = X_full.mean(dim=0, keepdim=True)
        y_mean = y_full.mean(dim=0, keepdim=True)
        X_full = X_full - X_mean
        y_full = y_full - y_mean

        forb_idx = self._forbidden_index_matrix(
            D, T, device=X_full.device)  # (B,T)

        # validation split
        n_val = max(1, int(N * self.val_fraction))
        perm = torch.randperm(N, device=X_full.device)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_tr = X_full[tr_idx]
        y_tr = y_full[tr_idx]
        X_va = X_full[val_idx]
        y_va = y_full[val_idx]

        # decide primal vs dual (like your original)
        is_primal = (N >= D)

        tgt_all = torch.arange(T, device=X_full.device)
        csz = max(1, min(self.smw_target_chunk, T))

        best_r2 = -float("inf")
        best_alpha = self.alphas[-1]

        # ----------------- CV: constrained -----------------
        if not is_primal:
            # DUAL CV
            K_tr = X_tr @ X_tr.T
            K_cross = X_va @ X_tr.T
            I_tr = torch.eye(
                K_tr.shape[0], device=X_full.device, dtype=torch.float32)

            for alpha in self.alphas:
                A = K_tr + float(alpha) * I_tr
                L = torch.linalg.cholesky(A)

                preds_all = []
                for j in range(0, T, csz):
                    tgt_idx = tgt_all[j:j+csz]
                    preds, _ = self._dual_constrained_chunk(
                        L=L, X_tr=X_tr, Y_tr=y_tr,
                        X_va=X_va, K_cross=K_cross,
                        forb_idx=forb_idx, tgt_idx=tgt_idx
                    )
                    preds_all.append(preds)

                y_pred = torch.cat(preds_all, dim=1)  # (Nva,T)
                r2 = self._mean_r2_centered(y_va, y_pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_alpha = float(alpha)

        else:
            # PRIMAL CV
            # build XtX_train, XtY_train efficiently
            XtX_full = X_full.T @ X_full       # (D,D)
            XtY_full = X_full.T @ y_full       # (D,T)

            XtX_val = X_va.T @ X_va
            XtY_val = X_va.T @ y_va

            XtX_tr = XtX_full - XtX_val
            XtY_tr = XtY_full - XtY_val

            I_D = torch.eye(D, device=X_full.device, dtype=torch.float32)

            for alpha in self.alphas:
                A = XtX_tr + float(alpha) * I_D
                L = torch.linalg.cholesky(A)

                preds_chunks = []
                for j in range(0, T, csz):
                    tgt_idx = tgt_all[j:j+csz]
                    Wc = self._primal_constrained_chunk(
                        L=L, XtY=XtY_tr, forb_idx=forb_idx,
                        tgt_idx=tgt_idx, n_features=D
                    )  # (D,c)
                    preds_chunks.append(X_va @ Wc)  # (Nva,c)

                y_pred = torch.cat(preds_chunks, dim=1)
                r2 = self._mean_r2_centered(y_va, y_pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_alpha = float(alpha)

            del XtX_full, XtY_full, XtX_val, XtY_val, XtX_tr, XtY_tr

        self.best_alpha = best_alpha

        # ----------------- refit on full data (constrained) -----------------
        self.coef_ = torch.zeros((D, T), dtype=torch.float32, device="cpu")

        if not is_primal:
            # DUAL refit
            K_full = X_full @ X_full.T
            I_N = torch.eye(N, device=X_full.device, dtype=torch.float32)
            A = K_full + self.best_alpha * I_N
            L = torch.linalg.cholesky(A)

            for j in range(0, T, csz):
                tgt_idx = tgt_all[j:j+csz]
                _, a = self._dual_constrained_chunk(
                    L=L, X_tr=X_full, Y_tr=y_full,
                    X_va=None, K_cross=None,
                    forb_idx=forb_idx, tgt_idx=tgt_idx
                )  # (N,c)
                Wc = X_full.T @ a  # (D,c)

                # exact enforce (safety)
                rows = forb_idx[:, tgt_idx]  # (B,c)
                cols = torch.arange(tgt_idx.numel(), device=X_full.device).unsqueeze(
                    0).expand_as(rows)
                Wc[rows, cols] = 0.0

                self.coef_[:, tgt_idx] = Wc.cpu()

            del K_full, A, L

        else:
            # PRIMAL refit
            XtX_full = X_full.T @ X_full
            XtY_full = X_full.T @ y_full
            I_D = torch.eye(D, device=X_full.device, dtype=torch.float32)

            A = XtX_full + self.best_alpha * I_D
            L = torch.linalg.cholesky(A)

            for j in range(0, T, csz):
                tgt_idx = tgt_all[j:j+csz]
                Wc = self._primal_constrained_chunk(
                    L=L, XtY=XtY_full, forb_idx=forb_idx,
                    tgt_idx=tgt_idx, n_features=D
                )  # (D,c)
                self.coef_[:, tgt_idx] = Wc.cpu()

            del XtX_full, XtY_full, A, L

        # intercept (on CPU, consistent with constrained weights)
        X_mean_cpu = X_mean.cpu()
        y_mean_cpu = y_mean.cpu()
        self.intercept_ = y_mean_cpu - X_mean_cpu @ self.coef_

        # cleanup
        del X_full, y_full, X_tr, y_tr, X_va, y_va
        if self.device != "cpu":
            torch.cuda.empty_cache()

        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(dtype=torch.float32, device="cpu")
        return (X @ self.coef_ + self.intercept_).numpy()


class TorchBlockRidgeCV:
    """
    Exact ridge under block exclusion/selection constraints by explicitly
    selecting the allowed feature columns for each target block.

    Modes:
      - exclusion: predict targets in region R using all features EXCEPT region R features
      - selection: predict targets in region R using ONLY region R features

    Notes:
      - Chooses alpha per-region by constrained validation R2 (mean over targets in region).
      - Refits on full data per-region with the chosen alpha.
      - coef_ stored on CPU.
    """

    def __init__(
        self,
        alphas: List[float],
        device: str = "cuda",
        val_fraction: float = 0.1,
        feature_period: int = 1024,
        regions: Optional[Dict[str, Tuple[int, int]]] = None,
        jitter: float = 1e-6,
        seed: Optional[int] = None,
    ):
        self.alphas = list(alphas)
        self.device = device
        self.val_fraction = float(val_fraction)
        self.feature_period = int(feature_period)
        self.jitter = float(jitter)
        self.seed = seed

        self.regions = regions or {
            "V1": (0, 512),
            "V4": (512, 768),
            "IT": (768, 1024),
        }

        self.coef_: Optional[torch.Tensor] = None
        self.intercept_: Optional[torch.Tensor] = None
        self.best_alpha_by_region: Dict[str, float] = {}

    @staticmethod
    def _mean_r2_centered(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        # per-target R2, then mean
        ss_res = ((y_true - y_pred) ** 2).sum(0)
        # centered => SST is sum of squares around 0
        ss_tot = (y_true ** 2).sum(0)
        r2 = (1.0 - ss_res / (ss_tot + 1e-6)).mean()
        return float(r2.item())

    @staticmethod
    def _chol_solve(A: torch.Tensor, B: torch.Tensor, jitter: float) -> torch.Tensor:
        # Solve A X = B using Cholesky, with tiny jitter for safety
        A = A + jitter * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        L = torch.linalg.cholesky(A)
        return torch.cholesky_solve(B, L)

    def fit(self, X, y, block_mode: str = "exclusion"):
        if block_mode not in ("exclusion", "selection"):
            raise ValueError("block_mode must be 'exclusion' or 'selection'.")

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        X_full = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_full = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        if y_full.ndim == 1:
            y_full = y_full.unsqueeze(1)

        n_samples, n_features = X_full.shape
        _, n_targets = y_full.shape

        # Center once globally (keeps consistency across regions)
        X_mean = X_full.mean(dim=0, keepdim=True)
        y_mean = y_full.mean(dim=0, keepdim=True)
        X_full = X_full - X_mean
        y_full = y_full - y_mean

        # Output weights on CPU
        self.coef_ = torch.zeros(
            (n_features, n_targets), dtype=torch.float32, device="cpu")
        self.best_alpha_by_region = {}

        # Train/val split once, shared across blocks (good practice)
        n_val = max(1, int(n_samples * self.val_fraction))
        perm = torch.randperm(n_samples, device=self.device)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        # Precompute modulo indices (vectorized)
        feat_idx = torch.arange(n_features, device=self.device)
        feat_mod = feat_idx % self.feature_period

        targ_idx = torch.arange(n_targets, device=self.device)
        targ_mod = targ_idx % self.feature_period

        print(
            f"   [BlockRidgeCV] Solving by blocks ({block_mode}) on {self.device}...")

        for name, (start, end) in self.regions.items():
            # Targets in this region
            tmask = (targ_mod >= start) & (targ_mod < end)
            if not bool(tmask.any()):
                continue

            target_indices = targ_idx[tmask]              # (T_block,)
            y_subset = y_full[:, target_indices]          # (N, T_block)

            # Allowed features mask
            if block_mode == "exclusion":
                fmask = (feat_mod < start) | (feat_mod >= end)
            else:
                fmask = (feat_mod >= start) & (feat_mod < end)

            if not bool(fmask.any()):
                raise ValueError(
                    f"No features selected for region {name} under mode={block_mode}.")

            # Slice X once per region (only ~3 times)
            X_subset = X_full[:, fmask]                   # (N, D_sub)
            D_sub = X_subset.shape[1]

            # Split
            X_tr = X_subset[tr_idx]
            y_tr = y_subset[tr_idx]
            X_va = X_subset[val_idx]
            y_va = y_subset[val_idx]

            # Choose solver by shape
            is_primal = (X_tr.shape[0] >= D_sub)

            best_r2 = -float("inf")
            best_alpha = self.alphas[-1]

            print(
                f"   ... Region {name}: targets={target_indices.numel()}, features={D_sub}, solver={'primal' if is_primal else 'dual'}")

            if is_primal:
                # Precompute train cross-products once
                XtX = X_tr.T @ X_tr                        # (D_sub,D_sub)
                XtY = X_tr.T @ y_tr                        # (D_sub,T_block)

                for alpha in self.alphas:
                    W = self._chol_solve(
                        XtX + float(alpha) * torch.eye(D_sub, device=self.device), XtY, self.jitter)
                    pred = X_va @ W
                    r2 = self._mean_r2_centered(y_va, pred)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_alpha = float(alpha)

                # Refit full
                XtX_full = X_subset.T @ X_subset
                XtY_full = X_subset.T @ y_subset
                W_final = self._chol_solve(
                    XtX_full + best_alpha * torch.eye(D_sub, device=self.device), XtY_full, self.jitter)

            else:
                # Dual: kernel on samples
                K_tr = X_tr @ X_tr.T                       # (Ntr,Ntr)
                K_va = X_va @ X_tr.T                       # (Nva,Ntr)
                Ntr = K_tr.shape[0]

                for alpha in self.alphas:
                    C = self._chol_solve(
                        K_tr + float(alpha) * torch.eye(Ntr, device=self.device), y_tr, self.jitter)
                    pred = K_va @ C
                    r2 = self._mean_r2_centered(y_va, pred)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_alpha = float(alpha)

                # Refit full
                K_full = X_subset @ X_subset.T
                C_full = self._chol_solve(
                    K_full + best_alpha * torch.eye(n_samples, device=self.device), y_subset, self.jitter)
                W_final = X_subset.T @ C_full               # (D_sub,T_block)

            self.best_alpha_by_region[name] = best_alpha

            # Place weights into global matrix (CPU) using 2D advanced indexing
            feat_cpu = torch.nonzero(fmask).squeeze(
                1).cpu()             # (D_sub,)
            targ_cpu = target_indices.cpu()                               # (T_block,)
            W_cpu = W_final.cpu()                                         # (D_sub,T_block)
            self.coef_[feat_cpu[:, None], targ_cpu[None, :]] = W_cpu

            # cleanup
            del X_subset, y_subset, X_tr, y_tr, X_va, y_va, W_final
            if self.device != "cpu":
                torch.cuda.empty_cache()

        # Intercept consistent with constrained weights
        X_mean_cpu = X_mean.cpu()
        y_mean_cpu = y_mean.cpu()
        self.intercept_ = y_mean_cpu - X_mean_cpu @ self.coef_

        print("   [BlockRidgeCV] Done.")
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(dtype=torch.float32, device="cpu")
        return (X @ self.coef_ + self.intercept_).numpy()

# --- Additions for Online Metrics ---


class LinearInternalModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:  # (batch, seq_len, features)
            # print(
            #    f"Warning: LinearInternalModel received sequential input of shape {x.shape}. Flattening over sequence dimension.")
            N, T, D = x.shape  # B, T, D
            # let's average. If concatenation is preferred, this can be changed.
            # x = x.mean(dim=1) # Average over sequence length --> (N, D)
            # Or, flatten:
            x = x.reshape(N, -1)
        elif x.ndim > 3:  # e.g. (batch, channels, height, width) for images
            # print(
            #    f"Warning: LinearInternalModel received input of shape {x.shape}. Flattening non-batch dimensions.")
            x = x.reshape(x.shape[0], -1)

        return self.linear(x)


class AttentionPoolingInternalModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, reduced_dim: int = 16384, seed: int = 42):
        super().__init__()
        # Random projection - fixed, not learned, seeded for reproducibility
        self.random_proj = nn.Linear(input_dim, reduced_dim, bias=False)
        torch.manual_seed(seed)
        nn.init.normal_(self.random_proj.weight, mean=0,
                        std=1/np.sqrt(reduced_dim))
        self.random_proj.weight.requires_grad = False

        self.layer_norm = nn.LayerNorm(reduced_dim)
        self.attn = nn.Linear(reduced_dim, 1)
        self.linear = nn.Linear(reduced_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:  # (batch, seq, features) = (64, 30, 263168)
            x = self.random_proj(x)  # Output: (64, 30, 16384) ✓
            x = self.layer_norm(x)    # Works on last dim
            attn_weights = torch.softmax(self.attn(x), dim=1)  # (64, 30, 1)
            x = (x * attn_weights).sum(dim=1)  # (64, 16384)
        elif x.ndim == 2:  # (batch, features)
            x = self.random_proj(x)  # Output: (64, 16384) ✓
            x = self.layer_norm(x)
        return self.linear(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        # Create pe buffer once and reuse
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .transpose(0, 1) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_length, d_model]
        # self.pe shape: [1, max_len, d_model]
        # We need to select up to seq_length from pe
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerInternalModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 embed_dim: int = 256, num_heads: int = 12,
                 num_encoder_layers: int = 1):  # num_encoder_layers for AttentiveClassifier depth
        super().__init__()
        self.output_dim = output_dim  # num_classes for classifier
        self.input_dim = input_dim   # D_fe from feature extractor
        self.embed_dim = embed_dim   # Internal embedding dimension

        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.ReLU(),  # Changed from F.relu to nn.ReLU()
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.positional_encoder = PositionalEncoding(self.embed_dim)

        # Using num_encoder_layers for the depth of the attentive pooler
        self.attentive_pooler_head = AttentiveClassifier(
            embed_dim=self.embed_dim,
            num_classes=self.output_dim,
            depth=num_encoder_layers,
            num_heads=num_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim] (input_dim is D_fe)
        # If x is [batch_size, input_dim] for static features, unsqueeze to add seq_len=1
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        N, T, D_input = x.shape

        # Project each time step's features
        projected_x = self.input_projection(
            x.reshape(N * T, D_input))  # Apply to (N*T, D_input)
        # Reshape to (N, T, embed_dim)
        projected_x = projected_x.view(N, T, self.embed_dim)

        positioned_x = self.positional_encoder(projected_x)

        # Attentive pooler head should take (N, T, embed_dim) and output (N, output_dim)
        pooled_output = self.attentive_pooler_head(positioned_x)

        return pooled_output
