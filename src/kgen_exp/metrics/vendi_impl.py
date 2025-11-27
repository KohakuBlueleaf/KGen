import math
import numpy as np
import scipy
import scipy.linalg
from sklearn import preprocessing
import torch


# Original NumPy implementation
def weight_K(K, p=None):
    if p is None:
        return K / K.shape[0]
    else:
        return K * np.outer(np.sqrt(p), np.sqrt(p))


def normalize_K(K):
    d = np.sqrt(np.diagonal(K))
    return K / np.outer(d, d)


def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_**q).sum()) / (1 - q)


def score_K(K, q=1, p=None, normalize=False):
    if normalize:
        K = normalize_K(K)
    K_ = weight_K(K, p)
    w = scipy.linalg.eigvalsh(K_)
    return np.exp(entropy_q(w, q=q))


def score_X(X, q=1, p=None, normalize=True):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    K = X @ X.T
    return score_K(K, q=1, p=p)


def score_dual(X, q=1, normalize=True):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    n = X.shape[0]
    S = X.T @ X
    w = scipy.linalg.eigvalsh(S / n)
    m = w > 0
    return np.exp(entropy_q(w, q=q))


def score(samples, k, q=1, p=None, normalize=False):
    n = len(samples)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = k(samples[i], samples[j])
    return score_K(K, p=p, q=q, normalize=normalize)


def intdiv_K(K, q=1, p=None):
    K_ = K**q
    if p is None:
        p = np.ones(K.shape[0]) / K.shape[0]
    return 1 - np.sum(K_ * np.outer(p, p))


def intdiv_X(X, q=1, p=None, normalize=True):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    K = X @ X.T
    return intdiv_K(K, q=q, p=p)


def intdiv(samples, k, q=1, p=None):
    n = len(samples)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = k(samples[i], samples[j])
    return intdiv_K(K, q=q, p=p)


# ==================== PyTorch Implementation ====================


def weight_K_torch(K, p=None, device=None):
    """
    Weight the kernel matrix K by probabilities p.

    Args:
        K (torch.Tensor): Kernel matrix of shape (n, n)
        p (torch.Tensor, optional): Probability vector of shape (n,)
        device (torch.device, optional): Device to use

    Returns:
        torch.Tensor: Weighted kernel matrix
    """
    if device is None:
        device = K.device

    if p is None:
        return K / K.shape[0]
    else:
        sqrt_p = torch.sqrt(p).to(device)
        return K * torch.outer(sqrt_p, sqrt_p)


def normalize_K_torch(K):
    """
    Normalize the kernel matrix K.

    Args:
        K (torch.Tensor): Kernel matrix of shape (n, n)

    Returns:
        torch.Tensor: Normalized kernel matrix
    """
    d = torch.sqrt(torch.diag(K))
    return K / torch.outer(d, d)


def entropy_q_torch(p, q=1):
    """
    Compute the q-entropy of the probability vector p.

    Args:
        p (torch.Tensor): Probability vector
        q (int or str, optional): Order of entropy

    Returns:
        torch.Tensor: q-entropy of p
    """
    # Keep only positive eigenvalues
    p_ = p[p > 0]

    if q == 1:
        return -(p_ * torch.log(p_)).sum()
    if q == "inf":
        return -torch.log(torch.max(p))
    return torch.log((p_**q).sum()) / (1 - q)


def score_K_torch(K, q=1, p=None, normalize=False, device=None):
    """
    Compute diversity score based on the kernel matrix.

    Args:
        K (torch.Tensor): Kernel matrix of shape (n, n)
        q (int or str, optional): Order of entropy
        p (torch.Tensor, optional): Probability vector of shape (n,)
        normalize (bool, optional): Whether to normalize the kernel matrix
        device (torch.device, optional): Device to use

    Returns:
        torch.Tensor: Diversity score
    """
    if device is None:
        device = K.device

    if normalize:
        K = normalize_K_torch(K)

    K_ = weight_K_torch(K, p, device)

    # Compute eigenvalues using torch.linalg
    w = torch.linalg.eigvalsh(K_)

    return torch.exp(entropy_q_torch(w, q=q))


def score_X_torch(X, q=1, p=None, normalize=True, device=None):
    """
    Compute diversity score based on the data matrix X.

    Args:
        X (torch.Tensor): Data matrix of shape (n, d)
        q (int or str, optional): Order of entropy
        p (torch.Tensor, optional): Probability vector of shape (n,)
        normalize (bool, optional): Whether to normalize the rows of X
        device (torch.device, optional): Device to use

    Returns:
        torch.Tensor: Diversity score
    """
    if device is None:
        device = X.device

    if normalize:
        X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    else:
        X_norm = X

    K = X_norm @ X_norm.T

    return score_K_torch(K, q=1, p=p, device=device)


def score_dual_torch(X, q=1, normalize=True, device=None):
    """
    Compute diversity score using the dual formulation.

    Args:
        X (torch.Tensor): Data matrix of shape (n, d)
        q (int or str, optional): Order of entropy
        normalize (bool, optional): Whether to normalize the rows of X
        device (torch.device, optional): Device to use

    Returns:
        torch.Tensor: Diversity score
    """
    if device is None:
        device = X.device

    if normalize:
        X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    else:
        X_norm = X

    n = X_norm.shape[0]
    S = X_norm.T @ X_norm

    w = torch.linalg.eigvalsh(S / n)

    return torch.exp(entropy_q_torch(w, q=q))


def score_torch(samples, k, q=1, p=None, normalize=False, device=None):
    """
    Compute diversity score using a custom kernel function.

    Args:
        samples (list or torch.Tensor): Sample data
        k (callable): Kernel function
        q (int or str, optional): Order of entropy
        p (torch.Tensor, optional): Probability vector
        normalize (bool, optional): Whether to normalize the kernel matrix
        device (torch.device, optional): Device to use

    Returns:
        torch.Tensor: Diversity score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(samples)
    K = torch.zeros((n, n), device=device)

    # Can be optimized with batch processing for GPU
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = k(samples[i], samples[j])

    return score_K_torch(K, p=p, q=q, normalize=normalize, device=device)


def intdiv_K_torch(K, q=1, p=None, device=None):
    """
    Compute intrinsic diversity based on the kernel matrix.

    Args:
        K (torch.Tensor): Kernel matrix of shape (n, n)
        q (int, optional): Power parameter
        p (torch.Tensor, optional): Probability vector of shape (n,)
        device (torch.device, optional): Device to use

    Returns:
        torch.Tensor: Intrinsic diversity
    """
    if device is None:
        device = K.device

    K_ = K**q

    if p is None:
        p = torch.ones(K.shape[0], device=device) / K.shape[0]

    return 1 - torch.sum(K_ * torch.outer(p, p))


def intdiv_X_torch(X, q=1, p=None, normalize=True, device=None):
    """
    Compute intrinsic diversity based on the data matrix X.

    Args:
        X (torch.Tensor): Data matrix of shape (n, d)
        q (int, optional): Power parameter
        p (torch.Tensor, optional): Probability vector of shape (n,)
        normalize (bool, optional): Whether to normalize the rows of X
        device (torch.device, optional): Device to use

    Returns:
        torch.Tensor: Intrinsic diversity
    """
    if device is None:
        device = X.device

    if normalize:
        X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    else:
        X_norm = X

    K = X_norm @ X_norm.T

    return intdiv_K_torch(K, q=q, p=p, device=device)


def intdiv_torch(samples, k, q=1, p=None, device=None):
    """
    Compute intrinsic diversity using a custom kernel function.

    Args:
        samples (list or torch.Tensor): Sample data
        k (callable): Kernel function
        q (int, optional): Power parameter
        p (torch.Tensor, optional): Probability vector
        device (torch.device, optional): Device to use

    Returns:
        torch.Tensor: Intrinsic diversity
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(samples)
    K = torch.zeros((n, n), device=device)

    # Can be optimized with batch processing for GPU
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = k(samples[i], samples[j])

    return intdiv_K_torch(K, q=q, p=p, device=device)


if __name__ == "__main__":
    with torch.no_grad(), torch.inference_mode():
        # Validation code to compare NumPy and PyTorch implementations
        import time

        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Choose device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Generate random data
        n_samples = 4096
        n_features = 256

        # NumPy data
        X_np = np.random.rand(n_samples, n_features)

        # PyTorch data
        X_torch = torch.tensor(X_np, dtype=torch.float32, device=device)

        # Test score_X
        print("\nTesting score_X:")

        start_time = time.time()
        score_np = score_X(X_np, q=1, normalize=True)
        np_time = time.time() - start_time
        print(f"NumPy result: {score_np:.6f}, Time: {np_time:.4f} seconds")

        start_time = time.time()
        score_torch_result = score_X_torch(X_torch, q=1, normalize=True, device=device)
        torch_time = time.time() - start_time
        print(
            f"PyTorch result: {score_torch_result.cpu().numpy():.6f}, Time: {torch_time:.4f} seconds"
        )
        print(f"Speedup: {np_time / torch_time:.2f}x")
        print(
            f"Absolute difference: {abs(score_np - score_torch_result.cpu().numpy()):.8f}"
        )

        # Test score_dual
        print("\nTesting score_dual:")

        start_time = time.time()
        score_dual_np = score_dual(X_np, q=1, normalize=True)
        np_time = time.time() - start_time
        print(f"NumPy result: {score_dual_np:.6f}, Time: {np_time:.4f} seconds")

        start_time = time.time()
        score_dual_torch_result = score_dual_torch(
            X_torch, q=1, normalize=True, device=device
        )
        torch_time = time.time() - start_time
        print(
            f"PyTorch result: {score_dual_torch_result.cpu().numpy():.6f}, Time: {torch_time:.4f} seconds"
        )
        print(f"Speedup: {np_time / torch_time:.2f}x")
        print(
            f"Absolute difference: {abs(score_dual_np - score_dual_torch_result.cpu().numpy()):.8f}"
        )

        # Test with different q values
        print("\nTesting with different q values:")
        q_values = [0.5, 1, 2, "inf"]

        for q in q_values:
            print(f"\nq = {q}:")

            # NumPy
            start_time = time.time()
            score_np = score_X(X_np, q=q, normalize=True)
            np_time = time.time() - start_time
            print(f"NumPy result: {score_np:.6f}, Time: {np_time:.4f} seconds")

            # PyTorch
            start_time = time.time()
            score_torch_result = score_X_torch(
                X_torch, q=q, normalize=True, device=device
            )
            torch_time = time.time() - start_time
            print(
                f"PyTorch result: {score_torch_result.cpu().numpy():.6f}, Time: {torch_time:.4f} seconds"
            )
            print(f"Speedup: {np_time / torch_time:.2f}x")
            print(
                f"Absolute difference: {abs(score_np - score_torch_result.cpu().numpy()):.8f}"
            )

        # Test with custom kernel
        print("\nTesting with custom kernel:")

        # Create smaller samples for custom kernel test (slower operation)
        n_small = 100
        samples_np = [np.random.rand(10) for _ in range(n_small)]
        samples_torch = [
            torch.tensor(s, dtype=torch.float32, device=device) for s in samples_np
        ]

        # Define kernel functions
        def rbf_kernel_np(x, y, gamma=1.0):
            return np.exp(-gamma * np.sum((x - y) ** 2))

        def rbf_kernel_torch(x, y, gamma=1.0):
            return torch.exp(-gamma * torch.sum((x - y) ** 2))

        # Test score with custom kernel
        start_time = time.time()
        score_np = score(samples_np, rbf_kernel_np, q=1, normalize=False)
        np_time = time.time() - start_time
        print(f"NumPy result: {score_np:.6f}, Time: {np_time:.4f} seconds")

        start_time = time.time()
        score_torch_result = score_torch(
            samples_torch, rbf_kernel_torch, q=1, normalize=False, device=device
        )
        torch_time = time.time() - start_time
        print(
            f"PyTorch result: {score_torch_result.cpu().numpy():.6f}, Time: {torch_time:.4f} seconds"
        )
        print(f"Speedup: {np_time / torch_time:.2f}x")
        print(
            f"Absolute difference: {abs(score_np - score_torch_result.cpu().numpy()):.8f}"
        )

        # Test intdiv_X
        print("\nTesting intdiv_X:")

        start_time = time.time()
        intdiv_np = intdiv_X(X_np, q=2, normalize=True)
        np_time = time.time() - start_time
        print(f"NumPy result: {intdiv_np:.6f}, Time: {np_time:.4f} seconds")

        start_time = time.time()
        intdiv_torch_result = intdiv_X_torch(
            X_torch, q=2, normalize=True, device=device
        )
        torch_time = time.time() - start_time
        print(
            f"PyTorch result: {intdiv_torch_result.cpu().numpy():.6f}, Time: {torch_time:.4f} seconds"
        )
        print(f"Speedup: {np_time / torch_time:.2f}x")
        print(
            f"Absolute difference: {abs(intdiv_np - intdiv_torch_result.cpu().numpy()):.8f}"
        )
