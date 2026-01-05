"""
Activation Functions from Scratch

This module implements activation functions with both forward and backward passes.
Each function transforms its input non-linearly and must compute gradients for
backpropagation.

Key Concepts:
- Forward pass: compute output from input
- Backward pass: compute gradient of loss w.r.t. input, given gradient of loss w.r.t. output
- Chain rule: if y = f(x) and L is loss, then dL/dx = dL/dy * dy/dx
"""

import numpy as np


class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.

    Forward:
        y = max(0, x)

    This is the simplest and most widely used activation function.
    It introduces non-linearity by zeroing out negative values.

    Backward:
        dy/dx = 1 if x > 0, else 0

    The gradient is 1 for positive inputs (pass gradient through unchanged)
    and 0 for negative inputs (kill the gradient).

    Note: The gradient at exactly x=0 is technically undefined, but we
    conventionally set it to 0.
    """

    def __init__(self):
        self.cache = None  # Store input for backward pass

    def forward(self, x):
        """
        Compute ReLU activation.

        Args:
            x: Input array of any shape

        Returns:
            Output array of same shape, with negative values replaced by 0
        """
        # Cache input for backward pass - we need to know which elements were > 0
        self.cache = x

        # Element-wise maximum with 0
        # np.maximum broadcasts and compares element-wise
        return np.maximum(0, x)

    def backward(self, grad_output):
        """
        Compute gradient of loss w.r.t. input.

        Args:
            grad_output: Gradient of loss w.r.t. output (dL/dy)
                         Same shape as forward output

        Returns:
            Gradient of loss w.r.t. input (dL/dx)
            Same shape as forward input

        Math:
            dL/dx = dL/dy * dy/dx
                  = grad_output * (1 if x > 0 else 0)
                  = grad_output * (x > 0)
        """
        x = self.cache

        # (x > 0) creates a boolean array: True where x > 0, False elsewhere
        # When multiplied, True becomes 1.0 and False becomes 0.0
        # This implements the piecewise gradient: 1 for positive, 0 for non-positive
        return grad_output * (x > 0)


class Softmax:
    """
    Softmax activation function.

    Forward:
        softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    Converts a vector of real numbers into a probability distribution.
    All outputs are positive and sum to 1.

    Numerical Stability:
        Computing exp(x) directly can overflow for large x values.
        We use the identity: softmax(x) = softmax(x - c) for any constant c.
        By choosing c = max(x), we ensure all exponents are <= 0,
        preventing overflow while maintaining numerical precision.

    Backward:
        The Jacobian of softmax is complex:
        d(softmax_i)/d(x_j) = softmax_i * (delta_ij - softmax_j)

        Where delta_ij = 1 if i==j, else 0 (Kronecker delta)

        For a single sample, if s = softmax(x):
        Jacobian = diag(s) - outer(s, s)

        For efficiency, we use the identity:
        dL/dx = s * (dL/ds - sum_i(dL/ds_i * s_i))
    """

    def __init__(self, eps=1e-8):
        self.eps = eps  # For numerical stability
        self.output = None  # Cache softmax output for backward

    def forward(self, x, axis=-1):
        """
        Compute softmax along specified axis.

        Args:
            x: Input array of shape (..., num_classes) typically
            axis: Axis along which to compute softmax (default: last axis)

        Returns:
            Probability distribution along specified axis
            Same shape as input, values in (0, 1), sum to 1 along axis
        """
        # Step 1: Subtract max for numerical stability
        # keepdims=True maintains shape for broadcasting
        x_max = np.max(x, axis=axis, keepdims=True)
        x_shifted = x - x_max  # Now max value is 0, all others are negative

        # Step 2: Compute exponentials
        # Since x_shifted <= 0, exp(x_shifted) <= 1, no overflow possible
        exp_x = np.exp(x_shifted)

        # Step 3: Normalize to get probabilities
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        self.output = exp_x / sum_exp

        return self.output

    def backward(self, grad_output):
        """
        Compute gradient of loss w.r.t. input.

        Args:
            grad_output: Gradient of loss w.r.t. softmax output (dL/ds)
                         Same shape as forward output

        Returns:
            Gradient of loss w.r.t. input (dL/dx)
            Same shape as forward input

        Math:
            For softmax output s and input x:
            dL/dx_i = sum_j(dL/ds_j * ds_j/dx_i)

            Using the softmax Jacobian:
            ds_j/dx_i = s_j * (delta_ij - s_i)

            This simplifies to:
            dL/dx = s * (dL/ds - sum(dL/ds * s))

            Intuition:
            - Each input x_i affects all outputs through the denominator
            - The gradient combines direct effect and indirect effects
        """
        s = self.output

        # Compute the dot product sum(grad_output * s) for each sample
        # This captures how much the normalization affects the gradient
        sum_grad_s = np.sum(grad_output * s, axis=-1, keepdims=True)

        # Apply the softmax backward formula
        # s * grad_output: direct effect of x_i on s_i
        # s * sum_grad_s: indirect effect through normalization
        grad_input = s * (grad_output - sum_grad_s)

        return grad_input


# =============================================================================
# GRADIENT VERIFICATION UTILITIES
# =============================================================================

def numerical_gradient(func, x, eps=1e-5):
    """
    Compute numerical gradient using central difference.

    This is used to verify our analytical gradients are correct.
    The numerical gradient is an approximation:
        df/dx ≈ (f(x + eps) - f(x - eps)) / (2 * eps)

    Central difference is more accurate than forward difference because
    it cancels out the second-order error term.

    Args:
        func: Function that takes x and returns a scalar
        x: Point at which to compute gradient
        eps: Small perturbation size

    Returns:
        Numerical gradient, same shape as x
    """
    grad = np.zeros_like(x)

    # Iterate over each element of x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        # Store original value
        original = x[idx]

        # Compute f(x + eps)
        x[idx] = original + eps
        f_plus = func(x)

        # Compute f(x - eps)
        x[idx] = original - eps
        f_minus = func(x)

        # Central difference
        grad[idx] = (f_plus - f_minus) / (2 * eps)

        # Restore original value
        x[idx] = original

        it.iternext()

    return grad


def check_gradient(layer, x, eps=1e-5, tolerance=1e-4):
    """
    Verify that analytical gradient matches numerical gradient.

    Args:
        layer: Layer with forward() and backward() methods
        x: Input to test
        eps: Perturbation size for numerical gradient
        tolerance: Maximum allowed relative error

    Returns:
        True if gradients match within tolerance
    """
    # Forward pass
    y = layer.forward(x.copy())

    # Random gradient from "upstream" (simulates dL/dy)
    grad_output = np.random.randn(*y.shape)

    # Analytical gradient
    analytical_grad = layer.backward(grad_output)

    # Numerical gradient
    def loss_func(x_test):
        y_test = layer.forward(x_test)
        return np.sum(y_test * grad_output)

    numerical_grad = numerical_gradient(loss_func, x.copy(), eps)

    # Compare
    diff = np.abs(analytical_grad - numerical_grad)
    max_diff = np.max(diff)
    relative_error = max_diff / (np.max(np.abs(numerical_grad)) + eps)

    if relative_error < tolerance:
        print(f"Gradient check PASSED. Max difference: {max_diff:.2e}")
        return True
    else:
        print(f"Gradient check FAILED. Max difference: {max_diff:.2e}")
        print(f"Relative error: {relative_error:.2e}")
        return False


# =============================================================================
# SIMPLE TESTS
# =============================================================================

if __name__ == "__main__":
    print("Testing ReLU...")
    relu = ReLU()

    # Test forward
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    y = relu.forward(x)
    print(f"  Input:  {x}")
    print(f"  Output: {y}")  # Expected: [0, 0, 0, 1, 2]

    # Test gradient check
    x_test = np.random.randn(5).astype(np.float64)
    check_gradient(relu, x_test)

    print("\nTesting Softmax...")
    softmax = Softmax()

    # Test forward
    x = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float64)
    y = softmax.forward(x)
    print(f"  Input:\n{x}")
    print(f"  Output:\n{y}")
    print(f"  Sum per row: {np.sum(y, axis=1)}")  # Should be [1, 1]

    # Test gradient check
    x_test = np.random.randn(2, 4).astype(np.float64)
    check_gradient(softmax, x_test)
