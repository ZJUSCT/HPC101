#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void gemv(double* __restrict y, double* __restrict A, double* __restrict x, int N) {
    // y = A * x
    for (int i = 0; i < N; i++) {
        y[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y[i] += A[i * N + j] * x[j];
        }
    }
}

double dot_product(double* __restrict x, double* __restrict y, int N) {
    // dot product of x and y
    double result = 0.0;
    for (int i = 0; i < N; i++) {
        result += x[i] * y[i];
    }
    return result;
}

void precondition(double* __restrict A, double* __restrict K2_inv, int N) {
    // K2_inv = 1 / diag(A)
    for (int i = 0; i < N; i++) {
        K2_inv[i] = 1.0 / A[i * N + i];
    }
}

void precondition_apply(double* __restrict z, double* __restrict K2_inv, double* __restrict r, int N) {
    // z = K2_inv * r
    for (int i = 0; i < N; i++) {
        z[i] = K2_inv[i] * r[i];
    }
}

int bicgstab(int N, double* A, double* b, double* x, int max_iter, double tol) {
    /**
     * Algorithm: BICGSTAB
     *  r: residual
     *  r_hat: modified residual
     *  p: search direction
     *  K2_inv: preconditioner (We only store the diagonal of K2_inv)
     * Reference: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
     */
    double* r      = (double*)calloc(N, sizeof(double));
    double* r_hat  = (double*)calloc(N, sizeof(double));
    double* p      = (double*)calloc(N, sizeof(double));
    double* v      = (double*)calloc(N, sizeof(double));
    double* s      = (double*)calloc(N, sizeof(double));
    double* h      = (double*)calloc(N, sizeof(double));
    double* t      = (double*)calloc(N, sizeof(double));
    double* y      = (double*)calloc(N, sizeof(double));
    double* z      = (double*)calloc(N, sizeof(double));
    double* K2_inv = (double*)calloc(N, sizeof(double));

    double rho_old = 1, alpha = 1, omega = 1;
    double rho = 1, beta = 1;
    double tol_squared = tol * tol;

    // Take M_inv as the preconditioner
    // Note that we only use K2_inv (in wikipedia)
    precondition(A, K2_inv, N);

    // 1. r0 = b - A * x0
    gemv(r, A, x, N);
    for (int i = 0; i < N; i++) {
        r[i] = b[i] - r[i];
    }

    // 2. Choose an arbitary vector r_hat that is not orthogonal to r
    // We just take r_hat = r, please do not change this initial value
    memmove(r_hat, r, N * sizeof(double));  // memmove is safer memcpy :)

    // 3. rho_0 = (r_hat, r)
    rho = dot_product(r_hat, r, N);

    // 4. p_0 = r_0
    memmove(p, r, N * sizeof(double));

    int iter;
    for (iter = 1; iter <= max_iter; iter++) {
        if (iter % 1000 == 0) {
            printf("Iteration %d, residul = %e\n", iter, sqrt(dot_product(r, r, N)));
        }

        // 1. y = K2_inv * p (apply preconditioner)
        precondition_apply(y, K2_inv, p, N);

        // 2. v = Ay
        gemv(v, A, y, N);

        // 3. alpha = rho / (r_hat, v)
        alpha = rho / dot_product(r_hat, v, N);

        // 4. h = x_{i-1} + alpha * y
        for (int i = 0; i < N; i++) {
            h[i] = x[i] + alpha * y[i];
        }

        // 5. s = r_{i-1} - alpha * v
        for (int i = 0; i < N; i++) {
            s[i] = r[i] - alpha * v[i];
        }

        // 6. Is h is accurate enough, then x_i = h and quit
        if (dot_product(s, s, N) < tol_squared) {
            memmove(x, h, N * sizeof(double));
            break;
        }

        // 7. z = K2_inv * s
        precondition_apply(z, K2_inv, s, N);

        // 8. t = Az
        gemv(t, A, z, N);

        // 9. omega = (t, s) / (t, t)
        omega = dot_product(t, s, N) / dot_product(t, t, N);

        // 10. x_i = h + omega * z
        for (int i = 0; i < N; i++) {
            x[i] = h[i] + omega * z[i];
        }

        // 11. r_i = s - omega * t
        for (int i = 0; i < N; i++) {
            r[i] = s[i] - omega * t[i];
        }

        // 12. If x_i is accurate enough, then quit
        if (dot_product(r, r, N) < tol_squared) break;

        rho_old = rho;
        // 13. rho_i = (r_hat, r)
        rho = dot_product(r_hat, r, N);

        // 14. beta = (rho_i / rho_{i-1}) * (alpha / omega)
        beta = (rho / rho_old) * (alpha / omega);

        // 15. p_i = r_i + beta * (p_{i-1} - omega * v)
        for (int i = 0; i < N; i++) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
    }

    free(r);
    free(r_hat);
    free(p);
    free(v);
    free(s);
    free(h);
    free(t);
    free(y);
    free(z);
    free(K2_inv);

    if (iter >= max_iter)
        return -1;
    else
        return iter;
}
