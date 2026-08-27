"""Batched banded QR solve on the Apple GPU (Metal, via MLX) -- the partridge.

Many independent same-shape banded systems (one per spectral/Fourier mode) are
solved at once: one GPU thread per system runs the full Givens band QR factor +
solve.  Pivot-free => no warp divergence; batch is the fastest array axis =>
coalesced access.  fp32 (Apple GPUs are fp32-native; fine for well-conditioned
banded systems, not for the ill-conditioned spectral operators -- use the CPU
fp64 path there).

    from bandqr_gpu import solve_batched
    X = solve_batched(ab, b, kl, ku)   # ab: (n, kl+ku+1, batch), b: (n, batch)
"""
import mlx.core as mx

_CACHE = {}


def _kernel(n, kl, ku, batch):
    key = (n, kl, ku, batch)
    if key in _CACHE:
        return _CACHE[key]
    win = kl + ku + 1            # input band width
    W = 2 * kl + ku + 2          # working width (sub | diag | super R | guard)
    src = f"""
        const int N={n}, KL={kl}, KU={ku}, WIN={win}, W={W}, BATCH={batch};
        uint s = thread_position_in_grid.x;
        if (s >= (uint)BATCH) return;
        // load compact band into the wider working band (guard region -> 0)
        for (int i = 0; i < N; i++) {{
            for (int o = 0; o < W; o++)
                wab[(i*W+o)*BATCH+s] = (o < WIN) ? ab[(i*WIN+o)*BATCH+s] : 0.0f;
            wb[i*BATCH+s] = b[i*BATCH+s];
        }}
        // Givens band QR, rotations fused into wb (no need to store them)
        for (int jc = 0; jc < N; jc++) {{
            int ilast = jc+KL < N-1 ? jc+KL : N-1;
            for (int q = ilast; q >= jc+1; q--) {{
                int p = q-1;
                float a  = wab[(p*W + (jc-p+KL))*BATCH+s];
                float bb = wab[(q*W + (jc-q+KL))*BATCH+s];
                float c, sn;
                if (bb == 0.0f) {{ c = 1.0f; sn = 0.0f; }}
                else {{ float t = sqrt(a*a+bb*bb); c = a/t; sn = bb/t; }}
                int jmax = q+KU+KL < N-1 ? q+KU+KL : N-1;
                for (int j = jc; j <= jmax; j++) {{
                    int ip = (p*W + (j-p+KL))*BATCH+s;
                    int iq = (q*W + (j-q+KL))*BATCH+s;
                    float ap = wab[ip], aq = wab[iq];
                    wab[ip] = c*ap + sn*aq;
                    wab[iq] = c*aq - sn*ap;
                }}
                float bp = wb[p*BATCH+s], bq = wb[q*BATCH+s];
                wb[p*BATCH+s] = c*bp + sn*bq;
                wb[q*BATCH+s] = c*bq - sn*bp;
            }}
        }}
        // banded back-substitution
        for (int i = N-1; i >= 0; i--) {{
            int jhi = i+KU+KL < N-1 ? i+KU+KL : N-1;
            float sum = 0.0f;
            for (int j = i+1; j <= jhi; j++)
                sum += wab[(i*W + (j-i+KL))*BATCH+s] * x[j*BATCH+s];
            x[i*BATCH+s] = (wb[i*BATCH+s] - sum) / wab[(i*W+KL)*BATCH+s];
        }}
    """
    k = mx.fast.metal_kernel(
        name=f"bandqr_{n}_{kl}_{ku}_{batch}",
        input_names=["ab", "b"],
        output_names=["x", "wab", "wb"],
        source=src,
    )
    _CACHE[key] = (k, W)
    return _CACHE[key]


def solve_batched(ab, b, kl, ku, threadgroup=256):
    """Solve `batch` banded systems at once on the GPU.
    ab : (n, kl+ku+1, batch) fp32 band storage  (batch is the fastest axis)
    b  : (n, batch) fp32
    returns x : (n, batch) fp32.
    """
    ab = mx.array(ab, dtype=mx.float32)
    b = mx.array(b, dtype=mx.float32)
    n, win, batch = ab.shape
    assert win == kl + ku + 1, "band width must be kl+ku+1"
    kern, W = _kernel(n, kl, ku, batch)
    x, _, _ = kern(
        inputs=[ab, b],
        output_shapes=[(n, batch), (n, W, batch), (n, batch)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(batch, 1, 1),
        threadgroup=(min(threadgroup, batch), 1, 1),
    )
    mx.eval(x)
    return x
