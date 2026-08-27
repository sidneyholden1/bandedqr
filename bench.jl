# ApproxFun / Olver-Townsend almost-banded QR benchmark (factor & solve, ms).
# Emits one JSON line: {"1024":{"factor":..,"solve":..}, ...}
using BandedMatrices, FastAlmostBandedMatrices, LinearAlgebra, BenchmarkTools, Printf
makeA(N; l=2, u=2, r=2) = begin
    B = brandn(Float64, N, N, l, u)
    @inbounds for i in 1:N; B[i,i] += (l+u+3.0); end
    AlmostBandedMatrix(B, randn(r, N))
end
Ns = length(ARGS) > 0 ? parse.(Int, ARGS) : [1024, 4096, 16384]
print("{")
for (idx, N) in enumerate(Ns)
    A = makeA(N); b = randn(N)
    tf = @belapsed qr(Ac) setup=(Ac=copy($A)) evals=1 samples=30
    F  = qr(copy(A)); ts = @belapsed ($F \ $b) samples=200
    @printf("\"%d\":{\"factor\":%.4f,\"solve\":%.4f}%s", N, tf*1e3, ts*1e3, idx < length(Ns) ? "," : "")
end
println("}")
