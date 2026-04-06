"""Quick benchmark: OMP_NUM_THREADS impact on QNG_full VQE (4 qubits)."""
import os, subprocess, sys, time, json

N_STEPS = 3
THREAD_COUNTS = [1, 2, 4]

def run_bench(n_steps):
    from src.models import make_vqe_circuit, init_params_vqe
    from src.training import train_vqe

    circuit, _H = make_vqe_circuit(4, 4, 1.0, 1.0)
    params = init_params_vqe(4, 4, seed=0)

    t0 = time.perf_counter()
    train_vqe(circuit, params, "QNG_full", lr=0.01, n_steps=n_steps,
              n_layers=4, verbose=False)
    elapsed = time.perf_counter() - t0
    print(json.dumps({"elapsed": elapsed, "per_step": elapsed / n_steps}))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_bench(N_STEPS)
    else:
        print(f"Benchmark: {N_STEPS} steps of VQE/QNG_full (4 qubits, 4 layers)")
        print(f"{'Threads':>8s}  {'Total (s)':>10s}  {'Per step (s)':>12s}")
        print("-" * 35)

        for nt in THREAD_COUNTS:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(nt)
            env["MKL_NUM_THREADS"] = str(nt)
            env["OPENBLAS_NUM_THREADS"] = str(nt)
            result = subprocess.run(
                [sys.executable, __file__, "--child"],
                capture_output=True, text=True, env=env,
            )
            if result.returncode != 0:
                print(f"{nt:>8d}  FAILED: {result.stderr.strip()}")
                continue
            for line in result.stdout.strip().splitlines():
                try:
                    data = json.loads(line)
                    print(f"{nt:>8d}  {data['elapsed']:>10.2f}  {data['per_step']:>12.2f}")
                except json.JSONDecodeError:
                    pass

        print("\nDone.")
