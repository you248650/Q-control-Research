import argparse
from .pipeline import run

def main():
    p = argparse.ArgumentParser(description="NG-RC guided policy-gradient pulse design (Qiskit + TF)")
    p.add_argument("--cycles", type=int, default=2500, help="Optimization cycles")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size per policy update")
    p.add_argument("--use-nav", type=int, default=1, help="Use NG-RC navigation prescreen (1) or not (0)")
    p.add_argument("--backend", type=str, default="ibm_brisbane", help="IBM backend name for noise model")
    p.add_argument("--min-len", type=int, default=45, help="Min waveform length (samples)")
    p.add_argument("--max-len", type=int, default=70, help="Max waveform length (samples)")
    args = p.parse_args()

    run(
        cycles=args.cycles,
        batch_size=args.batch_size,
        use_nav=bool(args.use_nav),
        backend_name=args.backend,
        min_waveform_length=args.min_len,
        max_waveform_length=args.max_len,
    )
