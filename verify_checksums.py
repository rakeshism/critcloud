"""Compare regenerated result files with the checksums of the published run.
Bit-identical results are expected only with the pinned versions in requirements.txt
(or the Dockerfile); other NumPy/SciPy/BLAS builds may differ in the last decimals."""
import hashlib, pathlib, sys
ref = dict(l.split()[::-1] for l in pathlib.Path("CHECKSUMS.sha256").read_text().splitlines() if l.strip())
bad = [f for f, h in ref.items() if hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() != h]
print(f"{len(ref) - len(bad)}/{len(ref)} result files identical")
for f in bad: print("  differs:", f)
sys.exit(1 if bad else 0)
