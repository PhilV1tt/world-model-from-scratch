"""Repack un dataset parking HDF5 avec un chunking adapte a la lecture par frame.

Le dataset v2 stocke obs en chunks=(N,4,4,1), ce qui oblige a lire des centaines
de chunks pour une seule frame 64x64x3 (lecture aleatoire pathologique pendant
l'entrainement). On reecrit obs et goal_obs en chunks=(1,64,64,3) et on copie tout
le reste verbatim. src/data.py lit le fichier sans modification.

Usage:
    python scripts/repack_h5.py --src data/parking/train_v2.h5 --dst data/parking/train_v2_fast.h5
"""
import argparse
import time

import h5py
import numpy as np

REWRITE = ("obs", "goal_obs")


def repack(src_path: str, dst_path: str, block: int = 4096) -> None:
    with h5py.File(src_path, "r") as fsrc, h5py.File(dst_path, "w") as fdst:
        for k, v in fsrc.attrs.items():
            fdst.attrs[k] = v
        for name in fsrc.keys():
            if name in REWRITE:
                src_ds = fsrc[name]
                n = src_ds.shape[0]
                chunks = (1,) + tuple(src_ds.shape[1:])
                dst_ds = fdst.create_dataset(
                    name, shape=src_ds.shape, dtype=src_ds.dtype,
                    chunks=chunks, compression="lzf",
                )
                for k, val in src_ds.attrs.items():
                    dst_ds.attrs[k] = val
                t0 = time.time()
                for i in range(0, n, block):
                    dst_ds[i : i + block] = src_ds[i : i + block]
                print(f"  {name}: {src_ds.shape} chunks {src_ds.chunks} -> {chunks} ({time.time()-t0:.1f}s)")
            else:
                fsrc.copy(name, fdst)
                print(f"  {name}: copied verbatim {fsrc[name].shape}")


def verify(src_path: str, dst_path: str, n_samples: int = 64) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(src_path, "r") as a, h5py.File(dst_path, "r") as b:
        assert set(a.keys()) == set(b.keys()), (set(a.keys()), set(b.keys()))
        for name in ("obs", "goal_obs"):
            n = a[name].shape[0]
            idx = sorted(int(i) for i in rng.choice(n, size=min(n_samples, n), replace=False))
            for i in idx:
                assert np.array_equal(a[name][i], b[name][i]), f"{name}[{i}] differs"
        for name in a.keys():
            if name in REWRITE:
                continue
            assert np.array_equal(a[name][()], b[name][()]), f"{name} differs"
        # timing: random single-frame reads
        n = a["obs"].shape[0]
        idx = [int(i) for i in rng.choice(n, size=256, replace=False)]
        for label, f in (("src", a), ("dst", b)):
            t0 = time.time()
            for i in idx:
                _ = f["obs"][i]
            print(f"  random-read 256 frames [{label}]: {time.time()-t0:.3f}s")
    print("verify OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/parking/train_v2.h5")
    ap.add_argument("--dst", default="data/parking/train_v2_fast.h5")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        print(f"repack {args.src} -> {args.dst}")
        repack(args.src, args.dst)
    verify(args.src, args.dst)
