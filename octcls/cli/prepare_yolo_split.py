def main():
    import argparse
    from pathlib import Path
    from octcls.data.prepare_split import SplitConfig, prepare_yolo_split

    p = argparse.ArgumentParser(description="Create YOLO split with symlinks or copies")
    p.add_argument("--images_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--labels_root", default=None)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--names", nargs="+", default=["roi"])
    p.add_argument("--copy", action="store_true")
    p.add_argument("--clobber", action="store_true")
    p.add_argument("--allow_missing_labels", action="store_true")
    args = p.parse_args()

    cfg = SplitConfig(
        images_root=Path(args.images_root).expanduser().resolve(),
        out_root=Path(args.out_root).expanduser().resolve(),
        labels_root=Path(args.labels_root).expanduser().resolve() if args.labels_root else None,
        val_size=args.val_size,
        seed=args.seed,
        names=tuple(args.names),
        copy=bool(args.copy),
        clobber=bool(args.clobber),
        allow_missing_labels=bool(args.allow_missing_labels),
    )
    yaml_path = prepare_yolo_split(cfg)
    print(f"Prepared dataset at {cfg.out_root}; YAML: {yaml_path}")


if __name__ == "__main__":
    main()
