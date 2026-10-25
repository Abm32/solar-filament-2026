cd "$HOME/solar-filament-2026"; mkdir -p outputs/v77
V75_PROCS=24 .venv/bin/python scripts/v77_refined_1280.py > outputs/v77/run.log 2>&1
touch outputs/v77/done
