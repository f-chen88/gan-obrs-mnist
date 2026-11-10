# Sourced by the search jobs. ROOT = repository root on Juliet.
set -u
ROOT="${ROOT:?ROOT not set}"
cd "$ROOT"
[ -f venv/bin/activate ] && source venv/bin/activate   # absent on SSP Cloud
export DATA="$(cd "${DATA:-/projects/m25146/data/}" && pwd)"   # absolute: the jobs run in search_work/<task>/
export PYTHONUNBUFFERED=1

export CONFIGS="${CONFIGS:-search/configs.txt}"   # relative to ROOT
export RES="${RES:-results_search}"                # results folder, relative to ROOT

config_line () {   # config_line <index> -> "name|options" of the index-th non-comment line (0-based)
  grep -v '^\s*#' "$ROOT/$CONFIGS" | grep -v '^\s*$' | sed -n "$(( $1 + 1 ))p"
}
run_name ()    { config_line "$1" | cut -d'|' -f1 | xargs; }
run_options () { config_line "$1" | cut -d'|' -f2- | xargs; }
run_arch () {   # D architecture of a trained run (config.json), 'mlp' by default
  python -c "import json,sys; print(json.load(open(sys.argv[1])).get('D_arch','mlp'))" \
    "$ROOT/checkpoints/search/$1/config.json" 2>/dev/null || echo mlp
}

workdir () {   # private working directory: generate.py / evaluate.py write samples/ and gen_log.json
  local w="$ROOT/search_work/$1"
  mkdir -p "$w"
  for f in "$ROOT"/*.py; do ln -sf "$f" "$w/"; done
  ln -sfn "$ROOT/checkpoints" "$w/checkpoints"
  ln -sf "$ROOT/eval_classifier.pth" "$w/"
  ln -sf "$ROOT/incep_real_test.pt" "$w/"
  cd "$w"
}

evalcfg () {   # evalcfg "<label>" <csv> <seed> <generate.py options...>
  local label="$1" csv="$2" seed="$3"; shift 3
  echo "=== $label | seed $seed"
  rm -rf samples gen_log.json
  python generate.py --batch_size 256 --seed "$seed" "$@" \
    && python evaluate.py --inception --label "$label | seed $seed" --out "$csv"
}
