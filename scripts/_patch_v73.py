from pathlib import Path
p = Path("scripts/_v73_launch.sh"); t = p.read_text()
old = '  [ "$dev" = cpu ] && dev=""\n'
new = old + '''  # WAIT_V68="gpu:recipe ...": hold this worker until that GPU's V68 chain has finished
  local w; w=$(tr " " "\\n" <<< "${WAIT_V68:-}" | grep "^$1:" | cut -d: -f2)
  if [ -n "$w" ]; then
    status "worker $1 waiting for V68 $w"
    while [ ! -f "outputs/v68/chain_$w.done" ]; do sleep 60; done
  fi
'''
assert t.count(old) == 1
p.write_text(t.replace(old, new))
print("ok")
