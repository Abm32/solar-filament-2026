from pathlib import Path
def sub(path, old, new):
    p = Path(path); t = p.read_text()
    assert t.count(old) == 1, (path, old[:60], t.count(old))
    p.write_text(t.replace(old, new))
T = "scripts/train_v61_fold.py"
sub(T, '''           "p2first_m", "p2first_1280", "p2first_long", "p2first_flip")
V68_ARMS = ("p2first_m", "p2first_1280", "p2first_long", "p2first_flip")''',
'''           "p2first_m", "p2first_1280", "p2first_long", "p2first_flip", "p2first_rot")
V68_ARMS = ("p2first_m", "p2first_1280", "p2first_long", "p2first_flip", "p2first_rot")''')
sub(T, '''        args = dict(args, fliplr=0.5, flipud=0.5)
    return args''', '''        args = dict(args, fliplr=0.5, flipud=0.5)
    if recipe == "p2first_rot":     # V68 addendum: full-angle rotation + flips
        args = dict(args, degrees=180, fliplr=0.5, flipud=0.5)
    return args''')
print("ok")
