import glob
import html
import json
import re
import sys


def strip(h):
    t = re.sub("<[^>]+>", " ", h or "")
    return html.unescape(re.sub("[ \t]+", " ", t)).strip()


def body_of(m):
    return strip(m.get("message") or m.get("rawMarkdown") or "")


def author_of(m):
    return m.get("authorUserDisplayName") or (m.get("author") or {}).get("displayName", "?")


def walk(msgs, out, depth=0):
    for m in msgs or []:
        b = body_of(m)
        au = author_of(m)
        if b:
            out.append(("  " * depth) + "[" + au + "] " + b)
        for key in ("comments", "replies", "children", "messages"):
            if m.get(key):
                walk(m.get(key), out, depth + 1)


def main():
    tids = sys.argv[1:] or [p.split("_")[-1].split(".")[0] for p in glob.glob("outputs/topic_*.json")]
    for tid in tids:
        try:
            d = json.load(open("outputs/topic_%s.json" % tid))
        except Exception as e:
            print("skip", tid, e)
            continue
        ft = d.get("forumTopic") or d
        print("\n" + "=" * 70)
        print("TOPIC", tid, "::", ft.get("name") or ft.get("title"))
        print("=" * 70)
        out = []
        fm = ft.get("firstMessage")
        if isinstance(fm, dict):
            b = body_of(fm)
            if b:
                out.append("[OP:" + author_of(fm) + "] " + b)
        top = ft.get("message")
        if top and not out:
            out.append("[OP:" + ft.get("authorUserDisplayName", "?") + "] " + strip(top))
        for key in ("messages", "comments", "forumMessages", "replies"):
            if ft.get(key):
                walk(ft.get(key), out)
        if out:
            print("\n\n".join(o[:1800] for o in out)[:7000])
        else:
            print("(no parsed body; keys=%s)" % sorted(ft.keys()))


if __name__ == "__main__":
    main()
