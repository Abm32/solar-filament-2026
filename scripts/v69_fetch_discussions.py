import json, os, re, sys, base64, urllib.request, html
cred = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
auth = base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode()
def call(ep, body):
    req = urllib.request.Request(f"https://www.kaggle.com/api/i/discussions.DiscussionsService/{ep}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Basic " + auth})
    return json.load(urllib.request.urlopen(req, timeout=60))
out = sys.argv[1]; since = sys.argv[2]
topics = []
for page in range(1, 8):
    r = call("GetTopicListByForumId", {"forumId": 11317517, "page": page})
    t = r.get("topics", [])
    if not t: break
    topics += t

json.dump(topics, open(f"{out}/topics.json", "w"))
strip = lambda s: html.unescape(re.sub(r"<[^>]+>", " ", s or ""))
lines = []
def walk(c, d=0):
    a = c.get("author", {}).get("displayName") or c.get("authorUserDisplayName", "?")
    lines.append(f"{'  '*d}- [{c.get('postDate','')[:16]}] {a}: {strip(c.get('rawMarkdown') or c.get('content'))[:3000]}")
    for k in c.get("replies", []) or []: walk(k, d + 1)
for t in topics:
    if t.get("lastCommentPostDate", "") < since: continue
    r = call("GetForumTopicById", {"forumTopicId": t["id"], "includeComments": True})
    ft = r["forumTopic"]
    lines.append(f"\n## {t['id']} {ft.get('name')} (by {ft.get('authorUserDisplayName')}, {ft.get('postDate','')[:10]}, last {t['lastCommentPostDate'][:16]})")
    fm = ft.get("firstMessage", {})
    lines.append(strip(fm.get("rawMarkdown") or fm.get("content"))[:5000])
    for c in ft.get("comments", []) or []: walk(c)
open(f"{out}/discussions_since.txt", "w").write("\n".join(lines))
print(len(topics), "topics;", sum(1 for t in topics if t.get("lastCommentPostDate","") >= since), "active since", since)
