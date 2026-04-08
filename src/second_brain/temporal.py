from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from .retrieval import retrieve


def topic_timeline(topic: str, k: int = 30) -> str:
    docs = retrieve(topic, k=k)
    if not docs:
        return "Aucun résultat pour ce sujet."

    buckets: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        ts = d.metadata.get("source_mtime_ts")
        if ts is None:
            continue
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        key = f"{dt.year:04d}-{dt.month:02d}"
        snippet = d.page_content[:180].replace("\n", " ")
        buckets[key].append(snippet)

    if not buckets:
        return "Résultats trouvés, mais sans métadonnées temporelles exploitables."

    lines = [f"Timeline pour: {topic}"]
    for month in sorted(buckets):
        lines.append(f"\n{month}")
        preview = buckets[month][:3]
        for item in preview:
            lines.append(f"- {item}")
        if len(buckets[month]) > 3:
            lines.append(f"- ... ({len(buckets[month]) - 3} éléments de plus)")

    return "\n".join(lines)
