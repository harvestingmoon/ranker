"""
Imports the real gigs and provider profiles extracted by the BT4103-Scrape-and-Tag
pipeline (docs/hirers.csv, docs/providers.csv) into this repo's JSON format, so
the Stage-1 retrievers can pool candidates for LLM judging and the Stage-2
rankers can train on real data instead of the synthetic corpus.

Writes to pipeline/data_sat/ -- a separate folder, so the synthetic data/ and
every result built on it stay untouched:
    hirers.json     [{hire_id, hire_title, hire_description, hire_description_additional_notes, ...}]
    providers.json  [{provider_id, about_title, about_description, services_offered_title,
                      services_offered_description, relevant_experience, ...}]
    id_map.json     {"hirers": {id: source_file}, "providers": {id: source_file}}

Filters (the same quality bar Scrape-and-Tag's labeller/sample.py uses for gigs):
  gigs       grounding review kept it on the first pass (no repair), current prompt
             format (has a "Deliverable:" line), real industry tag (not OTHER), no
             additional notes, unique title, 300-1200 chars once the engagement line
             is stripped.
  providers  classified PROVIDER (not UNCERTAIN) with a headline and an About
             section. Profiles with some empty fields are kept: the provider side
             is the search corpus, so it should be as large and realistic as the
             data allows -- the per-gig judging pool, not the corpus, sets LLM cost.

"Engagement duration: ..." is stripped from every gig description: Scrape-and-Tag's
extract prompt appends it to every gig, so it says nothing about fit.

Run: python pipeline/import_scrape_and_tag.py [--src ../BT4103-Scrape-and-Tag/docs]
"""
import argparse
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "data_sat"
DEFAULT_SRC = BASE.parent.parent / "BT4103-Scrape-and-Tag" / "docs"

ENGAGEMENT_RE = re.compile(r"\s*Engagement duration:[^\n]*$", re.IGNORECASE)
HIRER_FIELDS = ["hire_title", "hire_description", "hire_description_additional_notes"]
PROVIDER_FIELDS = ["about_title", "about_description", "services_offered_title",
                   "services_offered_description", "relevant_experience"]
# carried along for analysis only; nothing in the ranker reads them
META_FIELDS = ["industry", "secondary_industry"]

csv.field_size_limit(10**9)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean(v: str) -> str:
    s = (v or "").strip()
    if s[:1] == "'" and s[1:2] in ("=", "+", "-", "@", "\t", "\r"):
        s = s[1:]  # undo extract.py's CSV-formula guard
    return "" if s == "null" else s


def written_hirers(manifest: Path) -> dict:
    """source_file -> the extract_manifest entry of the attempt that wrote its row."""
    out = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("entity_type") == "HIRER" and r.get("status") == "written":
                    out[r["file"]] = r
    return out


def good_hirers(rows: list[dict], manifest: dict) -> list[dict]:
    seen, keep = set(), []
    for r in rows:
        m = manifest.get(r["source_file"], {})
        desc = ENGAGEMENT_RE.sub("", clean(r["hire_description"])).strip()
        title = clean(r["hire_title"])
        if ((m.get("review") or {}).get("decision") != "keep" or m.get("repaired")
                or "Deliverable:" not in desc
                or r.get("industry", "") in ("", "OTHER")
                or clean(r["hire_description_additional_notes"])
                or not 300 <= len(desc) <= 1200
                or title.lower() in seen):
            continue
        seen.add(title.lower())
        keep.append({**r, "hire_title": title, "hire_description": desc,
                     "hire_description_additional_notes": ""})
    return keep


def good_providers(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if r["classify_label"] == "PROVIDER"
            and clean(r["about_title"]) and clean(r["about_description"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="Scrape-and-Tag docs/ folder (default: sibling checkout)")
    args = ap.parse_args()

    hirers = good_hirers(load_csv(args.src / "hirers.csv"),
                         written_hirers(args.src / "extract_manifest.jsonl"))
    providers = good_providers(load_csv(args.src / "providers.csv"))

    # ids follow source_file order, so a re-import of the same CSVs gives the same ids
    hirers.sort(key=lambda r: r["source_file"])
    providers.sort(key=lambda r: r["source_file"])
    h_out = [{"hire_id": i, **{k: clean(r[k]) for k in HIRER_FIELDS + META_FIELDS}}
             for i, r in enumerate(hirers, 1)]
    p_out = [{"provider_id": i, **{k: clean(r[k]) for k in PROVIDER_FIELDS + META_FIELDS}}
             for i, r in enumerate(providers, 1)]
    id_map = {"hirers": {i: r["source_file"] for i, r in enumerate(hirers, 1)},
              "providers": {i: r["source_file"] for i, r in enumerate(providers, 1)}}

    OUT_DIR.mkdir(exist_ok=True)
    for name, obj in [("hirers.json", h_out), ("providers.json", p_out), ("id_map.json", id_map)]:
        (OUT_DIR / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(h_out)} gigs, {len(p_out)} providers -> {OUT_DIR}")


if __name__ == "__main__":
    main()
