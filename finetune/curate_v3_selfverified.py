"""Turn banked v3 trajectories into training data WITHOUT gold labels.

Plain rejection sampling cannot fix the defect we actually measured. The
dominant trainable error is MODEL-MISSED: 13.6 pct of the sealed set and 18.8
pct of the campaign set, where the owner's name sat in the gathered evidence
and the model abstained anyway. Those rows produce no answer, so a filter that
only keeps correct answers throws away exactly the cases we need to learn
from, and trains the model to keep doing what it already does.

The way out is that the verifier can EXTRACT, not just check. When the evidence
says "X - eigenaar of <this company>", the correct answer is derivable from the
evidence itself, with no gold label anywhere. So each banked trajectory is
sorted into one of four classes:

  KEEP      model answered and the evidence ties that person to THIS company
            as an owner. Reinforces correct behaviour.
  CORRECT   model abstained (or named someone the evidence does not tie to the
            company) while the evidence DOES name an owner of it. The final
            turn is rewritten to that owner. This is the class that fixes
            MODEL-MISSED, and it cannot come from rejection sampling.
  NEGATIVE  evidence names no owner of this company and the model abstained.
            Correct behaviour, kept as an abstention example.
  DROP      model named someone with no owner-evidence and none is derivable.
            Ambiguous, so it teaches nothing safe.

The company-token requirement is load-bearing. Validated against the two known
wrong-role errors in the sealed run: coopkracht's "Industrieel ontwerper" and
dsd-stalinrichting's "Planning, Verkoop binnendienst, PR & Marketing" are both
rejected, because neither is an owner word. An earlier, looser version that
only looked for an owner word nearby credited raypack.nl's "dga @ studium
academy", a real DGA of an entirely different firm.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

from finetune.curate_dataset import first_json_object, split_groups
from finetune.namefold import TUSSENVOEGSELS as TUSSEN
from finetune.namefold import fold

OWNER_WORDS = re.compile(
    r"eigenaar|oprichter|founder|owner|directeur|director|ceo|managing partner"
    r"|managing director|bestuurder|dga|mede-eigenaar|oprichtster",
    re.I,
)

# Letters that appear in real Dutch-market owner names. An ASCII-only class
# cannot match them at all: "Michal" with a Polish l-stroke fails the [a-z]
# tail, so the whole person match fails and the owner becomes invisible to
# extract_owner. Turkish, Polish and German surnames are common in the
# Netherlands, so this silently discarded a slice of every banked run.
# Case is kept strict (uppercase initial, lowercase tail) on purpose: allowing
# uppercase in the tail would match ALL-CAPS company names as people, and this
# evidence is full of them.
_UP = "A-ZÀ-ÖØ-ÞŁİŞĞĆŚŹŻČŠŽĐ"
_LO = "a-zß-öø-ÿłışğćśźżčšžđ"

# "Firstname Lastname" with optional Dutch tussenvoegsel, as it appears in
# search snippets and LinkedIn titles.
#
# The tussenvoegsel run must allow TWO words, not one. "Jan van den Brom" is a
# real owner this project has already verified by hand, and a single-word
# particle could not match it: "den" fails the uppercase-surname class, so the
# whole person match failed and the owner vanished. "van der", "van den" and
# "van de" are among the most common name forms in the Netherlands, which is
# the market this pool is drawn from. Matched case-insensitively because
# snippets capitalise particles at the start of a sentence.
PERSON = re.compile(
    rf"\b([{_UP}][{_LO}]{{1,20}})\s+"
    rf"((?:(?i:van|de|der|den|ter|ten|te|du|le|la|het)\s+){{0,2}})"
    rf"([{_UP}][{_LO}]{{2,25}})\b"
)

NOT_PERSON = {
    "onze", "over", "meer", "alle", "deze", "voor", "door", "het", "een",
    "welkom", "contact", "home", "nieuws", "bekijk", "lees", "neem", "wij",
    "agro", "agri", "the", "our", "read", "more", "view", "about", "team",
    "privacy", "cookie", "algemene", "januari", "februari", "maart", "april",
    "juni", "juli", "augustus", "september", "oktober", "november", "december",
    "tool", "error", "client", "found", "url", "http", "https", "web", "search",
    "linked", "linkedin", "facebook", "google", "bekijken", "profiel",
}


def surname(value: str) -> str:
    parts = [p for p in re.sub(r"[^a-z ]", " ", fold(value)).split()
             if p not in TUSSEN and len(p) > 1]
    return parts[-1] if parts else ""


def company_tokens(domain: str, company_name: str = "") -> list[str]:
    """Tokens that identify THIS company inside a window of evidence.

    A domain that concatenates a multi-word name into one unsplittable token is
    the common case here, not the exception: 36 pct of the v3 pool looks like
    farmoniefarmtech.nl for "Farmonie FarmTech". Pages write the name with a
    space, so the single token "farmoniefarmtech" never appears and the
    company-tie test could not fire at all. Measured on the 150-company pilot
    that alone dropped 49 rows the evidence plainly supported, including a
    LinkedIn title reading "Directeur bij Feed Farm BV".

    The extra tokens come from the company NAME but are kept only when they also
    appear inside the domain stem. That is deliberately tight: every token is
    still anchored to the domain, so this widens what can be RECOGNISED without
    widening what can be CREDITED. A looser version keyed on name words alone
    would re-open the raypack failure, where a real DGA of an entirely
    different firm was accepted.
    """
    stem = re.sub(r"\.(nl|com|eu|co\.uk|net|org|de|be)$", "", fold(domain))
    parts = [p for p in re.split(r"[^a-z0-9]+", stem) if len(p) > 3]
    flat = re.sub(r"[^a-z0-9]", "", stem)
    extra = [w for w in re.split(r"[^a-z0-9]+", fold(company_name))
             if len(w) > 3 and w in flat]
    return sorted(set(parts + extra)) or [stem]


def tied_to_company(evidence_raw: str, name: str, domain: str, window: int = 220,
                    company_name: str = "") -> bool:
    key = surname(name)
    if not key:
        return False
    ev = fold(evidence_raw)
    tokens = company_tokens(domain, company_name)
    for match in re.finditer(re.escape(key), ev):
        chunk = ev[max(0, match.start() - window): match.start() + window]
        if OWNER_WORDS.search(chunk) and any(t in chunk for t in tokens):
            return True
    return False


def extract_owner(evidence_raw: str, domain: str, window: int = 200,
                  company_name: str = "") -> str | None:
    """The person the EVIDENCE calls an owner of this company, if any.

    Scans windows containing both an owner word and a company token, and takes
    the most frequently co-occurring plausible person name.
    """
    tokens = company_tokens(domain, company_name)
    votes: Counter = Counter()
    for match in OWNER_WORDS.finditer(evidence_raw):
        chunk = evidence_raw[max(0, match.start() - window): match.start() + window]
        if not any(t in fold(chunk) for t in tokens):
            continue
        for person in PERSON.finditer(chunk):
            first, tussen, last = person.group(1), (person.group(2) or "").strip(), person.group(3)
            if first.lower() in NOT_PERSON or last.lower() in NOT_PERSON:
                continue
            if any(t == first.lower() or t == last.lower() for t in tokens):
                continue  # the company name itself, not a person
            full = " ".join(x for x in (first, tussen, last) if x)
            votes[full] += 1
    if not votes:
        return None
    best, count = votes.most_common(1)[0]
    return best if count >= 1 else None


def rewrite_final(final: dict, answer: dict) -> dict:
    content = final.get("content") or ""
    _, start, end = first_json_object(content)
    payload = json.dumps(answer, ensure_ascii=False, indent=2)
    if start < 0:
        return {**final, "content": payload}
    return {**final, "content": content[:start] + payload + content[end:]}


def process(row: dict) -> tuple[dict | None, str]:
    subject = row.get("subject") or {}
    domain = subject.get("domain") or ""
    company_name = subject.get("company_name") or ""
    preamble, groups, final = split_groups(row["messages"])
    if final is None or not groups:
        return None, "no-final"
    evidence = "\n".join(
        m.get("content") or "" for g in groups for m in g if m["role"] == "tool"
    )
    if not evidence.strip():
        return None, "no-evidence"

    answer, _, _ = first_json_object(final.get("content") or "")
    named = (answer or {}).get("owner_name")

    if named and tied_to_company(evidence, named, domain, company_name=company_name):
        return row, "KEEP"

    derived = extract_owner(evidence, domain, company_name=company_name)
    if derived:
        fixed = dict(answer or {})
        fixed["owner_name"] = derived
        fixed.setdefault("title", None)
        fixed["linkedin_url"] = (answer or {}).get("linkedin_url")
        fixed["confidence"] = "medium"
        messages = list(preamble)
        for group in groups:
            messages.extend(group)
        messages.append(rewrite_final(final, fixed))
        return {**row, "messages": messages}, ("CORRECT-abstention" if not named else "CORRECT-wrong-person")

    if not named:
        return row, "NEGATIVE"
    return None, "DROP-unverifiable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", required=True)
    parser.add_argument("--out", default="finetune/data/v3/curated.jsonl")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(str(Path(args.traj_dir) / "trajectories-*.jsonl"))):
        rows += [json.loads(line) for line in open(path, encoding="utf-8")]

    kept, kinds = [], Counter()
    for row in rows:
        out, kind = process(row)
        kinds[kind] += 1
        if out is not None:
            out.setdefault("meta", {})
            out["meta"] = {**(out.get("meta") or {}), "v3_class": kind}
            kept.append(out)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"banked trajectories: {len(rows)}")
    for kind, n in kinds.most_common():
        print(f"  {kind:22} {n:4}")
    print(f"\nusable training rows: {len(kept)}  -> {dest}")


if __name__ == "__main__":
    main()
