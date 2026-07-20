# DACH — Germany/Austria/Switzerland owner-research playbook

**Core loop:** company website `/impressum` → registry mirror → search-snippet fallback. German law (§5 DDG; AT §5 ECG; CH UWG) **requires** an Impressum naming the legal entity + responsible person — `https://{domain}/impressum` is the single highest-yield fetch (e.g. "Geschäftsführer: Thilo Salmon · Tim Mois").

## 1. Registries & free lookups (verified)
- DE **northdata.de** — `northdata.de/{Name}+{Rechtsform},{Stadt}` — HRB, address; current GF paywalled BUT free "Historie" publication texts often leak names. Fetchable.
- DE **companyhouse.de** — GF names free but 403s bots → read via search snippets: `site:companyhouse.de {company}` (snippet: "Thilo Salmon - Geschäftsführer der sipgate GmbH").
- DE handelsregister.de / unternehmensregister.de — session/form-based, NOT agent-fetchable. Skip.
- AT **firmenabc.at** — GOLDMINE: Geschäftsführer, Prokuristen, Gesellschafter + stakes, FN nr, all free. Find via `site:firmenabc.at {name}`, then fetch the page.
- AT firmen.wko.at — `firmen.wko.at/suche/?firma={q}` company lists (no people names). Fetchable.
- CH zefix.ch — SPA, not fetchable → `site:zefix.ch {name}` snippets for existence/UID.
- CH **moneyhouse.ch** — mgmt names on free tier but 403s bots → `site:moneyhouse.ch {company}` snippets.

**Snippet trick (critical):** any 403/paywalled source — run a `site:{domain} {company}` search; GF names usually appear in the result title/snippet.

## 2. German search terms
- People: Geschäftsführer (GmbH/UG MD) · Inhaber (owner, e.K./sole trader) · Gründer (founder) · Eigentümer · Gesellschafter (shareholder) · Vorstand (AG board) · Geschäftsleitung.
- Queries: `{company} Geschäftsführer` · `{company} Impressum` · `{company} Gründer Interview` · `Inhaber {company} {Stadt}`
- Legal forms decode the target: **GmbH/UG** → Geschäftsführer; **e.K./Einzelunternehmen** → Inhaber IS the owner (often in the company name); **AG** → Vorstandsvorsitzender; **GmbH & Co. KG** → GF of the Komplementär-GmbH.

## 3. LinkedIn & XING (snippet-only, never fetch profiles)
- `site:de.linkedin.com/in Geschäftsführer {company}` or `site:linkedin.com/in {company} Gründer OR founder OR CEO` — name + title in snippet.
- XING still relevant for traditional Mittelstand/Handwerk/older owners: `site:xing.com/profile {company}` snippets. Tech/marketing → LinkedIn first.

## 4. Niche directories (no login)
- All AT niches: firmenabc.at (names!) + firmen.wko.at.
- Any DE local biz: **dasoertliche.de** `/Themen/{Branche}/{Stadt}.html` — owner names often in listing titles ("Kirchhoff Michael Dr. Rechtsanwalt"). Fetchable.
- Steuerberater: `steuerberaterverzeichnis.berufs-org.de` (all DE tax advisors, free).
- Law: anwalt.de + gelbeseiten.de 403 → snippets; dasoertliche works.
- Consulting/coaching: provenexpert.com profiles show owner names (`provenexpert.com/de-de/{slug}/`). Fetchable.
- E-commerce: trustedshops.de `/shops/{kategorie}/` → then the shop's own /impressum.
- Marketing agencies / IT / recruitment / biotech / construction: search `{niche} {stadt}` → agency site `/impressum` or `/team` (the Impressum route beats directories; wlw.de + sortlist.de are not fetchable).

## 5. Website page paths (fetch in this order)
1. `/impressum` (legally mandatory, ~always exists; also `/imprint`)
2. `/ueber-uns` (also `/unternehmen`, `/about`)
3. `/team` (also `/unser-team`, `/ansprechpartner`)
4. `/kontakt`
5. `/datenschutz` — names the data controller (often the owner) as backup.

## 6. Agent decision tree
1. Website known? → fetch `/impressum`. Name after "Geschäftsführer:" / "Inhaber:" / "Vertreten durch:" = target. Done ~70% of the time.
2. Else — DE: `site:companyhouse.de {name}` + northdata Historie. AT: `site:firmenabc.at {name}` → fetch. CH: `site:moneyhouse.ch {name}` snippets.
3. Confirm current via LinkedIn/XING snippet search with company name.
4. Watch umlaut spellings (Müller/Mueller); strip titles (Dr., Dipl.-Ing., Mag.) from names.
5. e.K./Einzelunternehmen: the company name often contains the owner's name itself.
