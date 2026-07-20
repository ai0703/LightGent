# NORDICS — Sweden/Finland owner-research playbook

**Golden rule:** FI sources are fetch-friendly; SE company-data sites are bot-hostile (403/JS-empty) — for SE, harvest via search-engine snippets (`site:` queries), never raw fetch.

## 1. Registries & free lookups (verified)
- **FI PRH open API** — `avoindata.prh.fi/opendata-ytj-api/v3/companies?name={q}` or `?businessId={id}` — returns JSON: names, Y-tunnus, form (Oy), address, industry, website. No officers, no key needed. Perfect for agents.
- **FI finder.fi** — search `/search?what={q}`; company pages have a **Päättäjät** tab: toimitusjohtaja, hallituksen puheenjohtaja/jäsen + revenue, employees. Fetches clean — best FI source.
- **FI asiakastieto.fi** — `/yritykset/fi/{slug}/{businessid-nodash}/yleiskuva` — CEO + board names free. Fetches clean.
- **SE allabolag.se** — VD, styrelse, org.nr, revenue — but JS-rendered (raw fetch EMPTY). Use `site:allabolag.se {company}` search; titles/snippets carry org.nr + officer summaries. (proff.se redirects into allabolag.)
- **SE ratsit.se / merinfo.se** — VD + board on free tier but 403 bots → `site:ratsit.se {company}` / `site:merinfo.se {company}` snippets only.
- ID formats: FI Y-tunnus `1234567-8`; SE org.nr `556703-7485` (AB = 556/559 prefix). Company footer usually shows it → feed into registry.

## 2. Local search terms
- **SE:** ägare, grundare, medgrundare, VD (=CEO), styrelseordförande, delägare. Legal form **AB**. Recipes: `{company} VD` · `{company} grundare` · `{niche} {stad} ägare`
- **FI:** omistaja, perustaja, toimitusjohtaja (=CEO, "tj"), hallituksen puheenjohtaja, yrittäjä. Forms **Oy/Oyj/Tmi/Ky**. Recipes: `{company} toimitusjohtaja` · `{company} Oy perustaja`
- Small companies: the owner IS usually the VD/toimitusjohtaja; a single-person board ⇒ styrelseledamot = owner.

## 3. LinkedIn (snippet-only)
- `site:se.linkedin.com/in {company} VD OR grundare` / `site:fi.linkedin.com/in {company}` (also try plain linkedin.com/in). Result titles read "Name – VD – Company | LinkedIn" — extract from the title string. Never fetch profiles (authwall).

## 4. Niche keywords & directories
- Marketing: SE reklambyrå/marknadsföringsbyrå · FI mainostoimisto/markkinointitoimisto → finder.fi (`search?what=mainostoimisto helsinki` verified: 1200+ with päättäjät)
- Recruitment: SE rekryteringsföretag/bemanningsföretag · FI rekrytointiyritys/henkilöstöpalvelu
- IT/MSP: SE IT-konsult/IT-drift · FI IT-palvelutalo/ohjelmistotalo
- Consulting: SE konsultbolag · FI konsulttitoimisto
- Biotech: SE `swedenbio.se/medlemscommunity/vara-medlemmar/` (330+, verified) · FI bioteknologia → finder.fi
- Coaching: SE ledarskapscoach/affärscoach · FI valmentaja/business coach (ICF chapters)
- Law: SE advokatbyrå (advokatsamfundet blocks bots) · FI asianajotoimisto (findanattorney.fi = JS, snippet-harvest)
- Accounting: SE redovisningsbyrå · FI **taloushallintoliitto.fi/tilitoimistot/** (verified, 903 firms, free)
- E-commerce: SE e-handel/webbutik · FI verkkokauppa
- Construction: SE byggföretag · FI rakennusliike
- Any SE niche list: `site:allabolag.se {keyword} {stad}` → company names + org.nr from titles.

## 5. Company-site page paths
- **SE:** `/om-oss` · `/om` · `/team` · `/vart-team` · `/medarbetare` · `/kontakt`
- **FI:** `/meista` · `/tietoa-meista` · `/yritys` · `/tiimi` · `/henkilosto` · `/yhteystiedot`
- Look for "Förnamn Efternamn, VD" / "Etunimi Sukunimi, toimitusjohtaja"; footer → org.nr/Y-tunnus.

## 6. Agent workflow (per prospect)
1. **FI:** PRH API (confirm Y-tunnus) → finder.fi or asiakastieto yleiskuva (CEO/board, fetch OK) → site `/yhteystiedot` → LinkedIn snippet confirm.
2. **SE:** `site:allabolag.se {company}` (org.nr + officer snippets) → `{company} VD` snippet search across ratsit/merinfo → site `/om-oss` → LinkedIn snippet confirm.
3. "Skyddad identitet" (protected identity) entries = dead end — pick another officer. Two independent sources per name before use.
