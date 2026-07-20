# US — United States owner-research playbook

## 0. Fastest path (try in order)
1. **Company website**: `/about`, `/about-us`, `/team`, `/our-team`, `/leadership`, `/our-story`, `/staff` — also footer, Contact page, blog author bylines.
2. **BBB profile** — usually names the owner outright (§1).
3. **SERP LinkedIn recipe** (§5) — the name arrives in the search-result title, no login needed.
4. **State registry / license lookup** (§2, §6).

## 1. BBB — best single free source (verified fetchable)
- Discover via search: `site:bbb.org profile COMPANY CITY` (add `owner`).
- URL pattern: `bbb.org/us/{st}/{city}/profile/{category}/{slug}-{id}`.
- Payload: **"Business Management: <Name>, Owner/President"**, Principal Contacts, years in business. Excellent for local services/agencies. Franchises → names the local franchisee.

## 2. State registries (Secretary of State)
- **Registered agent ≠ owner.** Agent names like "Northwest Registered Agent", "United States Corporation Agents", "CT Corp" are filing services — NEVER report them. Officer / authorized person / managing member fields are the real signal.
- **FL Sunbiz (best)**: officer names + titles on detail pages. Direct fetch 403s → use `site:search.sunbiz.org COMPANY` (fully indexed).
- **DE**: name + agent only, officers paywalled — low value. **CA/NY**: JS apps, not fetchable — use BBB/licenses/LinkedIn. **TX**: paid/POST-only — skip direct.
- **OpenCorporates**: direct fetch = captcha; harvest officer names from SERP snippets: `site:opencorporates.com COMPANY`.

## 3. SEC EDGAR (companies that raised money or are public)
- Direct fetch 403s (needs declared UA). Route via SERP: `site:sec.gov COMPANY` — **Form D** names executive officers of even small private companies that raised capital.

## 4. General search recipes
- `COMPANY founder OR owner OR CEO OR president OR principal`
- `"founded by" COMPANY` / `founder of COMPANY` — press releases + local news (openings, anniversaries, "40 under 40", Inc 5000, Business Journal lists).
- Titles: small **LLC** → Owner / Managing Member / Principal; **Inc** → President / CEO. Search both sets.
- Eponymous names ("Smith & Sons") — the surname is a strong lead; confirm via BBB/LinkedIn.

## 5. LinkedIn (SERP-only — profiles are login-walled, never fetch)
- `site:linkedin.com/in COMPANY owner OR founder OR CEO` — result titles read literally "Joe Archer - Business Owner at Archer Plumbing | LinkedIn". The title/snippet IS the answer.
- Disambiguate same-name companies by adding city/state.

## 6. Niche directories & license lookups (verified)
- **Construction**: state contractor licenses name the qualifying individual (≈ owner). FL: `myfloridalicense.com` is GET-fetchable. CA CSLB is POST-only → use BBB. Others: `site:<board domain> COMPANY`.
- **Law**: state bar detail pages fetch (CA: `apps.calbar.ca.gov/attorney/Licensee/Detail/{barNumber}`); Avvo loads without login. Firm named after partners → partners are the owners.
- **Accounting**: firm name usually = partner surnames; SERP + state board.
- **Marketing agencies**: Clutch profiles fetch (`clutch.co/profile/{slug}`) — reviews name executives; pair with `/about`.
- **Recruitment**: ASA directory fetches (company info only) — chain to the LinkedIn recipe.
- **IT/MSP**: no useful public directory — BBB + LinkedIn recipe + site `/team`.
- **Biotech**: EDGAR Form D via SERP + `/leadership` + press releases.
- **Coaching/consulting/e-com**: personal-brand heavy — the owner usually IS the site; `/about`, LinkedIn recipe, then `COMPANY podcast OR interview` (guest spots always name the founder). WHOIS is redacted — don't bother.

## 7. Verification rules
- **Two-source rule**: registry/license/BBB must agree with LinkedIn or the site.
- Never report a registered agent, attorney, or filing service as the owner.
- Prefer sources <~2 years old (businesses sell); treat an old annual-report "President" as provisional.
- A 403 usually means bot-blocking, not paywall: retry via `site:` search before giving up.
