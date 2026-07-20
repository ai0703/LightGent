# NL — Netherlands owner-research playbook

## 0. Core logic: legal form tells you where the owner is
- **eenmanszaak** (sole prop) / **VOF** (partnership) / **maatschap** → the owner(s) ARE the registered persons; company often named after them ("Bakkerij Jansen" → owner Jansen). Small trades/coaches/admin offices are usually this.
- **BV** (besloten vennootschap) → look for **bestuurder** / **DGA** (directeur-grootaandeelhouder). Bestuurder is often a personal holding ("J. de Vries Beheer B.V.") — the person's name is inside the holding's name.
- Terms that surface owners in search + on pages: eigenaar, oprichter, mede-oprichter, directeur, DGA, bestuurder, vennoot, maat, partner, algemeen directeur.

## 1. Registry lookups (verified fetchable, no login)
- **Drimble** — best free mirror. Page: `drimble.nl/bedrijf/{plaats}/{id}/{slug}.html` (id is internal, NOT the KVK nr → don't construct; find via search `site:drimble.nl/bedrijf <company>`). Exposes: KVK nr, **rechtsvorm**, address, SBI, and often **bestuurder / procuratiehouder names** (incl. holding-BV bestuurders — mine the holding name for the person).
- **Oozo** — find via `site:oozo.nl/bedrijven <company>`. Exposes: KVK nr, rechtsvorm, SBI, address, employee bracket, start date. No person names — use for rechtsvorm + KVK nr, then pivot.
- **Bedrijvenregister** — `bedrijvenregister.nl/{plaats}/{company-slug}`. KVK nr, rechtsvorm, handelsnamen, SBI (owner name is paywalled — still useful for rechtsvorm).
- **KVK official** (kvk.nl) — JS app, unfetchable; UBO closed. **openkvk.nl → 403.** Skip both; use Drimble.
- Recipe: search `<company> KVK eigenaar` usually lands on Drimble/Oozo directly.

## 2. Search recipes (owner discovery)
1. `<company> eigenaar OR oprichter OR directeur` (Dutch web is small; this alone solves most SMBs)
2. `site:drimble.nl <company>` → rechtsvorm + bestuurder/holding names
3. `<company> DGA OR mede-oprichter` → interviews, trade press (Emerce, MT/Sprout, Quote, regional papers = goldmines for founder names)
4. eenmanszaak/VOF + surname-like company name → assume surname = owner, confirm on site/LinkedIn
5. Holding pattern: bestuurder = "X Beheer B.V." / "X Holding B.V." → search `X <company>`

## 3. LinkedIn (never fetch directly — authwall)
- Profiles at `nl.linkedin.com/in/{slug}`. Use search, not fetch: `site:linkedin.com/in <company> eigenaar OR oprichter OR founder OR directeur` and `site:nl.linkedin.com/in <company>`.
- The search snippet (name — title — company) is usually enough; cross-check role recency with the company site.

## 4. Niche directories (verified fetchable, no login)
- Digital/marketing bureaus: `dutchdigitalagencies.com/leden/` (147 agencies, profile pages)
- Recruitment/staffing: `abu.nl/over-abu/ledenregister/` (name, KVK, website; paginate /page/2/) + NBBU leden
- IT/digital: `nldigital.nl/onze-leden/` → `nldigital.nl/leden/{slug}/`
- Biotech: `hollandbio.nl/leden/` (292 members)
- Coaching: `nobco.nl/vind-een-coach/` (filter theme/region)
- Law: `zoekeenadvocaat.advocatenorde.nl` (partly JS — fall back to site: search)
- Accounting/admin: `portaal.noab.nl/kantoren/` (firm names often = owner surname)
- E-commerce: `thuiswinkel.org/leden/{slug}/`
- Construction: `bouwendnederland.nl/vereniging/lidbedrijven`
- BLOCKED (403, do not fetch): sortlist.nl, openkvk.nl

## 5. Company-website pages to fetch (in this order)
`/over-ons` · `/team` · `/wie-zijn-wij` · `/over` · `/medewerkers` · `/ons-team` · `/contact` · `/about`
- Many NL B2B sites are bilingual — also try `/about-us`.
- Grab the name next to: eigenaar, oprichter, directeur, "opgericht door", "het gezicht achter", founder photos with titles.
- Footer often shows the KVK nr → feeds step 1 for rechtsvorm confirmation.

## 6. Decision flow (per prospect)
1. Fetch site `/over-ons` + `/team` → name found? done (eigenaar/oprichter/directeur beats "manager").
2. Else `site:drimble.nl <company>` → rechtsvorm; eenmanszaak/VOF → registered person = owner; BV → bestuurder/holding name.
3. Else LinkedIn site: search (step 3).
4. Else niche directory (step 4) + trade-press search.
5. Confidence: registry-confirmed > own-website > LinkedIn snippet > directory. Two agreeing sources = ship it.
