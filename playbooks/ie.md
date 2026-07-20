# IE — Ireland owner-research playbook

**Reality check:** Irish director names are NOT free online — CRO, OpenCorporates, Vision-Net, goldenpages, LinkedIn all block bots. Free owner names come from: the company's own site, LinkedIn search snippets, Irish business press, and the directories below. Registries confirm entity type/status only.

## 1. Registries & free lookups
- **SoloCheck (CRO mirror)** — `solocheck.ie/Irish-Company/{Name-Slug}-{cro_number}` — fetchable. Company type (LTD/DAC), status, address, inc. date, COUNT of directors (names paywalled). Business-name pages flag sole traders.
- CRO official (core.cro.ie) — 403 to bots. Find the CRO number instead: Irish sites must print "Registered in Ireland, No. NNNNNN" in the footer/terms → `solocheck.ie` URL or `site:solocheck.ie NNNNNN`.
- **Charities**: full register CSV with **trustee names** free via data.gov.ie ("register-of-charities-in-ireland") — owner names for nonprofits.

## 2. Search recipes for owners
- `{company} founder OR co-founder OR "founded by" Ireland`
- `{company} managing director` — **MD is the standard Irish SME owner title**, not CEO. Also: Principal (professional firms), Proprietor.
- `{company} siliconrepublic OR businessplus OR thinkbusiness OR independent.ie` — Irish business press names founders.
- `{company} award winner` — SFA Awards, Deloitte Best Managed, National Startup Awards pages name owners.
- **Naming convention:** Irish SMEs are very often surname-named ("Murphy Plant Hire", "O'Brien & Co") → owner surname is in the company name; search `{surname} {company} managing director`.
- Suffixes: Ltd/DAC = company (directors ≈ owners at SME scale); `t/a` = trading name; no suffix = likely sole trader (registrant IS the owner).

## 3. LinkedIn (snippet-only, never fetch)
- `site:ie.linkedin.com/in {company}` — snippet shows "Name — Title — Company".
- `site:ie.linkedin.com/in {company} founder OR owner OR managing director OR principal`
- Fallback: `{person name} {company} linkedin` → snippet confirms current role.

## 4. Niche directories (fetch-verified unless noted)
- Marketing agencies: **iapi.ie/members** → `iapi.ie/members/{slug}` (~75 agencies, profile pages). ✅
- Biotech: **biopharmguy.com/links/country-ireland-all-location.php** (200+ companies, static HTML). ✅
- Law: **lawsociety.ie/find-a-solicitor/Solicitor-Firm-Search/** (no login; small-firm solicitors ≈ principals). ✅
- Accountants: **portal.cpaireland.ie/firmdirectory.aspx** (no login, by county). ✅
- Construction: **voluntaryconstructionregister.ie** (search by name/county/category). ✅
- Coaching: icfireland.com/find-a-coach (coaches self-list = person IS the owner).
- Recruitment: erfireland.com blocks bots → `site:erfireland.com` snippets.
- IT/MSP + consulting: no fetchable directory → `"managed IT services" {county} site:.ie` then team pages; `site:ie.linkedin.com/in managing director "IT services"`.
- Any local business: goldenpages.ie 403s → `site:goldenpages.ie {category} {town}` snippets.

## 5. Company-site page paths (try in order)
`/about` · `/about-us` · `/team` · `/our-team` · `/meet-the-team` · `/our-people` · `/who-we-are` · `/leadership` · `/our-story` · `/contact`
- Terms/privacy/footer: registered name + CRO number.
- Irish sites often bury the owner in `/our-story` ("founded by X in 19xx") rather than a team grid.

## 6. Decision flow
1. Company site `/about|/team|/our-story` → owner name + title.
2. `site:ie.linkedin.com/in {company}` snippets → MD/founder.
3. Irish press recipes (§2) → founder named in article snippets.
4. Footer CRO number → SoloCheck → confirm LTD vs sole trader, status (legitimacy check, not names).
5. Nonprofit → charities CSV trustees. Professional firm → Law Society / CPA directory.
