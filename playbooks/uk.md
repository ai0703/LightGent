# UK — United Kingdom owner-research playbook

## 1. Companies House — the gold source (verified fetchable, no auth)
Base: `https://find-and-update.company-information.service.gov.uk`
- Company search: `/search/companies?q={name}` → results link to `/company/{8-digit-number}`
- Company overview: `/company/{number}` — status, registered address, SIC codes
- **Officers/directors: `/company/{number}/officers`** — names, roles, appointment dates
- **PSC (the actual owners): `/company/{number}/persons-with-significant-control`** — names + % share bands. If the PSC is another Ltd (holding co), recurse into that company's PSC page to reach the human.
- Officer reverse-lookup: `/search/officers?q={person name}` → all companies a person directs
- Niche list building: `/advanced-search/get-results?sicCodes={code}&status=active&registeredOfficeAddress={town}&page=N`
- Company numbers: England/Wales 8 digits; Scotland `SC`, N. Ireland `NI`, LLPs `OC`.

Useful SIC codes: 73110 advertising/marketing · 70229 mgmt consultancy · 78109/78200 recruitment · 62020/62090 IT/MSP · 69201 accountants · 69102 solicitors · 47910 e-commerce · 41202+43xxx construction · 72110/72190 biotech R&D · 85590 coaching/training.

## 2. Search recipes
- `{company} companies house` → jump straight to the CH page
- `{company} founder OR managing director OR owner site:.co.uk`
- `site:uk.linkedin.com/in founder {company}`
- `{first last} {company} director` — confirm officer match via CH DOB month + town
- Entity conventions: **Ltd** → PSC names the owner directly. **LLP** → officers page lists "LLP Designated Members" = the partners/owners. **Sole traders** → NOT on Companies House; use directories, LinkedIn, site/Facebook. **PLC** → target CEO/MD from officers.
- Titles: small UK firms use "Managing Director"/MD more than CEO; "Owner"/"Director" for micro; "Founder" for agencies.

## 3. LinkedIn
- Prefer `site:uk.linkedin.com/in` via web_search and read title/snippet ("Name - Title at Company"); fetch profile only as confirmation (authwall is intermittent).

## 4. Niche directories (fetch status verified)
- ✅ Marketing/digital: `bima.co.uk/members/` · Consulting: `mca.org.uk/members` · Biotech: `bioindustry.org/membership/directory.html` · Recruitment: `rec.uk.com/member-directory` · Law: `legal500.com/c/{city}/`
- ❌ 403 direct (use `site:` search + snippets instead): solicitors.lawsociety.org.uk, find.icaew.com, cloudtango.net, fmb.org.uk, lifecoach-directory.org.uk, trustpilot, clutch.
- Rule: when a directory 403s, `site:` search it and read the snippets — the search index is not blocked.

## 5. Company-site page paths (try in order)
`/about` · `/about-us` · `/team` · `/our-team` · `/meet-the-team` · `/our-people` · `/our-story` · `/contact`
Also scan the homepage footer + T&Cs/privacy pages for "© {Legal Name} Ltd · Company No. {number}" — UK sites legally must display it; that number is the exact CH lookup key.

## 6. Recommended flow
1. Legal name + company number from site footer/T&Cs (or `/search/companies?q=`).
2. Fetch `/company/{no}/persons-with-significant-control` → human owner (recurse holding cos).
3. Fetch `/company/{no}/officers` → MD/founder is usually the earliest-appointed active director.
4. Cross-check via `site:uk.linkedin.com/in` search + the site's `/team` page.
5. Sole trader / no CH record → directories + LinkedIn + Facebook "About".
