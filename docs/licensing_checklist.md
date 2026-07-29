# Source Licensing & Rights Checklist

**Current status: this corpus is private research use, shared at most with one
teammate — not published or redistributed.** `scripts/export_to_hf.py` therefore does
NOT gate on `license_status` by default (pass `--clear-only` to restrict to `clear`
sources, for whenever a public-release subset is actually being prepared). This
document stays maintained anyway, for two reasons: (1) `license_status` is recorded on
every document regardless, so a future public subset needs no re-collection; (2) one
entry below (Palestine Remembered) reflects a hard technical/ethical line that holds
*regardless* of public-vs-private use, not a copyright judgment call — see that entry.

Per-source assessment of redistribution rights. This is a rights checklist, not a
technical-feasibility one — a source can be technically easy to scrape and still be
`blocked` here, and vice versa. Status values match `LicenseStatus` in
`src/ingestion/schemas.py`:

- **`clear`** — redistribution rights confirmed.
- **`needs_review`** — real, unresolved uncertainty (legal and/or ethical). Collected
  for private research use; would need resolving before any public export.
- **`blocked`** — do not collect, for a reason that isn't about publication scope at
  all (see Palestine Remembered below).

Evidence for the "needs_review"/"blocked" sources below (robots.txt fetches) was
gathered by reading each site's own published crawling policy — a standard,
non-invasive diligence step, not scraping of any actual content.

## Wikipedia (Arabic + English) — `clear`

CC BY-SA 4.0, explicitly licensed for reuse and redistribution with attribution and
share-alike. This is the only source currently implemented (`WikipediaCollector`) and
exported. No further action needed.

## Semantic Scholar (S2AG API) — `clear`, scoped to title + abstract only

The Semantic Scholar Academic Graph API's Terms of Use permit programmatic use of the
metadata it serves (titles, abstracts, authors, venue, citation counts) for research
and product purposes — this is the intended use of a public API that returns exactly
this data. **Full-text PDF redistribution is explicitly out of scope and not
implemented** — that depends on each paper's individual open-access status, which the
collector does not attempt to resolve. `SemanticScholarCollector` only ever requests
and stores `title`/`abstract`/metadata fields.

## GDELT — `needs_review`

GDELT's own processed event data (entities, tone, coordinates, source URLs) is
released by the GDELT Project for free reuse, including commercial and academic use,
with attribution requested. However, GDELT is an *index* — it does not serve article
text. Building a document from a GDELT record means fetching the full article from
whatever outlet GDELT points to, which reintroduces a **separate, per-outlet rights
question** GDELT's own license does not resolve. Not implemented in this track: a
GDELT collector that stored only GDELT's own metadata (no scraped article text) could
plausibly be `clear`, but that is a materially different, smaller deliverable than
"collect GDELT-indexed Palestinian news articles," which is what the original roadmap
entry implied. Revisit as its own scoped task.

## WAFA (wafa.ps) — `needs_review`

**Evidence:** `robots.txt` explicitly allows crawling (`User-agent: * / Allow: /`),
which is a real, positive signal — but it governs *crawling/indexing*, not
*redistribution*. WAFA is the official Palestinian news agency; its articles are
copyrighted news content, and no open-license grant was found alongside the permissive
`robots.txt`. Crawl-permission ≠ republish-permission. **Recommendation:** contact WAFA
directly for a data-sharing agreement or a documented reuse policy before collecting
at scale; do not treat the open `robots.txt` alone as clearance.

## Nakba Archive (nakba-archive.org) — `needs_review` (legal *and* ethical)

**Evidence:** `robots.txt` fetch redirected (301) without resolving a usable policy —
inconclusive on the technical question. More importantly, this archive holds **oral
testimony from Nakba survivors** — first-person trauma narratives. Even where
technically crawlable and even where copyright might arguably permit reuse, collecting
and redistributing personal survivor testimony without verified informed consent for
this specific use is an ethical question a `robots.txt` check cannot answer.
**Recommendation:** do not collect without direct partnership with the archive
confirming both rights *and* consent; this is not a "check a license field" problem.

## Palestine Remembered (palestineremembered.com) — `blocked`

**Evidence:** the `robots.txt` request did not return a robots policy at all — it hit
an active Cloudflare bot-detection JavaScript challenge (`"Enable JavaScript and
cookies to continue"`, a managed challenge token, `noindex,nofollow`). This is an
explicit, current, technical signal that the site does not want automated access, full
stop. **This project will not attempt to bypass it** — circumventing a bot-detection
challenge is exactly the kind of detection evasion this project is committed to never
doing, independent of what the underlying content licensing might otherwise allow. If
this archive's content is wanted, the only acceptable path is contacting the site
directly for API access or a data-sharing agreement.

## Summary

| Source | Status | Blocking factor |
|---|---|---|
| Wikipedia (ar/en) | `clear` | — |
| Semantic Scholar (title+abstract) | `clear` | — |
| GDELT | `needs_review` | Article text is a separate per-outlet rights question |
| WAFA | `needs_review` | No confirmed redistribution license, despite open `robots.txt` |
| Nakba Archive | `needs_review` | Survivor-testimony consent, not just copyright |
| Palestine Remembered | `blocked` | Active bot-detection; will not be circumvented |
