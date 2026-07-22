# WRDS I/B/E/S Quarterly-EPS Pilot

**Purpose:** source-contract and ownership audit only

**Maximum licensed observations:** 25

**Performance metrics:** prohibited

**Git status of outputs:** prohibited

## Why the contract uses unadjusted EPS plus CRSP factors

WRDS documents two separate hazards. Adjusted I/B/E/S history can suffer
material rounding error, while a naive merge of unadjusted estimates and
actuals can compare different per-share bases around stock splits. WRDS calls
its CRSP-based Method 3 the relatively most reliable approach: link the security
to PERMNO, obtain the CRSP cumulative share-adjustment factor at the estimate
and report dates, and move the reported actual onto the estimate-date share
basis.

This pilot implements the source side of that method for monthly summary
statistics:

```text
actual_on_consensus_basis =
    actual_unadjusted
    × cfacshr_at_consensus_statistical_period
    ÷ cfacshr_at_report_date
```

It intentionally leaves that calculation to the manual validation exercise.
The pilot uses current CRSP daily table `crsp.dsf_v2`; the legacy `crsp.dsf`
available under this entitlement ends on December 31, 2024. Because the current
table ends on December 31, 2025, this source audit is restricted to 2025
announcements rather than extrapolating a stale share factor.

Official references:

- [I/B/E/S on WRDS — Introduction and Research Guide](https://wrds-www.wharton.upenn.edu/pages/grid-items/ibes-wrds-101-introduction-and-research-guide/)
- [A Note on IBES Unadjusted Data](https://wrds-www.wharton.upenn.edu/documents/5/A_Note_on_IBES_Unadjusted_Data_pdf.pdf)

## Safety and entitlement rules

- Run only under the user's authorized WRDS account.
- Keep the password in the local PostgreSQL password file; never pass it on the
  command line, print it, or commit it.
- Keep every licensed row beneath `data/private/`, which is ignored by Git.
- The hard cap in code is 25 observations. A command-line option cannot raise
  it.
- Never print the query result. Console output contains counts and paths only.
- Do not calculate returns, predictive correlations, IC, Sharpe, or portfolios
  during this audit.
- Do not assign a time zone to I/B/E/S `anntims` or `acttims` until the provider
  convention is documented.

## Dry run

The default command does not connect to WRDS. It validates the window and cap,
then prints a query hash and safe execution plan:

```powershell
uv run --locked --extra wrds python tools/run_wrds_ibes_eps_pilot.py
```

## Authorized live run

Set the username for the process and the explicit live gate. PostgreSQL's
password file supplies the password without an interactive prompt.

```powershell
$env:WRDS_USERNAME = "your-wrds-username"
$env:SENTIMENT_LAB_ENABLE_LIVE_WRDS_IBES = "1"
uv run --locked --extra wrds python tools/run_wrds_ibes_eps_pilot.py --live --max-events 25
Remove-Item Env:WRDS_USERNAME
Remove-Item Env:SENTIMENT_LAB_ENABLE_LIVE_WRDS_IBES
```

The command writes:

- `events.parquet`: restricted source rows and split factors;
- `receipt.json`: safe counts, contract labels, and hashes; and
- `manual_validation_observation.json`: one restricted observation plus the
  five-step ownership exercise.

None of these files may be committed or shared publicly.

## Manual ownership gate

Open `manual_validation_observation.json` locally and, without AI calculating
the answer:

1. verify the consensus statistical period predates the announcement date;
2. place the actual on the consensus-date share basis using the two CRSP
   factors;
3. subtract the consensus mean to obtain raw EPS surprise;
4. explain why skipping the split-basis correction can produce a false
   surprise; and
5. identify the raw announcement and activation times that remain unsafe to
   convert to UTC.

Record the derivation and explanation in the private study notebook. The
pipeline does not scale until this observation passes review.
