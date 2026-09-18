# 42. Building a Record-Keeping System with an AI Agent — One-Page Case

> **Scope.** This is a methodology note, not a research result. The ECG denoising
> study itself is in `91_report.md`. Full Korean narrative: `40_ai_collaboration_case.md`.
> Verbatim transcripts: `41_ai_collaboration_transcript.md`.
> Every quote below is taken **verbatim** from the session log and translated;
> the Korean original is in the appendix.

## Context

A ~4-week solo research project (ECG denoising, classical DSP vs. deep learning)
run almost entirely through an AI coding agent. Over that period the repository
accumulated a record-keeping system — 4 record types, a convention document, an
automated conformance checker, and a 3-tier rule hierarchy — that was **not
designed up front.** It grew through **four interventions**, each triggered by the
user identifying a failure *in the process*, not in the output.

## The four interventions

| # | Problem the user named | What was built | Commit |
|---:|---|---|---|
| 1 | "Context gets compacted; reasoning that isn't written down is lost" | 3 record types (Finding / Decision / Incident), each with a **discriminating question**, plus a written convention | `f2f752b` |
| 2 | "Mandatory fields will pressure you into **inventing** the missing ones" | Evidence tags (`[measured]` `[log]` `[commit]` `[reconstructed]`), an explicit **"no record" state**, and automated cross-checking of cited numbers against result files | `96b0bc0` |
| 3 | "This feedback process is a different kind of thing — record it separately" | A fourth record type for AI-collaboration lessons, each of which **must end in a reusable rule or it doesn't qualify** | `71502b7` |
| 4 | "We have rules and we check them, **yet the same failures recur**" | 3-tier split: always-loaded placard (60-line cap) / per-task checklists / detailed records | `ecdd78b` |

An additional intervention (`7f58b18`) came from a diagnostic question:
*"Was this record missing because the **procedure** doesn't ask for it, or because
the experiment wasn't finished yet? Those are entirely different problems."*
The answer was **both**, and it produced three checks that detect **absent**
records — the prior checks could only judge records that already existed.

## What made the requests unusually effective

1. **State the goal, delegate the method** — *"…for the goal of 'how can I look back
   in enough detail when I write the report later,' review and propose how the
   records should be organised, **and document that proposal too**."*
2. **Predict the side effects of your own request.** The hallucination warning (#2)
   arrived *while* the agent had already fabricated 14 reconstructed rationales —
   before anyone had noticed. The reply began: *"you're right, and the second point
   is **a problem I just committed**."*
3. **Ask for a cost analysis against yourself** — *"analyse the benefits and costs of
   applying my proposal (e.g. … token consumption increases — is that significant?)"*
4. **Split causal hypotheses and demand adjudication** (the quote above).
5. **Constrain the shape of the solution** — *"so that the record is not skipped even
   when context continuity is low, e.g. after the next experiment or a compaction."*
   This ruled out any fix that relies on memory.
6. **Fix the target level with an analogy.** For #4 the user proposed airline
   documentation — cockpit checklists / field manuals / thick inspection manuals /
   accident reports — *"I'm not asking our work to reach that level, but…"*.
   **That analogy became the architecture**, one tier per document class.
7. **Bring in external review.** The same conversation was handed to a second AI and
   the critique brought back, with the instruction to sort it into *adopt / develop
   further / incorrectly stated*.

## What the agent contributed

- **Counted before arguing.** For #4: *"Before talking about rulebooks I counted how
  often things actually recur"* — 5 repeats of one failure, 3 of another.
- **Turned its own error into a mechanism.** The fabrication in #2 became the
  evidence-tag system, rather than a silent fix.
- **Declined external proposals with reasons.** A proposed 12-field template was
  rejected: *"the pressure to fill fields is exactly what induces the hallucination
  — tripling the fields triples the pressure."*
- **Reported a flaw in its own verification.** While grepping transcripts for
  evidence: *"the verification method is itself broken — the document I just wrote
  is part of the transcript, so this is self-reference."*
- **Found the structural cause.** For #4, the decisive finding was that the rule
  summary written after the 5th repeat was **buried at line 932 of a 975-line
  incident file**. The repeats were not caused by missing rules but by the absence
  of a place that is *always read*.

## Outcome

```
L1  CLAUDE.md                 50 lines, hard cap 60, always loaded
L2  docs/17_checklists.md     7 sections, one per task trigger
L3  findings / decisions / incidents / AI-review   (108 entries)
    docs/19_record_keeping.md  the convention itself
Enforcement  scripts/check_records.py, run inside the test suite
```

The load-bearing part is the last line. Conventions are kept by people; **whether
they were kept is checked by machine**, so continuity survives session changes and
context compaction — which was the original problem in intervention #1.

**Self-test.** Committing this very case study tripped the repository's own
integrity checks three times. All three were resolved by *recording a reason or an
explicit marker* — none by disabling a check. Details in `40_ai_collaboration_case.md` §6.

## Limitations

One project, one user, ~4 weeks. The time spent on record-keeping was never
measured, so "it paid off" is a qualitative judgement. There is no counterfactual.
What is concrete is what the checks **actually caught**: a finding that never
reached the report, and three design choices that had no decision record.
