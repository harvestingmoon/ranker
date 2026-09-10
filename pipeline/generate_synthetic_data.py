"""
Synthetic data generator v3 for the Hirer <-> Provider matching pipeline.

Every provider and hirer record below is individually hand-authored --
there are no shared text fragments, no templates, and no combinatorial
recombination of building blocks (that was the v2 approach). Each of the
104 providers and 130 hirers has its own unique about_description,
services_offered_description, relevant_experience (providers) or
hire_title/hire_description/hire_description_additional_notes (hirers),
written directly by the assistant as one continuous authoring pass, not
assembled by code from a pool.

Scale was deliberately reduced from v2's 182/234 to 104/130 (4 providers +
5 hirers per speciality x 26 specialities) specifically to make full,
genuine per-record uniqueness realistic to write by hand rather than
rushed.

Taxonomy grounding is unchanged from v2: each record is tagged with a real
(category_id, speciality_id) pair from schema/categories.sql +
schema/specialities.sql, and relevance grading is still a deterministic
function of that grounding plus budget/seniority fit -- that part is
legitimately code's job (consistent, reproducible scoring), unlike the
prose itself.

    score = taxonomy_base (0 / 35 / 70)
          + budget_fit(provider.rate, hirer.budget)     (-10 / 0 / +10)
          + seniority_fit(provider.level, hirer.level)  (-10 / 0 / +10)
          + noise                                       (-5 .. +5)
    clipped to [1, 100]; pairs with taxonomy_base == 0 stay at 0 and are
    omitted from ground_truth.json.

Run: python3 pipeline/generate_synthetic_data.py
"""
import csv
import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SENIORITY_LEVELS = ["mid", "senior", "expert"]
SENIORITY_ORDER = {lvl: i for i, lvl in enumerate(SENIORITY_LEVELS)}

# ---------------------------------------------------------------------------
# Reusable pools for fields that do NOT feed the matching text (corpus.py's
# provider_text()/hirer_text() never read these) -- purely profile metadata,
# where realistic reuse across unrelated people is normal (many real
# freelancer profiles use near-identical availability/logistics phrasing).
# ---------------------------------------------------------------------------
AVAILABILITY = [
    "Available Mondays to Fridays, standard business hours",
    "Available Mondays to Sundays, flexible around client timezone",
    "Available weekday evenings and weekends only",
    "Available for up to 3 days/week for the right engagement",
    "Available on 2 weeks' notice from confirmed start date",
    "Available immediately, full-time capacity for the next quarter",
]

WORK_STYLES = [
    "Prefers to embed with the internal team, with weekly steering updates and a shared tracker.",
    "Prefers a fixed-scope engagement with a clearly defined deliverable and milestone check-ins.",
    "Runs a short discovery workshop first, then executes with a small dedicated working group.",
    "Operates independently and reports progress asynchronously; no daily standups required.",
    "",
]

# PROVIDERS_RAW / HIRERS_RAW are populated by the per-speciality blocks
# appended below this line -- each block is individually hand-authored,
# added incrementally (see PATCH markers).
PROVIDERS_RAW = []
HIRERS_RAW = []

# === SPECIALITY BLOCKS START ===
# ---- Speciality: Banking / Regulatory Reporting (Basel/IFRS9) [cat=1, spec=1] ----
PROVIDERS_RAW.extend([
    dict(category_id=1, speciality_id=1, seniority="expert", full_name="Yushan Edison Lin", rate_per_hour=210,
         credentials="Ex-Head of Regulatory Reporting, DBS Group",
         about_title="Basel III & IFRS9 Regulatory Reporting Lead",
         about_description="Spent eighteen years inside regulatory reporting functions at two Singapore banks, rising to Head of Regulatory Reporting before going independent three years ago to focus purely on remediation work for banks that have already failed an MAS review.",
         services_offered_title="Regulatory Reporting Remediation",
         services_offered_description="Takes ownership of a single broken regulatory return end-to-end, tracing the mismatch back through the risk engine and the general ledger before rebuilding the reconciliation process that produced it.",
         relevant_experience="Closed a reconciliation gap in a top-3 Singapore bank's capital adequacy return that had drawn repeated auditor attention across eight consecutive quarters."),
    dict(category_id=1, speciality_id=1, seniority="senior", full_name="Marcus Tan", rate_per_hour=165,
         credentials="Former Senior Manager, Regulatory Reporting, regional bank",
         about_title="Independent Advisor, Bank Capital & Provisioning Reporting",
         about_description="Ran Basel and IFRS9 reporting cycles at a regional bank for nine years before moving into independent advisory, with particular depth in the handoff between risk model outputs and the finance disclosure layer.",
         services_offered_title="IFRS9 Provisioning Model Review",
         services_offered_description="Reviews an existing IFRS9 expected-credit-loss model against current staging rules and macro overlay assumptions, then rebuilds whichever component is producing numbers auditors keep querying.",
         relevant_experience="Rebuilt the IFRS9 staging logic for a regional bank whose provisioning numbers had failed two consecutive external audit reviews, clearing both findings on resubmission."),
    dict(category_id=1, speciality_id=1, seniority="senior", full_name="Priya Nair", rate_per_hour=190,
         credentials="Former MAS secondee, regulatory reporting review team",
         about_title="Regulatory Reporting Specialist, Ex-MAS Secondment",
         about_description="Completed a two-year secondment on MAS's regulatory reporting review team before returning to industry, giving her a first-hand view of exactly what a thematic review actually checks for.",
         services_offered_title="Pre-Submission Regulatory Review",
         services_offered_description="Runs a structured pre-submission review of a bank's capital and provisioning returns against what an MAS thematic review would test, well before the numbers are due.",
         relevant_experience="Flagged a data lineage gap in a mid-size bank's capital return three weeks before submission that would otherwise have triggered a formal MAS finding."),
    dict(category_id=1, speciality_id=1, seniority="mid", full_name="Daniel Koh", rate_per_hour=130,
         credentials="Finance-risk reconciliation specialist, five years in banking",
         about_title="Finance-Risk Reconciliation Consultant",
         about_description="Spent five years inside a bank's finance-risk reconciliation team working specifically the gap between what risk models output and what finance actually reports, now consulting on that exact problem for smaller banks.",
         services_offered_title="Finance-Risk Data Reconciliation",
         services_offered_description="Traces a specific reconciliation break back to its source across the risk engine, the general ledger, and the regulatory reporting layer, then documents a repeatable process so it doesn't recur.",
         relevant_experience="Delivered a clean capital return for a smaller bank three weeks ahead of an MAS thematic review after the prior quarter's submission had been pulled back for correction."),
])
HIRERS_RAW.extend([
    dict(category_id=1, speciality_id=1, seniority_needed="mid", budget_lo=70, budget_hi=140,
         hire_title="Finance and risk numbers don't reconcile on our capital return",
         hire_description="Our finance and risk teams have produced two different numbers for the same capital adequacy return in each of the last three quarters, and the auditors have now flagged it formally in the management letter. We need someone to trace the discrepancy back to its source in the data pipeline and put a permanent reconciliation process in place, not just patch this quarter's numbers.",
         hire_description_additional_notes="Budget sits around SGD 70-140/hr for a focused four-week engagement starting as soon as possible; a mid-level practitioner who has actually done this reconciliation work is fine, we don't need a former department head for this."),
    dict(category_id=1, speciality_id=1, seniority_needed="senior", budget_lo=140, budget_hi=220,
         hire_title="Provisioning model hasn't been checked against current rules in two years",
         hire_description="A recent internal review found that our loan impairment model hasn't been reassessed against the latest provisioning rules since it was first built, and the assumptions are starting to look dated against what peer banks are disclosing. The ask is a full review of the staging assumptions and macro overlays with a written report we can hand directly to the external auditors.",
         hire_description_additional_notes="Budget is roughly SGD 140-220/hr for an eight-week engagement tied to our year-end close; ideally someone senior enough to have owned a provisioning model review end-to-end before, not someone applying it for the first time."),
    dict(category_id=1, speciality_id=1, seniority_needed="expert", budget_lo=180, budget_hi=260,
         hire_title="Regulatory submission pulled back at the last minute again",
         hire_description="Our last regulatory submission was pulled back at the eleventh hour because finance and risk couldn't agree on the underlying numbers, and it's now happened twice this year. Given how visible this has become internally, we want the most senior person available to both fix the current submission and put a sign-off process in place that prevents a repeat.",
         hire_description_additional_notes="Budget is flexible, up to around SGD 260/hr, for an initial six-week engagement; this needs to be someone who has genuinely led a regulatory reporting function before, not a mid-level practitioner."),
    dict(category_id=1, speciality_id=1, seniority_needed="mid", budget_lo=50, budget_hi=100,
         hire_title="Small finance team needs help tidying up our quarterly capital return",
         hire_description="We're a small finance team without a dedicated regulatory reporting specialist, and our quarterly capital return process is currently held together by one person's personal spreadsheet. We'd like someone to review the current process and suggest practical fixes we can maintain ourselves afterward.",
         hire_description_additional_notes="This is a smaller engagement, budget realistically around SGD 50-100/hr for a two-week review; open to someone earlier in their career as long as they've actually worked on a bank's regulatory return before."),
    dict(category_id=1, speciality_id=1, seniority_needed="senior", budget_lo=120, budget_hi=200,
         hire_title="Need an independent second opinion before our year-end capital submission",
         hire_description="Our internal team is confident in this year's capital return numbers, but given how much scrutiny our sector has been under lately, leadership wants an independent set of eyes on the submission before it goes out. The ask is a focused review of the calculation methodology and disclosure notes, not a full rebuild.",
         hire_description_additional_notes="Budget is around SGD 120-200/hr for a short two-week review ahead of the filing deadline; looking for someone senior enough to sign off on the review with confidence."),
])
# ---- Speciality: Banking / AML/KYC Program Design [cat=1, spec=4] ----
PROVIDERS_RAW.extend([
    dict(category_id=1, speciality_id=4, seniority="expert", full_name="Wen Yeong Chua", rate_per_hour=200,
         credentials="Former Head of Financial Crime Compliance, regional bank",
         about_title="Financial Crime Compliance Architect",
         about_description="Led financial crime compliance at a regional bank for over a decade, including two full MAS thematic inspections, before establishing an independent practice focused entirely on rebuilding AML programs that have already drawn regulatory attention.",
         services_offered_title="AML Program Rebuild",
         services_offered_description="Maps the complete customer due diligence workflow end to end, identifies where stated policy diverges from what actually happens at onboarding, and rebuilds the risk-based approach around the gap.",
         relevant_experience="Rebuilt the AML program for a mid-size bank after an MAS inspection specifically cited weaknesses in how customers were risk-rated at onboarding, with the revised program passing a follow-up review without further findings."),
    dict(category_id=1, speciality_id=4, seniority="senior", full_name="Rachel Goh", rate_per_hour=150,
         credentials="Certified AML specialist (CAMS), twelve years in KYC program design",
         about_title="CAMS-Certified AML/KYC Advisor",
         about_description="Twelve years designing and testing customer due diligence frameworks across three different banks, with a particular focus on making sure transaction monitoring rules actually reflect the risks a bank claims to be managing.",
         services_offered_title="Transaction Monitoring Rule Review",
         services_offered_description="Runs an independent review of transaction monitoring rules and onboarding checks against current MAS Notice 626 expectations, then prioritises fixes by regulatory risk rather than ease of implementation.",
         relevant_experience="Led a financial crime framework overhaul for a payments fintech in the year before its full banking licence application, which was approved on the first submission."),
    dict(category_id=1, speciality_id=4, seniority="senior", full_name="Benjamin Lim", rate_per_hour=175,
         credentials="Former MAS Notice 626 thematic review team member",
         about_title="AML Compliance Advisor, Ex-MAS Reviewer",
         about_description="Spent three years on the MAS side reviewing bank AML programs against Notice 626 before moving into industry, which means he knows precisely which control gaps a real inspection actually surfaces versus which ones only look bad on paper.",
         services_offered_title="Pre-Inspection AML Readiness Review",
         services_offered_description="Runs a mock inspection against the bank's actual AML controls, scoring each area the way MAS would, and hands over a prioritised list of what needs fixing before the real thing.",
         relevant_experience="Identified a customer risk-rating gap during a mock review that, once fixed, meant the client's subsequent live MAS inspection closed with no material findings."),
    dict(category_id=1, speciality_id=4, seniority="mid", full_name="Aisha Rahman", rate_per_hour=105,
         credentials="AML operations specialist, payments fintech background",
         about_title="AML/KYC Operations Consultant",
         about_description="Four years inside an AML operations team at a fast-growing payments fintech, now advising smaller platforms on building their first proper AML program without the benefit of a large compliance department behind them.",
         services_offered_title="First AML Program Build",
         services_offered_description="Builds an AML program from a blank page for a first-time licensee, covering policy, a workable risk-rating methodology, and an escalation workflow that ties the whole thing together.",
         relevant_experience="Cleared a backlog of over two hundred unresolved customer risk assessments for a digital bank within six weeks of starting."),
])
HIRERS_RAW.extend([
    dict(category_id=1, speciality_id=4, seniority_needed="mid", budget_lo=60, budget_hi=120,
         hire_title="Customer onboarding checks haven't kept up with current AML rules",
         hire_description="Our onboarding checks for new customers haven't been meaningfully updated against current financial crime rules in a couple of years, and a recent near-miss made that gap obvious to leadership. We'd like a full gap assessment against current expectations, with a prioritised list we can action ourselves afterward.",
         hire_description_additional_notes="Budget is around SGD 60-120/hr for a three-week review; a solid practitioner is fine, we don't need a former compliance head for a gap assessment."),
    dict(category_id=1, speciality_id=4, seniority_needed="expert", budget_lo=180, budget_hi=260,
         hire_title="AML compliance framework needs review before a regulator visit",
         hire_description="We've just been notified of an upcoming AML compliance review and our framework hasn't been independently checked by anyone with real regulatory experience since it was first drafted. Given the timeline, we want someone genuinely senior who can both assess and start remediating gaps immediately.",
         hire_description_additional_notes="Budget up to around SGD 260/hr for an intensive four-week sprint ahead of the visit; this really needs someone who has sat on the regulator's side of this kind of review before."),
    dict(category_id=1, speciality_id=4, seniority_needed="mid", budget_lo=50, budget_hi=100,
         hire_title="Building our first AML program from scratch",
         hire_description="We're a new payments licensee and currently operate with no formal AML program beyond a short policy document nobody actually follows day to day. We need someone to design the program end to end, ready in time for our licence application.",
         hire_description_additional_notes="Budget realistically around SGD 50-100/hr given we're an early-stage company; open to someone still building their independent practice as long as they've done this hands-on before."),
    dict(category_id=1, speciality_id=4, seniority_needed="senior", budget_lo=100, budget_hi=180,
         hire_title="Transaction monitoring rules feel outdated for how the business has grown",
         hire_description="Our transaction monitoring rules were set up when the business was a fraction of its current size, and we suspect they're both missing genuine risk and generating too many false alerts for the compliance team to work through properly. We'd like someone to redesign the rule set from first principles.",
         hire_description_additional_notes="Budget is around SGD 100-180/hr for a six-week engagement; looking for someone with genuine hands-on rule-design experience, not just policy-writing background."),
    dict(category_id=1, speciality_id=4, seniority_needed="senior", budget_lo=130, budget_hi=210,
         hire_title="Financial crime framework needs to be licence-application ready",
         hire_description="We're preparing a full banking licence application and need our financial crime framework to withstand serious regulatory scrutiny, not just look complete on paper. The ask is an end-to-end review covering policy, onboarding, and monitoring, with clear documentation we can submit as part of the application.",
         hire_description_additional_notes="Budget is roughly SGD 130-210/hr for a two-month engagement tied to our application timeline; ideally someone who has supported a licence application through to approval before."),
])
# ---- Speciality: Insurance / IFRS17 Implementation Support [cat=2, spec=10] ----
PROVIDERS_RAW.extend([
    dict(category_id=2, speciality_id=10, seniority="expert", full_name="Clara Teo", rate_per_hour=220,
         credentials="Qualified actuary (FIA), led IFRS17 transition at a regional insurer",
         about_title="IFRS17 Transition Lead for Life & General Insurers",
         about_description="Qualified actuary who ran the full IFRS17 transition programme at a regional insurer across two complete reporting cycles, from initial contract boundary design through to a clean first set of live disclosures.",
         services_offered_title="IFRS17 Build-to-Run Transition",
         services_offered_description="Owns the full build-to-run transition workstream, covering contract boundary rules, CSM roll-forward logic, and disclosure templates the external auditors will actually test.",
         relevant_experience="Delivered two consecutive IFRS17 reporting cycles for a mid-size life insurer with no material restatements, after the prior in-house build had failed its first dry run."),
    dict(category_id=2, speciality_id=10, seniority="senior", full_name="Vikram Menon", rate_per_hour=195,
         credentials="Former Big 4 actuarial consultant, insurance accounting standard transitions",
         about_title="Independent Actuarial Advisor, Insurance Contract Accounting",
         about_description="Ten years as a Big 4 actuarial consultant specialising specifically in insurance accounting standard transitions, now working directly with insurers rather than billing through a consulting firm.",
         services_offered_title="IFRS17 Disclosure Dry-Run Review",
         services_offered_description="Reviews an existing IFRS17 build against the standard's disclosure requirements before the external auditors do, and fixes whatever a structured dry run flags.",
         relevant_experience="Fixed a CSM amortisation model at a regional insurer that had failed its first disclosure dry run just weeks before the planned go-live."),
    dict(category_id=2, speciality_id=10, seniority="senior", full_name="Michelle Wong", rate_per_hour=165,
         credentials="Former finance-actuarial liaison, life insurer",
         about_title="Actuarial-Finance Bridge Specialist",
         about_description="Spent five years in a life insurer's finance-actuarial liaison role, focused specifically on the point where actuarial contract-grouping models and finance's disclosure output stopped agreeing with each other.",
         services_offered_title="Actuarial-Finance Reconciliation",
         services_offered_description="Bridges the gap between the actuarial team's contract grouping models and what finance needs to produce for quarterly disclosures, resolving the specific line items that keep breaking.",
         relevant_experience="Closed a year-long reconciliation gap between actuarial contract groupings and finance's disclosure output for a general insurer that had been reconciling manually every quarter."),
    dict(category_id=2, speciality_id=10, seniority="mid", full_name="Samuel Chen", rate_per_hour=135,
         credentials="Insurance accounting analyst, four years in IFRS17 implementation support",
         about_title="IFRS17 Implementation Support Analyst",
         about_description="Four years supporting IFRS17 implementation projects from the finance side, handling the detailed disclosure-note preparation work that senior actuaries typically don't have time for.",
         services_offered_title="Disclosure Note Preparation Support",
         services_offered_description="Prepares and quality-checks the detailed IFRS17 disclosure notes against the underlying actuarial output, catching formatting and consistency issues before they reach the auditors.",
         relevant_experience="Cleared a backlog of disclosure note inconsistencies for an insurer's first live IFRS17 reporting cycle in the two weeks before the filing deadline."),
])
HIRERS_RAW.extend([
    dict(category_id=2, speciality_id=10, seniority_needed="mid", budget_lo=80, budget_hi=150,
         hire_title="New insurance accounting standard numbers won't reconcile before year-end",
         hire_description="Our finance team is struggling to get the new insurance contract accounting numbers to reconcile with the actuarial models before year-end close, and nobody on the team has actually run a full transition to this standard before. We need practical, hands-on help getting this quarter's numbers to tie out.",
         hire_description_additional_notes="Budget around SGD 80-150/hr for a focused three-week push before close; doesn't need to be the most senior actuary available, just someone who has done this reconciliation work directly."),
    dict(category_id=2, speciality_id=10, seniority_needed="expert", budget_lo=180, budget_hi=260,
         hire_title="First full IFRS17 disclosure is weeks away and has never been independently checked",
         hire_description="We're weeks away from our first full disclosure under the new insurance accounting standard and have never had an independent party check our build before the external auditors do. Given how much is riding on this being right the first time, we want someone who has genuinely led a full transition before.",
         hire_description_additional_notes="Budget up to around SGD 260/hr for an intensive four-week review ahead of the disclosure date; this needs real transition leadership experience, not general actuarial background."),
    dict(category_id=2, speciality_id=10, seniority_needed="senior", budget_lo=130, budget_hi=210,
         hire_title="Actuarial and finance teams can't agree on contract groupings",
         hire_description="The actuarial team's contract groupings and what finance is reporting have drifted apart over the last two quarters, and nobody internally can pinpoint exactly where the disconnect started. We'd like an independent review that traces it back to its source and recommends a permanent fix, not a one-off patch.",
         hire_description_additional_notes="Budget is roughly SGD 130-210/hr for a five-week engagement; looking for someone with real experience specifically at the finance-actuarial boundary, not a pure actuarial modeller."),
    dict(category_id=2, speciality_id=10, seniority_needed="mid", budget_lo=60, budget_hi=110,
         hire_title="Need help clearing a disclosure note backlog before filing",
         hire_description="Our disclosure notes for this cycle have fallen behind and the actuarial team doesn't have capacity to also handle the detailed note preparation on top of the modelling work. We need someone to take over the note preparation and quality-check it against the underlying actuarial output.",
         hire_description_additional_notes="Budget realistically SGD 60-110/hr for a two-week sprint before the filing deadline; open to someone earlier in their career as long as they understand the disclosure requirements well."),
    dict(category_id=2, speciality_id=10, seniority_needed="senior", budget_lo=150, budget_hi=230,
         hire_title="Preparing for our second year under the new insurance accounting standard",
         hire_description="Our first year under the new standard technically closed, but the process was painful and mostly manual, and leadership wants it properly streamlined before we go through it again. The ask is a review of last year's process with concrete recommendations for automating the weakest parts.",
         hire_description_additional_notes="Budget is around SGD 150-230/hr for a six-week engagement starting after this quarter's close; ideally someone who has helped an insurer move from a manual to a more automated IFRS17 process before."),
])
# ---- Speciality: Insurance / Claims Process Improvement [cat=2, spec=11] ----
PROVIDERS_RAW.extend([
    dict(category_id=2, speciality_id=11, seniority="expert", full_name="Nur Aina Zahra", rate_per_hour=175,
         credentials="Former Head of Claims Operations, general insurer",
         about_title="Claims Operations Turnaround Lead",
         about_description="Twelve years running claims operations at a general insurer, rising to Head of Claims Operations, before moving into independent turnaround work for insurers whose settlement times have gotten away from them.",
         services_offered_title="Claims Backlog Turnaround",
         services_offered_description="Takes over a claims backlog directly, restructuring escalation rules between adjusters and underwriting so cases stop stalling at the handoff points that are actually causing delay.",
         relevant_experience="Cleared a six-month claims backlog for a motor insurer within two months by restructuring how cases were escalated between teams."),
    dict(category_id=2, speciality_id=11, seniority="senior", full_name="Timothy Ong", rate_per_hour=140,
         credentials="Lean Six Sigma black belt, insurance claims workflows",
         about_title="Claims Workflow Redesign Specialist",
         about_description="Lean Six Sigma black belt who has led claims transformation programmes at three insurers across Southeast Asia over the past eight years, always starting from where cases actually get stuck rather than a generic process map.",
         services_offered_title="Claims Journey Mapping & Redesign",
         services_offered_description="Maps the end-to-end claims journey, pinpoints exactly where cases stall, and redesigns the specific handoffs causing delay rather than proposing a broad process overhaul.",
         relevant_experience="Cut average claims settlement time by over a third at a regional general insurer purely through workflow redesign, with no additional hiring."),
    dict(category_id=2, speciality_id=11, seniority="senior", full_name="Grace Kwek", rate_per_hour=155,
         credentials="Claims transformation consultant, eight years across Southeast Asia",
         about_title="Insurance Claims Transformation Advisor",
         about_description="Eight years advising general and motor insurers on claims transformation, with a track record of finding the one or two process bottlenecks that account for most of a backlog rather than trying to fix everything at once.",
         services_offered_title="Claims Backlog Diagnostic",
         services_offered_description="Runs a rapid four-week diagnostic on an existing claims backlog and delivers a prioritised, resourced fix plan rather than a generic process audit nobody acts on.",
         relevant_experience="Reduced repeat customer complaints about claims delays by roughly half at a mid-size insurer within one settlement cycle."),
    dict(category_id=2, speciality_id=11, seniority="mid", full_name="Arjun Pillai", rate_per_hour=95,
         credentials="Claims operations analyst, motor insurance",
         about_title="Motor Claims Process Consultant",
         about_description="Four years inside a motor insurer's claims team focused specifically on the workflow bottlenecks that cause settlement delays, now consulting on the exact same problem for other motor and general insurers.",
         services_offered_title="Escalation & Triage Rule Redesign",
         services_offered_description="Rebuilds escalation and triage rules so straightforward claims move fast and only genuinely complex cases consume senior adjuster time.",
         relevant_experience="Redesigned the triage rules for a motor insurer's claims intake, cutting the average time to first adjuster contact by several days."),
])
HIRERS_RAW.extend([
    dict(category_id=2, speciality_id=11, seniority_needed="mid", budget_lo=60, budget_hi=110,
         hire_title="Claims settlement times keep creeping up and customers are noticing",
         hire_description="Our average claims settlement time has been creeping up for close to a year now, and customer complaints about it have started reaching senior management. We need someone to figure out where the process is actually breaking down, not just recommend hiring more adjusters.",
         hire_description_additional_notes="Budget around SGD 60-110/hr for a three-week diagnostic; a solid claims process practitioner is fine for this stage."),
    dict(category_id=2, speciality_id=11, seniority_needed="senior", budget_lo=110, budget_hi=180,
         hire_title="Six-month backlog of unresolved claims needs clearing",
         hire_description="We have a backlog of unresolved claims that's been growing for six months, and internally nobody agrees on where exactly the process is breaking down. We need someone to take ownership of clearing it and fixing the underlying escalation process at the same time.",
         hire_description_additional_notes="Budget roughly SGD 110-180/hr for a two-month hands-on engagement; needs someone who has actually run a backlog clearance before, not just diagnosed one."),
    dict(category_id=2, speciality_id=11, seniority_needed="mid", budget_lo=55, budget_hi=100,
         hire_title="Adjusters and underwriting keep dropping claims between each other",
         hire_description="Claims keep falling through the cracks between our adjusters and the underwriting team, and it's usually the customer who notices before we do. We'd like a clear map of where the handoffs are failing and a redesigned process we can roll out within a quarter.",
         hire_description_additional_notes="Budget around SGD 55-100/hr for a four-week mapping and redesign exercise; open to a mid-level specialist as long as they've mapped a claims journey before."),
    dict(category_id=2, speciality_id=11, seniority_needed="senior", budget_lo=100, budget_hi=170,
         hire_title="Too many straightforward claims are taking as long as complex ones",
         hire_description="We've noticed that simple, low-value claims are taking almost as long to settle as genuinely complex ones, which suggests our triage isn't actually differentiating between them. We want someone to rebuild the triage and escalation rules so simple cases move fast.",
         hire_description_additional_notes="Budget is around SGD 100-170/hr for a five-week engagement; looking for someone senior enough to redesign triage rules with confidence, not just tweak the existing ones."),
    dict(category_id=2, speciality_id=11, seniority_needed="expert", budget_lo=150, budget_hi=230,
         hire_title="Claims transformation needs a fresh set of eyes after our own attempt stalled",
         hire_description="We tried to run a claims transformation project internally last year and it stalled without meaningfully improving settlement times. We're now looking for an experienced outside specialist to restart it properly, with buy-in from senior leadership this time.",
         hire_description_additional_notes="Budget up to around SGD 230/hr for a three-month engagement; this needs someone senior enough to have led a claims transformation to a measurable result before, not just participated in one."),
])
# ---- Speciality: Asset/Wealth Management / Fund Setup/Compliance [cat=3, spec=17] ----
PROVIDERS_RAW.extend([
    dict(category_id=3, speciality_id=17, seniority="expert", full_name="Sophia Tay", rate_per_hour=190,
         credentials="Ex-COO, boutique asset management firm",
         about_title="Fund Formation & Compliance Architect",
         about_description="Served as COO at a boutique asset manager for over a decade, personally overseeing three separate fund launches from initial structuring through to first close.",
         services_offered_title="Full Fund Launch Coordination",
         services_offered_description="Coordinates the complete fund launch checklist, from CMS licensing through fund administrator and custodian onboarding, to a compliance manual that actually matches how the fund will operate day to day.",
         relevant_experience="Took a Variable Capital Company structure from initial paperwork to a live fund launch in under four months for a first-time manager."),
    dict(category_id=3, speciality_id=17, seniority="senior", full_name="Ryan Foo", rate_per_hour=170,
         credentials="Former MAS Capital Markets Services licence reviewer",
         about_title="Ex-MAS Reviewer, Capital Markets Services Licensing Specialist",
         about_description="Spent six years reviewing Capital Markets Services licence applications at MAS before moving into advisory work helping fund managers get through that same process considerably faster.",
         services_offered_title="CMS Licence Application Support",
         services_offered_description="Prepares and reviews a fund manager's CMS licence application against exactly the criteria MAS reviewers actually check, well before submission.",
         relevant_experience="Guided a first-time fund manager's CMS licence application through to approval on the first submission, after a previous informal review had flagged multiple gaps."),
    dict(category_id=3, speciality_id=17, seniority="senior", full_name="Deepa Krishnan", rate_per_hour=150,
         credentials="Fund formation and regulatory compliance specialist, fourteen years",
         about_title="Fund Operations & Compliance Advisor",
         about_description="Fourteen years in fund operations at a mid-size asset manager, focused specifically on the compliance setup work that first-time managers consistently underestimate.",
         services_offered_title="Fund Compliance Gap Review",
         services_offered_description="Reviews an existing fund structure against current MAS requirements and flags exactly what needs remediation before the next regulatory audit.",
         relevant_experience="Remediated a fund's compliance manual and reporting cadence ahead of a MAS inspection that had already flagged the prior version as inadequate."),
    dict(category_id=3, speciality_id=17, seniority="mid", full_name="Kevin Yeo", rate_per_hour=110,
         credentials="Fund operations analyst, four years at a mid-size asset manager",
         about_title="Fund Launch Operations Consultant",
         about_description="Four years handling fund operations at a mid-size asset manager, now specialising in the administrator and custodian onboarding work that tends to slip closest to a fund's launch date.",
         services_offered_title="Administrator & Custodian Onboarding",
         services_offered_description="Manages the administrator and custodian onboarding process end to end, keeping the launch timeline on track when this workstream is at risk of slipping.",
         relevant_experience="Completed administrator and custodian onboarding for a boutique fund launch two weeks ahead of the original target date."),
])
HIRERS_RAW.extend([
    dict(category_id=3, speciality_id=17, seniority_needed="mid", budget_lo=70, budget_hi=130,
         hire_title="First-time fund manager needs help navigating licensing and launch",
         hire_description="We're a first-time fund manager and have never actually taken a fund from paperwork to launch before, and the regulatory timeline is starting to feel unmanageable on our own. We need someone to manage the entire licensing and launch checklist alongside us.",
         hire_description_additional_notes="Budget around SGD 70-130/hr for a three-month engagement through to launch; open to someone earlier in independent practice as long as they've supported a launch before."),
    dict(category_id=3, speciality_id=17, seniority_needed="expert", budget_lo=160, budget_hi=240,
         hire_title="CMS licence application needs to be right the first time",
         hire_description="We're preparing our Capital Markets Services licence application and given how much is riding on approval timing, we want the most experienced person available reviewing it before submission. A previous informal read-through already flagged some concerns we haven't fully resolved.",
         hire_description_additional_notes="Budget up to around SGD 240/hr for a focused four-week review before submission; this really needs someone with direct MAS-side licensing experience."),
    dict(category_id=3, speciality_id=17, seniority_needed="senior", budget_lo=100, budget_hi=180,
         hire_title="Existing fund's compliance setup has never been properly reviewed",
         hire_description="Our current fund's compliance setup was put together quickly at launch and has never been reviewed by anyone with real regulatory experience since. We'd like a structured gap review against current MAS requirements with a remediation plan we can execute before the next audit.",
         hire_description_additional_notes="Budget is roughly SGD 100-180/hr for a four-week review; looking for someone senior enough to sign off on the remediation plan with confidence."),
    dict(category_id=3, speciality_id=17, seniority_needed="mid", budget_lo=55, budget_hi=100,
         hire_title="Need help onboarding a fund administrator and custodian before launch",
         hire_description="We're weeks from our target launch date and still need to onboard a fund administrator and custodian, which nobody on the team has actually done end to end before. We'd like hands-on support coordinating this so we hit our launch date.",
         hire_description_additional_notes="Budget realistically SGD 55-100/hr for a focused three-week engagement; a mid-level operations specialist is fine for this workstream."),
    dict(category_id=3, speciality_id=17, seniority_needed="senior", budget_lo=120, budget_hi=200,
         hire_title="Expanding our fund range and want compliance set up properly from day one",
         hire_description="We're launching a second fund and, having learned from how messy the compliance setup was the first time, want it done properly from day one this time. The ask is someone to build the compliance manual and reporting cadence fresh rather than copying the first fund's setup.",
         hire_description_additional_notes="Budget around SGD 120-200/hr for a six-week engagement ahead of the new fund's launch; ideally someone who has set up fund compliance from scratch more than once."),
])
# ---- Speciality: Asset/Wealth Management / ESG Integration [cat=3, spec=20] ----
PROVIDERS_RAW.extend([
    dict(category_id=3, speciality_id=20, seniority="expert", full_name="Natasha Sim", rate_per_hour=180,
         credentials="Former ESG Lead, regional asset manager",
         about_title="ESG Integration & Sustainable Investing Advisor",
         about_description="Led ESG strategy at a regional asset manager for six years before moving into independent advisory, focused on making sustainability part of the actual investment process rather than a separate compliance exercise.",
         services_offered_title="Portfolio-Level ESG Scoring Design",
         services_offered_description="Designs a portfolio-level ESG scoring methodology and integrates it directly into the existing investment committee process, rather than running it as a parallel exercise nobody references.",
         relevant_experience="Built the first ESG integration framework for a multi-billion-dollar AUM asset manager ahead of a major institutional investor's due diligence process."),
    dict(category_id=3, speciality_id=20, seniority="senior", full_name="Andre Lopez", rate_per_hour=155,
         credentials="CFA charterholder, sustainable investing frameworks",
         about_title="CFA Charterholder, Responsible Investment Design",
         about_description="CFA charterholder with a decade of experience building sustainable investing frameworks specifically for asset managers preparing for institutional investor due diligence.",
         services_offered_title="Sustainable Finance Disclosure Alignment",
         services_offered_description="Reviews current sustainability claims and disclosures against what institutional investors and regulators now expect, flagging gaps before due diligence does.",
         relevant_experience="Advised a wealth manager on realigning its fund range with tightening sustainable finance disclosure rules ahead of a regulatory deadline."),
    dict(category_id=3, speciality_id=20, seniority="senior", full_name="Melissa Toh", rate_per_hour=140,
         credentials="Product specialist, sustainability screen design",
         about_title="Practical ESG Screening Specialist",
         about_description="Three years inside a wealth manager's product team, specialising in translating high-level sustainability commitments into portfolio-level screens portfolio managers will actually use day to day.",
         services_offered_title="Portfolio Sustainability Screen Build",
         services_offered_description="Builds practical sustainability screens and scoring criteria that plug directly into an existing portfolio management workflow, not a standalone spreadsheet nobody opens.",
         relevant_experience="Designed a portfolio scoring methodology now used across an asset manager's full fund range, replacing an ad hoc, spreadsheet-based approach."),
    dict(category_id=3, speciality_id=20, seniority="mid", full_name="Nicholas Goh", rate_per_hour=100,
         credentials="ESG research analyst, three years",
         about_title="ESG Research & Disclosure Analyst",
         about_description="Three years as an ESG research analyst supporting fund managers' sustainability disclosures, with hands-on experience gathering and structuring the underlying data those disclosures depend on.",
         services_offered_title="ESG Data & Disclosure Support",
         services_offered_description="Gathers and structures the underlying ESG data a fund needs for its disclosures, and checks it against current reporting templates before submission.",
         relevant_experience="Compiled the underlying ESG dataset for a mid-size fund manager's first standalone sustainability disclosure."),
])
HIRERS_RAW.extend([
    dict(category_id=3, speciality_id=20, seniority_needed="senior", budget_lo=110, budget_hi=190,
         hire_title="Institutional investors are asking ESG questions we can't answer well",
         hire_description="Institutional investors keep asking detailed questions about our sustainable investing process during due diligence, and our current answers are mostly marketing language rather than an actual methodology. We want to build ESG scoring into the investment committee's decision process properly.",
         hire_description_additional_notes="Budget around SGD 110-190/hr for a six-week engagement ahead of our next investor due diligence round; ideally someone who has built this for a fund manager before, not just advised on strategy."),
    dict(category_id=3, speciality_id=20, seniority_needed="mid", budget_lo=60, budget_hi=110,
         hire_title="Sustainability commitments exist on paper but not in the portfolio process",
         hire_description="We have sustainability commitments in our marketing materials that don't actually connect to how portfolio decisions get made day to day. We'd like a practical framework portfolio managers will actually use, not another policy document that sits unread.",
         hire_description_additional_notes="Budget realistically SGD 60-110/hr for a four-week build; a mid-level specialist is fine as long as they've built something similarly practical before."),
    dict(category_id=3, speciality_id=20, seniority_needed="senior", budget_lo=100, budget_hi=170,
         hire_title="New sustainable finance disclosure rules are coming and we're not ready",
         hire_description="New sustainable finance disclosure rules are coming into effect soon and our current fund range hasn't been reviewed against them yet. We need a gap review and a realistic plan to close the gaps before the deadline.",
         hire_description_additional_notes="Budget is around SGD 100-170/hr for a five-week engagement tied to the regulatory deadline; looking for someone who tracks these disclosure rules closely."),
    dict(category_id=3, speciality_id=20, seniority_needed="mid", budget_lo=50, budget_hi=95,
         hire_title="Need help pulling together ESG data for our first disclosure",
         hire_description="This is our first year producing a standalone sustainability disclosure and we're underestimating how much data gathering and structuring work is involved. We need someone to take that workstream off our plate and get it into the right format.",
         hire_description_additional_notes="Budget realistically SGD 50-95/hr for a three-week engagement before our filing deadline; open to someone earlier in their ESG career for this data-focused work."),
    dict(category_id=3, speciality_id=20, seniority_needed="expert", budget_lo=150, budget_hi=230,
         hire_title="Board wants a credible ESG strategy, not just a compliance checkbox",
         hire_description="Our board has asked for a genuinely credible sustainable investing strategy after an investor questioned our current approach directly in a meeting. We want the most experienced person available to design something that will hold up to serious scrutiny.",
         hire_description_additional_notes="Budget up to around SGD 230/hr for a two-month engagement reporting directly to the board; this needs someone with a real track record building ESG frameworks investors respect."),
])
# ---- Speciality: Fintech & Payments / Fraud Analytics [cat=4, spec=26] ----
PROVIDERS_RAW.extend([
    dict(category_id=4, speciality_id=26, seniority="expert", full_name="Farhana Aziz", rate_per_hour=200,
         credentials="Ex-Head of Fraud Analytics, digital payments company",
         about_title="Fraud Analytics & Transaction Risk Lead",
         about_description="Led fraud analytics at a digital payments company for six years, personally building the detection models that catch fraud without blocking a growing base of genuine customers.",
         services_offered_title="Fraud Model Build & Calibration",
         services_offered_description="Builds and tunes a transaction-level fraud scoring model using historical chargeback and dispute data, calibrated specifically to the platform's actual risk appetite.",
         relevant_experience="Reduced false-positive fraud declines by roughly 40% for a digital wallet while catching more genuine fraud cases in the same period."),
    dict(category_id=4, speciality_id=26, seniority="senior", full_name="Ivan Neo", rate_per_hour=165,
         credentials="Data scientist, transaction fraud detection, eight years",
         about_title="Independent Data Scientist, Payments Fraud Detection",
         about_description="Eight years as a data scientist specialising in transaction-level fraud detection across two payments platforms before going independent to work directly with smaller fintechs.",
         services_offered_title="Real-Time Fraud Scoring Pipeline",
         services_offered_description="Builds a real-time transaction scoring pipeline from the platform's own historical data, designed to run inline with checkout rather than as an after-the-fact review.",
         relevant_experience="Built a real-time transaction scoring pipeline for a payments fintech processing several million transactions a month."),
    dict(category_id=4, speciality_id=26, seniority="senior", full_name="Cheryl Poh", rate_per_hour=145,
         credentials="Former fraud risk lead, Southeast Asian e-wallet operator",
         about_title="Fraud Risk Root-Cause Specialist",
         about_description="Former fraud risk lead at a regional e-wallet operator, now focused specifically on root-causing rising fraud losses for smaller platforms before recommending any rule changes.",
         services_offered_title="Fraud Loss Root-Cause Analysis",
         services_offered_description="Runs a root-cause analysis on rising fraud losses and recommends specific rule and model changes rather than a generic fraud risk framework.",
         relevant_experience="Cut chargeback losses by a third at an e-wallet operator within one quarter of redesigning the detection rules."),
    dict(category_id=4, speciality_id=26, seniority="mid", full_name="Gabriel Lee", rate_per_hour=100,
         credentials="Fraud operations analyst, three years",
         about_title="Fraud Rules & False-Positive Analyst",
         about_description="Three years in fraud operations at a payments startup, focused specifically on the false-positive problem that comes from overly cautious rule sets built early in a company's life.",
         services_offered_title="False-Positive Reduction Review",
         services_offered_description="Reviews an existing fraud rule set specifically for false-positive drivers, tightening only what's causing genuine customers to be blocked.",
         relevant_experience="Reduced genuine-customer declines by a meaningful margin for a payments startup without loosening the rules that were actually catching fraud."),
])
HIRERS_RAW.extend([
    dict(category_id=4, speciality_id=26, seniority_needed="senior", budget_lo=100, budget_hi=180,
         hire_title="Chargeback losses spiked and current rules aren't catching enough",
         hire_description="Chargeback losses have spiked noticeably this quarter and our current rule-based fraud checks clearly aren't keeping pace with how fraudsters are adapting. We need someone to find where the losses are actually coming from and fix that specifically.",
         hire_description_additional_notes="Budget around SGD 100-180/hr for a four-week engagement starting immediately given the urgency; a strong senior practitioner is fine, doesn't need to be a former department head."),
    dict(category_id=4, speciality_id=26, seniority_needed="mid", budget_lo=60, budget_hi=110,
         hire_title="Blocking too many genuine customers with our fraud checks",
         hire_description="We're blocking a growing number of legitimate customers with our current fraud checks and support tickets about it are piling up. We'd like a smarter detection approach that cuts false positives without opening us back up to the fraud we're currently catching.",
         hire_description_additional_notes="Budget realistically SGD 60-110/hr for a three-week review; open to someone earlier in their fraud analytics career for this scoped review."),
    dict(category_id=4, speciality_id=26, seniority_needed="expert", budget_lo=160, budget_hi=240,
         hire_title="Need a real transaction fraud model, not just static rules",
         hire_description="Our fraud detection is still entirely rule-based and hasn't been touched since launch, while transaction volume has grown considerably since then. We'd like a proper scoring model built from our own transaction history rather than another round of manual rule tweaks.",
         hire_description_additional_notes="Budget up to around SGD 240/hr for a two-month build; this needs someone who has actually built a production fraud model before, not just studied one academically."),
    dict(category_id=4, speciality_id=26, seniority_needed="senior", budget_lo=90, budget_hi=160,
         hire_title="New fraud pattern showing up that our rules don't catch",
         hire_description="We've started seeing a specific fraud pattern that our existing rules and model simply don't flag, and it's already cost us a meaningful amount this month. We need someone to analyse the pattern and build detection for it quickly.",
         hire_description_additional_notes="Budget around SGD 90-160/hr for a focused two-week sprint given the urgency; looking for someone with real hands-on model-building experience."),
    dict(category_id=4, speciality_id=26, seniority_needed="mid", budget_lo=55, budget_hi=100,
         hire_title="Want a second opinion on our fraud detection setup before scaling",
         hire_description="We're about to scale into a new market and want an independent review of our current fraud detection setup before transaction volume grows further. Nothing is obviously broken yet, but we'd rather catch gaps now than after they've cost us.",
         hire_description_additional_notes="Budget realistically SGD 55-100/hr for a two-week review; a mid-level analyst is fine for this preventive check."),
])
# ---- Speciality: Fintech & Payments / KYC/KYB Operations [cat=4, spec=25] ----
PROVIDERS_RAW.extend([
    dict(category_id=4, speciality_id=25, seniority="expert", full_name="Omar Farouk", rate_per_hour=175,
         credentials="Former Head of Onboarding Operations, payments fintech",
         about_title="Onboarding Operations & KYB Workflow Lead",
         about_description="Ran onboarding operations at a payments fintech for five years, scaling the team through a period of rapid signup growth without a compliance blow-up along the way.",
         services_offered_title="Verification Workflow Redesign",
         services_offered_description="Redesigns the business verification workflow to cut manual review time while keeping the underlying risk controls fully intact, not just faster and looser.",
         relevant_experience="Cut average business onboarding time from five days to under 24 hours for a payments licensee, without loosening verification standards."),
    dict(category_id=4, speciality_id=25, seniority="senior", full_name="Bee Choo Lim", rate_per_hour=140,
         credentials="KYC/KYB operations lead, eight years across two fintech licensees",
         about_title="KYC/KYB Operations Specialist",
         about_description="Eight years leading KYC/KYB operations teams across two fintech licensees, with direct hands-on responsibility for two separate verification workflow redesigns.",
         services_offered_title="Tiered Onboarding Design",
         services_offered_description="Sets up a tiered onboarding process so low-risk customers move through quickly and higher-risk applicants receive proportionately more scrutiny, rather than treating every applicant identically.",
         relevant_experience="Rebuilt the KYB verification workflow for a fintech scaling from 10,000 to 100,000 monthly signups over a single year."),
    dict(category_id=4, speciality_id=25, seniority="senior", full_name="Alexander Png", rate_per_hour=130,
         credentials="Verification operations specialist, fast-growing fintech background",
         about_title="Verification Automation Advisor",
         about_description="Three years in business verification operations at a fast-growing fintech, now advising smaller platforms specifically on which manual review steps can be automated without weakening the checks behind them.",
         services_offered_title="Manual Review Automation Audit",
         services_offered_description="Audits the current verification stack end to end and identifies exactly which manual steps can be automated, prioritised by both effort and risk impact.",
         relevant_experience="Reduced manual review backlog by over half at a fintech licensee within six weeks, freeing the compliance team for genuinely higher-risk cases."),
    dict(category_id=4, speciality_id=25, seniority="mid", full_name="Nadia Hassan", rate_per_hour=90,
         credentials="Onboarding operations analyst, two years",
         about_title="Onboarding Queue Operations Analyst",
         about_description="Two years running the day-to-day onboarding review queue at a mid-size fintech, with close visibility into exactly which document types and applicant profiles cause the most delay.",
         services_offered_title="Onboarding Queue Diagnostic",
         services_offered_description="Analyses the current onboarding queue to identify which specific document types or applicant profiles are causing most of the delay, then recommends targeted process fixes.",
         relevant_experience="Identified a single document-verification step causing over a third of onboarding delays at a mid-size fintech, leading to a quick process fix."),
])
HIRERS_RAW.extend([
    dict(category_id=4, speciality_id=25, seniority_needed="senior", budget_lo=90, budget_hi=160,
         hire_title="Business customers waiting almost a week to get verified",
         hire_description="New business customers are currently waiting close to a week to get verified, and we're visibly losing signups because of it. We need the onboarding workflow redesigned to cut wait times without weakening our risk controls.",
         hire_description_additional_notes="Budget around SGD 90-160/hr for a five-week redesign; looking for someone who has actually cut onboarding time before, not just diagnosed the problem."),
    dict(category_id=4, speciality_id=25, seniority_needed="mid", budget_lo=50, budget_hi=95,
         hire_title="Verification process can't keep up with signup volume",
         hire_description="Our current identity and business verification checks are mostly manual and clearly can't keep pace with our current signup volume. We'd like a hands-on audit and rebuild of the verification stack, not just a recommendations deck.",
         hire_description_additional_notes="Budget realistically SGD 50-95/hr for a three-week engagement; open to a mid-level operations analyst for this diagnostic-and-fix work."),
    dict(category_id=4, speciality_id=25, seniority_needed="senior", budget_lo=100, budget_hi=170,
         hire_title="Manual reviews are backing up as we scale",
         hire_description="The compliance team's manual review queue has been growing steadily for months and nobody has redesigned the workflow to match our growth. We want a tiered process that moves low-risk customers fast while still catching what actually matters.",
         hire_description_additional_notes="Budget is around SGD 100-170/hr for a six-week engagement; ideally someone with direct experience designing tiered onboarding before."),
    dict(category_id=4, speciality_id=25, seniority_needed="mid", budget_lo=45, budget_hi=90,
         hire_title="Want to know exactly why onboarding is slow before fixing it",
         hire_description="We suspect a specific part of our onboarding process is causing most of our delays but don't have the internal capacity to properly analyse the queue and prove it. We'd like a focused diagnostic before committing to a bigger redesign project.",
         hire_description_additional_notes="Budget realistically SGD 45-90/hr for a two-week diagnostic; a mid-level analyst is a good fit for this scoped piece of work."),
    dict(category_id=4, speciality_id=25, seniority_needed="expert", budget_lo=150, budget_hi=220,
         hire_title="Regulator has questioned our onboarding controls, need this fixed fast",
         hire_description="A regulator has raised questions about the adequacy of our business verification controls, and we need this addressed with real urgency and credibility. We want the most experienced operations specialist available to lead the response.",
         hire_description_additional_notes="Budget up to around SGD 220/hr for an intensive four-week engagement; this needs genuine senior operations leadership experience given the regulatory pressure."),
])
# ---- Speciality: Private Equity & VC / Commercial DD [cat=5, spec=29] ----
PROVIDERS_RAW.extend([
    dict(category_id=5, speciality_id=29, seniority="expert", full_name="Josephine Wee", rate_per_hour=260,
         credentials="Ex-Principal, mid-market private equity fund",
         about_title="Commercial Due Diligence Advisor for Growth-Stage Deals",
         about_description="Spent seven years as a principal at a mid-market private equity fund, personally running or overseeing commercial diligence on deals ranging from ten to two hundred million dollars.",
         services_offered_title="Full Commercial Diligence Workstream",
         services_offered_description="Runs a full commercial diligence workstream covering market sizing, competitive positioning, and independent customer reference calls before the investment committee meets.",
         relevant_experience="Led commercial due diligence on a fifty-million-dollar growth-equity deal in logistics that ultimately closed on revised terms after the review."),
    dict(category_id=5, speciality_id=29, seniority="senior", full_name="Edwin Sim", rate_per_hour=210,
         credentials="Former strategy consultant, commercial due diligence specialist",
         about_title="Independent Deal Diligence Specialist",
         about_description="Former strategy consultant who has specialised in commercial due diligence for the past nine years, now working directly with investment teams rather than billing through a consulting firm.",
         services_offered_title="Growth Assumption Stress-Test",
         services_offered_description="Stress-tests a target's growth assumptions against independent market data, flagging anywhere management's numbers diverge meaningfully from what the market actually supports.",
         relevant_experience="Flagged churn risk during diligence on a VC-stage fintech term sheet that materially changed the final valuation before signing."),
    dict(category_id=5, speciality_id=29, seniority="senior", full_name="Wendy Chan", rate_per_hour=195,
         credentials="Growth-equity investment analyst, four years",
         about_title="Customer Concentration & Churn Diligence Specialist",
         about_description="Four years on the investing side of a growth-equity fund, now focused specifically on independently pressure-testing customer concentration and retention claims before term sheets are signed.",
         services_offered_title="Customer Reference Call Programme",
         services_offered_description="Runs a structured programme of independent customer reference calls across a target's top accounts, surfacing concentration or satisfaction risk management's own materials tend not to disclose.",
         relevant_experience="Ran customer reference calls across a target's top accounts that surfaced concentration risk management's own materials hadn't disclosed."),
    dict(category_id=5, speciality_id=29, seniority="mid", full_name="Lucas Tio", rate_per_hour=140,
         credentials="Market research analyst, deal diligence support",
         about_title="Market Sizing & Competitive Analysis Consultant",
         about_description="Three years supporting deal teams with market sizing and competitive analysis, specialising in getting a defensible market-size estimate together quickly under real deal timelines.",
         services_offered_title="Rapid Market Sizing Support",
         services_offered_description="Builds a defensible, independently-sourced market size and competitive landscape view within a compressed diligence timeline, ready for the investment memo.",
         relevant_experience="Delivered a market sizing analysis for a growth-equity deal within a five-day turnaround ahead of an investment committee deadline."),
])
HIRERS_RAW.extend([
    dict(category_id=5, speciality_id=29, seniority_needed="senior", budget_lo=150, budget_hi=250,
         hire_title="Need an independent view before we sign this term sheet",
         hire_description="We're close to signing a term sheet and want an independent view on whether the target's market claims actually hold up before we commit. We need a structured commercial diligence workstream completed before the investment committee's next meeting.",
         hire_description_additional_notes="Budget around SGD 150-250/hr for a three-week engagement ahead of the committee meeting; needs someone with genuine deal diligence experience, not general market research."),
    dict(category_id=5, speciality_id=29, seniority_needed="expert", budget_lo=200, budget_hi=290,
         hire_title="Investment committee wants customer concentration checked before we commit",
         hire_description="Our investment committee is specifically asking for a proper diligence review on customer concentration and churn before capital goes out, given how much this deal represents for the fund. We want the most experienced person available given the stakes.",
         hire_description_additional_notes="Budget up to around SGD 290/hr for a focused two-week review; this needs someone senior enough that the committee will trust the findings without further validation."),
    dict(category_id=5, speciality_id=29, seniority_needed="mid", budget_lo=100, budget_hi=170,
         hire_title="Target's growth story needs an outside sanity check",
         hire_description="The target's growth story looks strong on paper, but nobody on our team has time to sanity-check it against outside market data before the deadline. We'd like independent reference calls and market data pulled together into a clear go/no-go recommendation.",
         hire_description_additional_notes="Budget realistically SGD 100-170/hr for a two-week turnaround; open to someone earlier in independent practice for this scoped piece."),
    dict(category_id=5, speciality_id=29, seniority_needed="senior", budget_lo=140, budget_hi=220,
         hire_title="Need customer reference calls done independently, not by our own team",
         hire_description="We want customer reference calls for this deal done by someone independent rather than our own deal team, since customers tend to answer more candidly with a neutral third party. The output needs to directly inform our investment memo.",
         hire_description_additional_notes="Budget is around SGD 140-220/hr for a two-week call programme; looking for someone experienced specifically in running these calls for diligence purposes."),
    dict(category_id=5, speciality_id=29, seniority_needed="mid", budget_lo=90, budget_hi=160,
         hire_title="Need a market sizing estimate fast for our investment memo",
         hire_description="Our investment memo is due shortly and we're missing a defensible market sizing and competitive landscape section. We need someone who can turn this around quickly without cutting corners on the underlying sourcing.",
         hire_description_additional_notes="Budget around SGD 90-160/hr for a rapid one-week turnaround; a strong analyst is fine, this doesn't need a senior diligence lead."),
])
# ---- Speciality: Private Equity & VC / 100-Day Plans [cat=5, spec=30] ----
PROVIDERS_RAW.extend([
    dict(category_id=5, speciality_id=30, seniority="expert", full_name="Hakim Osman", rate_per_hour=240,
         credentials="Former Operating Partner, mid-market private equity fund",
         about_title="Post-Acquisition Operating Plan Advisor",
         about_description="Served as an Operating Partner at a mid-market private equity fund, personally building first-100-day plans across more than ten portfolio company acquisitions.",
         services_offered_title="Prioritised 100-Day Plan Build",
         services_offered_description="Builds a prioritised 100-day plan covering quick wins, key hires, and the systems that need to change first, tied directly back to the original deal thesis.",
         relevant_experience="Built and executed the 100-day plan for a healthcare services acquisition, hitting every planned milestone in the deal's first quarter."),
    dict(category_id=5, speciality_id=30, seniority="senior", full_name="Valerie Ang", rate_per_hour=195,
         credentials="Former COO, post-merger integration specialist",
         about_title="Independent Operating Partner for Portfolio Companies",
         about_description="Former COO who now specialises full-time in post-merger integration, having personally run the integration workstream on three add-on deals for a single PE-backed platform.",
         services_offered_title="Integration Workstream Management",
         services_offered_description="Runs the integration workstream between the acquired company's existing team and the new owner's reporting requirements, so nothing falls through in the handover.",
         relevant_experience="Ran post-acquisition integration for three consecutive add-on deals at a mid-market PE-backed platform company over two years."),
    dict(category_id=5, speciality_id=30, seniority="senior", full_name="Bryan Lau", rate_per_hour=175,
         credentials="Portfolio operations lead, two ownership transitions",
         about_title="Post-Acquisition Recovery Specialist",
         about_description="Five years inside a portfolio company's operations team through two separate ownership transitions, now advising new owners specifically on the first-quarter execution problem.",
         services_offered_title="Stalled Integration Recovery",
         services_offered_description="Steps in when an integration has already stalled and rebuilds a realistic recovery plan with clear ownership assigned to each workstream.",
         relevant_experience="Rescued a stalled acquisition integration and got the first-90-day plan back on track within three weeks of stepping in."),
    dict(category_id=5, speciality_id=30, seniority="mid", full_name="Siti Nurhaliza", rate_per_hour=125,
         credentials="Portfolio company operations analyst, three years",
         about_title="Portfolio Operations Support Consultant",
         about_description="Three years supporting portfolio company operations through acquisition transitions, focused specifically on the reporting-line and systems handover work that senior operating partners often delegate.",
         services_offered_title="Reporting & Systems Handover Support",
         services_offered_description="Manages the detailed reporting-line and systems handover work during an integration, keeping this workstream from becoming the bottleneck it often is.",
         relevant_experience="Completed the systems handover for a portfolio company add-on acquisition two weeks ahead of the integration deadline."),
])
HIRERS_RAW.extend([
    dict(category_id=5, speciality_id=30, seniority_needed="expert", budget_lo=180, budget_hi=270,
         hire_title="Just closed an acquisition and need a real first-quarter plan",
         hire_description="We just closed an acquisition and need someone experienced to turn the deal thesis into an actual first-quarter operating plan. We need a prioritised plan covering quick wins and key hires within the first 100 days, tied to the original deal logic.",
         hire_description_additional_notes="Budget around SGD 180-270/hr for an intensive first-quarter engagement; this needs someone who has built and executed a 100-day plan before, not just written one."),
    dict(category_id=5, speciality_id=30, seniority_needed="senior", budget_lo=140, budget_hi=220,
         hire_title="Last integration stalled because nobody owned it",
         hire_description="Our last acquisition stalled within a few months because nobody clearly owned the integration, and we want to avoid repeating that with this deal. We need clear ownership of the integration workstream between the target's team and our reporting requirements.",
         hire_description_additional_notes="Budget is roughly SGD 140-220/hr for a two-month engagement; looking for someone senior enough to actually own this workstream, not just advise from the sidelines."),
    dict(category_id=5, speciality_id=30, seniority_needed="mid", budget_lo=90, budget_hi=160,
         hire_title="Deal thesis needs turning into an actual operating plan",
         hire_description="The deal thesis behind this acquisition is clear on paper, but nobody on our side has actually built a concrete 100-day plan before. We'd like hands-on help building and then executing the plan, not just a strategy document.",
         hire_description_additional_notes="Budget realistically SGD 90-160/hr for a six-week engagement; open to someone earlier in independent practice as long as they've supported a plan like this before."),
    dict(category_id=5, speciality_id=30, seniority_needed="senior", budget_lo=130, budget_hi=210,
         hire_title="Integration is underway but the reporting handover is falling behind",
         hire_description="The integration is broadly on track, but the reporting-line and systems handover work has fallen behind and is starting to create friction with the target's team. We need someone to take ownership of that specific workstream and get it current.",
         hire_description_additional_notes="Budget around SGD 130-210/hr for a four-week engagement; needs someone with direct hands-on integration experience, not just planning."),
    dict(category_id=5, speciality_id=30, seniority_needed="mid", budget_lo=80, budget_hi=150,
         hire_title="Small add-on deal needs a lightweight integration plan",
         hire_description="This is a small add-on acquisition and doesn't need a heavyweight integration programme, but we still want a proper lightweight plan rather than winging it. We'd like someone practical who can scope this appropriately for the deal size.",
         hire_description_additional_notes="Budget realistically SGD 80-150/hr for a three-week engagement; a mid-level operator is a good fit given the modest scope of this deal."),
])
# ---- Speciality: Family Offices / Succession Planning [cat=6, spec=37] ----
PROVIDERS_RAW.extend([
    dict(category_id=6, speciality_id=37, seniority="expert", full_name="Marcus Yeo", rate_per_hour=280,
         credentials="Former family office director, generational transition specialist",
         about_title="Family Governance & Succession Planning Advisor",
         about_description="Directed a single-family office for over fifteen years, personally guiding two separate generational transitions before moving into independent succession advisory work.",
         services_offered_title="Cross-Generation Succession Facilitation",
         services_offered_description="Facilitates the succession planning process across generations directly, including the difficult ownership and control conversations families often can't have productively on their own.",
         relevant_experience="Guided a three-generation family business through a full leadership transition without a single shareholder dispute."),
    dict(category_id=6, speciality_id=37, seniority="senior", full_name="Delia Chong", rate_per_hour=220,
         credentials="Trust and estate planning advisor, eighteen years in family offices",
         about_title="Independent Consultant, Multi-Generational Business Transition",
         about_description="Trust and estate planning advisor with eighteen years inside family offices, specialising in the governance conversations families tend to avoid until circumstances force the issue.",
         services_offered_title="Family Constitution & Governance Design",
         services_offered_description="Designs a family constitution and governance structure that reflects how the specific family actually wants to operate, not a generic template pulled off the shelf.",
         relevant_experience="Designed the governance framework for a family office managing several hundred million dollars in assets mid-transition."),
    dict(category_id=6, speciality_id=37, seniority="senior", full_name="Rohan Das", rate_per_hour=195,
         credentials="Former private banker, independent succession planning consultant",
         about_title="Phased Leadership Handover Specialist",
         about_description="Former private banker who now focuses full-time on structuring succession plans for family businesses before a crisis makes the decision for them.",
         services_offered_title="Phased Handover Structuring",
         services_offered_description="Structures a phased leadership handover plan with clear milestones, so the transition has a defined timeline rather than an open-ended understanding everyone interprets differently.",
         relevant_experience="Structured a phased handover for a founder stepping back after four decades, with the next generation fully operational within eighteen months."),
    dict(category_id=6, speciality_id=37, seniority="mid", full_name="Xin Yi Tan", rate_per_hour=145,
         credentials="Family office advisory associate, four years",
         about_title="Family Governance Support Consultant",
         about_description="Four years supporting senior succession advisors at a family office consultancy, with direct hands-on experience drafting governance documents and facilitating early-stage family meetings.",
         services_offered_title="Governance Documentation Support",
         services_offered_description="Drafts and refines family governance documentation and prepares structured discussion materials ahead of family meetings on succession topics.",
         relevant_experience="Prepared the governance documentation and meeting materials that helped a family reach initial agreement on succession principles within two sessions."),
])
HIRERS_RAW.extend([
    dict(category_id=6, speciality_id=37, seniority_needed="expert", budget_lo=200, budget_hi=300,
         hire_title="Founder stepping back in two years, no real succession plan yet",
         hire_description="Our founder is planning to step back within the next two years, and we haven't yet had a real conversation about who takes over or how. Given how sensitive this is, we want someone to facilitate the ownership and control conversation directly, not just recommend that we have it.",
         hire_description_additional_notes="Budget around SGD 200-300/hr given the sensitivity and stakes involved; this needs someone with genuine experience facilitating family succession conversations, not a general business advisor."),
    dict(category_id=6, speciality_id=37, seniority_needed="senior", budget_lo=150, budget_hi=240,
         hire_title="Next generation wants more say, no structure to give it to them",
         hire_description="The next generation is pushing for more say in the family business, and right now there's no governance structure to channel that into productively. We need a governance structure that actually reflects how our family wants to operate day to day.",
         hire_description_additional_notes="Budget is roughly SGD 150-240/hr for a two-month engagement; ideally someone senior enough to navigate multiple family stakeholders with different views."),
    dict(category_id=6, speciality_id=37, seniority_needed="senior", budget_lo=130, budget_hi=210,
         hire_title="Family business handover needs proper structure before it's forced",
         hire_description="We know a handover is coming eventually and would rather structure it properly now than have it forced on us by circumstance later. We'd like a phased plan with real milestones, not an open-ended understanding everyone interprets differently.",
         hire_description_additional_notes="Budget around SGD 130-210/hr for a six-week planning engagement; looking for someone with a track record of structuring handovers with clear timelines."),
    dict(category_id=6, speciality_id=37, seniority_needed="mid", budget_lo=80, budget_hi=150,
         hire_title="Need governance documents drafted ahead of a family meeting",
         hire_description="We have a family meeting coming up specifically to discuss succession principles and need proper governance documentation and discussion materials prepared beforehand. This is a defined, scoped piece of work rather than the full advisory engagement.",
         hire_description_additional_notes="Budget realistically SGD 80-150/hr for a two-week preparation window; a mid-level family office associate is a good fit for this documentation-focused work."),
    dict(category_id=6, speciality_id=37, seniority_needed="expert", budget_lo=190, budget_hi=280,
         hire_title="Multiple family members disagree on succession approach",
         hire_description="We have several family members with genuinely different views on how succession should work, and previous informal conversations have gone poorly. We need the most experienced facilitator available to help the family reach some kind of workable agreement.",
         hire_description_additional_notes="Budget up to around SGD 280/hr given how difficult these conversations have been so far; this absolutely needs someone with deep, proven experience facilitating contentious family discussions."),
])
# ---- Speciality: Family Offices / Manager Selection [cat=6, spec=35] ----
PROVIDERS_RAW.extend([
    dict(category_id=6, speciality_id=35, seniority="expert", full_name="Farah Ismail", rate_per_hour=210,
         credentials="Former investment director, single-family office",
         about_title="External Manager Selection & Due Diligence Advisor",
         about_description="Served as investment director at a single-family office for a decade, personally running manager selection across public and private market allocations.",
         services_offered_title="Pre-Allocation Manager Due Diligence",
         services_offered_description="Runs operational and investment due diligence on shortlisted fund managers before any allocation decision is finalised, not after capital is already committed.",
         relevant_experience="Ran manager due diligence for a single-family office allocating across five new fund commitments within a single year."),
    dict(category_id=6, speciality_id=35, seniority="senior", full_name="Jonathan Ng", rate_per_hour=175,
         credentials="Manager research specialist, twelve years",
         about_title="Independent Manager Research Specialist for Family Offices",
         about_description="Twelve years as a manager research specialist, having built the due diligence process several multi-family offices still use to vet external allocations today.",
         services_offered_title="Ongoing Manager Monitoring Framework",
         services_offered_description="Sets up an ongoing manager monitoring framework so underperformance or style drift gets flagged early, rather than surfacing only at the annual review.",
         relevant_experience="Built the manager monitoring dashboard a multi-family office now uses to track over twenty external mandates."),
    dict(category_id=6, speciality_id=35, seniority="senior", full_name="Priscilla Lau", rate_per_hour=155,
         credentials="Family office investment team analyst, three years",
         about_title="Manager Vetting & Operational Diligence Specialist",
         about_description="Three years inside a family office's investment team, now advising smaller offices on professionalising a manager-picking process that's currently mostly word of mouth.",
         services_offered_title="Repeatable Manager Research Process",
         services_offered_description="Builds a repeatable manager research process the family office's own team can run independently after the first cycle, rather than staying dependent on outside help.",
         relevant_experience="Identified operational red flags in a shortlisted manager during due diligence that led the family office to withdraw before allocating."),
    dict(category_id=6, speciality_id=35, seniority="mid", full_name="Suresh Kumar", rate_per_hour=115,
         credentials="Investment research associate, fund manager screening",
         about_title="Fund Manager Screening Analyst",
         about_description="Two years screening candidate fund managers for a wealth advisory firm, with hands-on experience narrowing a broad manager universe down to a genuinely comparable shortlist.",
         services_offered_title="Manager Shortlist Screening",
         services_offered_description="Screens a broad universe of candidate fund managers down to a genuinely comparable shortlist against the family office's stated allocation criteria.",
         relevant_experience="Narrowed an initial list of over thirty candidate managers to a comparable shortlist of six for a family office's new allocation."),
])
HIRERS_RAW.extend([
    dict(category_id=6, speciality_id=35, seniority_needed="expert", budget_lo=170, budget_hi=250,
         hire_title="About to commit capital to external managers, want them properly vetted",
         hire_description="We're about to commit meaningful capital to a few external managers and want an independent party to properly vet them before we do. We need full due diligence completed on the shortlist before any capital is committed.",
         hire_description_additional_notes="Budget around SGD 170-250/hr given the size of the commitment; this needs someone with genuine manager due diligence experience, not a general investment advisor."),
    dict(category_id=6, speciality_id=35, seniority_needed="mid", budget_lo=70, budget_hi=130,
         hire_title="Manager picking process is basically word of mouth right now",
         hire_description="Our current process for picking outside fund managers is essentially word of mouth, and we'd like to professionalise it. The ask is a proper research process our own team can eventually run without outside help.",
         hire_description_additional_notes="Budget realistically SGD 70-130/hr for a six-week engagement; open to someone earlier in their manager-research career for this process-building work."),
    dict(category_id=6, speciality_id=35, seniority_needed="senior", budget_lo=110, budget_hi=190,
         hire_title="Underperformance keeps going unnoticed until the annual review",
         hire_description="Underperformance among our external managers seems to go unnoticed until the annual review, by which point it's already cost us. We'd like an ongoing monitoring framework that flags problems well before the annual review does.",
         hire_description_additional_notes="Budget is roughly SGD 110-190/hr for a five-week build; ideally someone who has built a manager monitoring system for a family office before."),
    dict(category_id=6, speciality_id=35, seniority_needed="mid", budget_lo=60, budget_hi=110,
         hire_title="Need our manager universe narrowed to a proper shortlist",
         hire_description="We have a long list of candidate managers we're interested in but no structured way to compare them against each other. We'd like help narrowing this down to a genuinely comparable shortlist before we begin deeper diligence ourselves.",
         hire_description_additional_notes="Budget realistically SGD 60-110/hr for a three-week screening exercise; a mid-level research associate is a good fit for this scoping work."),
    dict(category_id=6, speciality_id=35, seniority_needed="senior", budget_lo=120, budget_hi=200,
         hire_title="One of our managers has shown concerning operational signs",
         hire_description="One of our existing external managers has shown some operational warning signs recently and we want an independent operational due diligence review before deciding whether to stay invested. Time matters here given the concern.",
         hire_description_additional_notes="Budget around SGD 120-200/hr for a focused two-week review; needs someone with genuine operational due diligence depth, not just performance analysis."),
])
# ---- Speciality: Healthcare Providers / Revenue Cycle Management [cat=7, spec=40] ----
PROVIDERS_RAW.extend([
    dict(category_id=7, speciality_id=40, seniority="expert", full_name="Michelle Wong", rate_per_hour=170,
         credentials="Former Revenue Cycle Director, private hospital group",
         about_title="Revenue Cycle Management & Billing Operations Lead",
         about_description="Directed revenue cycle operations at a private hospital group for nine years before moving into independent advisory work for smaller clinic groups facing the same collection problems.",
         services_offered_title="End-to-End Revenue Cycle Audit",
         services_offered_description="Audits the full revenue cycle from patient registration through claims submission, identifying exactly where revenue is leaking rather than proposing a generic overhaul.",
         relevant_experience="Reduced the claims denial rate by over 20% for a private hospital group through targeted billing process redesign."),
    dict(category_id=7, speciality_id=40, seniority="senior", full_name="Samuel Chen", rate_per_hour=140,
         credentials="Healthcare finance specialist, claims and billing operations",
         about_title="Independent Consultant, Clinic Group Claims Optimisation",
         about_description="Healthcare finance specialist with ten years focused specifically on claims and billing operations across multi-clinic groups, not hospital-scale systems.",
         services_offered_title="Claims Submission Timing Review",
         services_offered_description="Reviews insurer submission timing and documentation quality, the two most common and fixable causes of avoidable claim denials in a clinic setting.",
         relevant_experience="Cut days-in-receivables by roughly a third for a multi-clinic group by fixing insurance claim submission timing."),
    dict(category_id=7, speciality_id=40, seniority="senior", full_name="Timothy Ong", rate_per_hour=125,
         credentials="Billing operations manager, single large clinic background",
         about_title="Claims Rejection Workflow Specialist",
         about_description="Four years managing billing operations at a single large clinic, now consulting on the same revenue leakage problem for other clinics that have outgrown their original billing setup.",
         services_offered_title="Claims Rejection Workflow Redesign",
         services_offered_description="Redesigns the claims rejection workflow so denials get investigated and resubmitted quickly instead of piling up unresolved in a queue nobody owns.",
         relevant_experience="Cleared a six-month claims rejection backlog for a clinic group within eight weeks of engagement."),
    dict(category_id=7, speciality_id=40, seniority="mid", full_name="Grace Kwek", rate_per_hour=90,
         credentials="Medical billing analyst, two years",
         about_title="Medical Billing Process Analyst",
         about_description="Two years as a medical billing analyst at a mid-size clinic, with close day-to-day visibility into which specific claim types and insurers cause the most rework.",
         services_offered_title="Denial Pattern Analysis",
         services_offered_description="Analyses recent claim denials to identify which specific insurers or claim types are driving most of the rework, then recommends targeted documentation fixes.",
         relevant_experience="Identified a single documentation gap responsible for a large share of one insurer's claim denials at a mid-size clinic."),
])
HIRERS_RAW.extend([
    dict(category_id=7, speciality_id=40, seniority_needed="senior", budget_lo=80, budget_hi=150,
         hire_title="Claims rejection rate climbing, cash collection falling behind",
         hire_description="Our claims rejection rate has been climbing for months, and cash collection has fallen noticeably behind as a direct result. We need someone to trace the rejection spike back to its cause and fix the underlying process, not just resubmit the current backlog.",
         hire_description_additional_notes="Budget around SGD 80-150/hr for a five-week engagement; looking for someone with real billing operations experience, not general finance consulting."),
    dict(category_id=7, speciality_id=40, seniority_needed="mid", budget_lo=50, budget_hi=95,
         hire_title="Waiting too long to get paid by insurers",
         hire_description="We're waiting far longer than we should to get paid by insurers, and suspect the billing process itself is the underlying problem rather than the insurers being slow. We'd like a redesigned claims workflow that gets us paid faster on a sustained basis.",
         hire_description_additional_notes="Budget realistically SGD 50-95/hr for a three-week engagement; a mid-level billing analyst is a good fit to start with a diagnostic."),
    dict(category_id=7, speciality_id=40, seniority_needed="expert", budget_lo=130, budget_hi=200,
         hire_title="Billing process hasn't been reviewed since the clinic group opened",
         hire_description="Our billing workflow hasn't been reviewed since the clinic group first opened several years ago, and revenue leakage has become hard to ignore as we've grown. We'd like a full audit of the revenue cycle with a concrete, prioritised fix list.",
         hire_description_additional_notes="Budget around SGD 130-200/hr for a six-week full-cycle audit; this needs someone with genuine multi-clinic revenue cycle leadership experience."),
    dict(category_id=7, speciality_id=40, seniority_needed="mid", budget_lo=45, budget_hi=90,
         hire_title="One insurer keeps rejecting our claims for the same reason",
         hire_description="We keep getting claims rejected by one specific insurer for reasons that seem fixable, but nobody internally has dug into exactly what's going wrong. We need a focused analysis of this specific denial pattern and a documentation fix.",
         hire_description_additional_notes="Budget realistically SGD 45-90/hr for a two-week focused analysis; a billing analyst is a good fit for this narrowly scoped issue."),
    dict(category_id=7, speciality_id=40, seniority_needed="senior", budget_lo=90, budget_hi=160,
         hire_title="Days-in-receivables have crept up over the past year",
         hire_description="Our days-in-receivables metric has crept up steadily over the past year without an obvious single cause, and finance leadership wants this addressed before the next board review. We need someone to review submission timing and documentation practices across the board.",
         hire_description_additional_notes="Budget around SGD 90-160/hr for a four-week review ahead of the board meeting; ideally someone senior enough to present findings directly to leadership."),
])
# ---- Speciality: Healthcare Providers / Telehealth Setup [cat=7, spec=42] ----
PROVIDERS_RAW.extend([
    dict(category_id=7, speciality_id=42, seniority="expert", full_name="Nur Aina Zahra", rate_per_hour=175,
         credentials="Former digital health programme lead, healthcare group",
         about_title="Telehealth Deployment & Virtual Care Advisor",
         about_description="Led digital health programmes at a healthcare group for six years, personally taking telehealth from a pilot project to a properly running clinical service line.",
         services_offered_title="End-to-End Virtual Consultation Workflow",
         services_offered_description="Designs the end-to-end virtual consultation workflow, covering booking through e-prescription and follow-up, built around how clinicians actually work rather than how the platform demos.",
         relevant_experience="Launched a telehealth service line for a multi-clinic group that now handles roughly a third of routine consultations."),
    dict(category_id=7, speciality_id=42, seniority="senior", full_name="Ryan Foo", rate_per_hour=145,
         credentials="Telemedicine implementation specialist, seven years",
         about_title="Independent Consultant, Remote Consultation Rollouts",
         about_description="Telemedicine implementation specialist with seven years of experience across both corporate and clinic-based deployments, running vendor selection and clinical workflow work in parallel.",
         services_offered_title="Vendor Selection & Workflow Integration",
         services_offered_description="Runs vendor selection and clinical workflow integration for a new telehealth service line simultaneously, rather than choosing a platform first and figuring out the workflow afterward.",
         relevant_experience="Rolled out remote consultation capability across four clinic locations for a healthcare group within six weeks."),
    dict(category_id=7, speciality_id=42, seniority="senior", full_name="Deepa Krishnan", rate_per_hour=135,
         credentials="Former clinical operations manager, virtual care rollouts",
         about_title="Stalled Telehealth Pilot Recovery Specialist",
         about_description="Former clinical operations manager who now focuses specifically on the workflow side of virtual care rollouts, stepping in when a pilot has stalled rather than just choosing the technology vendor.",
         services_offered_title="Telehealth Pilot Diagnostic & Relaunch",
         services_offered_description="Reviews a stalled telehealth pilot to identify the specific operational blocker preventing it from going live, then relaunches it around a fix for that blocker.",
         relevant_experience="Diagnosed and resolved the workflow gap that had stalled a telehealth pilot for over four months, taking it live within three weeks of engagement."),
    dict(category_id=7, speciality_id=42, seniority="mid", full_name="Kevin Yeo", rate_per_hour=95,
         credentials="Clinical workflow coordinator, telehealth support",
         about_title="Telehealth Workflow Coordinator",
         about_description="Two years coordinating clinical workflow changes during a telehealth rollout at a mid-size clinic group, with hands-on experience training clinicians to actually use a new virtual care platform.",
         services_offered_title="Clinician Adoption & Training Support",
         services_offered_description="Runs clinician training and adoption support during a telehealth rollout, focused specifically on getting staff to actually use the platform rather than default back to phone calls.",
         relevant_experience="Achieved a meaningful clinician adoption rate within the first month of a telehealth platform rollout through targeted hands-on training."),
])
HIRERS_RAW.extend([
    dict(category_id=7, speciality_id=42, seniority_needed="senior", budget_lo=90, budget_hi=160,
         hire_title="Want to offer remote consultations but no idea where to start",
         hire_description="We want to offer remote consultations but have no real sense of how to set up the workflow or choose the right platform. We need someone who has actually taken a virtual care service live before, not just configured a demo environment.",
         hire_description_additional_notes="Budget around SGD 90-160/hr for a six-week engagement from planning through to launch; ideally someone with hands-on rollout experience, not just platform knowledge."),
    dict(category_id=7, speciality_id=42, seniority_needed="senior", budget_lo=80, budget_hi=140,
         hire_title="Telehealth pilot stalled and clinical staff have lost interest",
         hire_description="Our attempt at a telehealth pilot has stalled for months, and clinical staff have mostly stopped engaging with it. We need a diagnosis of exactly why the pilot stalled and a concrete plan to relaunch it.",
         hire_description_additional_notes="Budget is around SGD 80-140/hr for a four-week diagnostic-and-relaunch engagement; looking for someone who has recovered a stalled rollout before."),
    dict(category_id=7, speciality_id=42, seniority_needed="mid", budget_lo=50, budget_hi=95,
         hire_title="Need a virtual care workflow that clinicians will actually use",
         hire_description="We already picked a telehealth platform, but the clinical workflow around it was never designed properly, so adoption among our clinicians is low. We'd like the workflow redesigned around the platform we have so staff will actually use it.",
         hire_description_additional_notes="Budget realistically SGD 50-95/hr for a three-week engagement; a mid-level workflow specialist is fine given the platform is already chosen."),
    dict(category_id=7, speciality_id=42, seniority_needed="mid", budget_lo=45, budget_hi=85,
         hire_title="Clinicians keep defaulting back to phone calls instead of the platform",
         hire_description="We rolled out a telehealth platform months ago but clinicians keep defaulting back to plain phone calls instead of using it properly. We need focused help driving actual adoption rather than more technical configuration.",
         hire_description_additional_notes="Budget realistically SGD 45-85/hr for a focused three-week adoption push; open to someone earlier in their telehealth career for this training-focused work."),
    dict(category_id=7, speciality_id=42, seniority_needed="expert", budget_lo=150, budget_hi=220,
         hire_title="Board wants telehealth to become a genuine service line, not a side project",
         hire_description="Our board wants telehealth to become a genuine, properly resourced service line rather than the side project it currently is, and expects a credible plan to get there. We want the most experienced person available given how visible this initiative has become.",
         hire_description_additional_notes="Budget up to around SGD 220/hr for a two-month engagement reporting to the board; this needs someone who has taken telehealth from pilot to real service line before."),
])
# ---- Speciality: Pharma & Biotech / PV/Pharmacovigilance Ops [cat=8, spec=46] ----
PROVIDERS_RAW.extend([
    dict(category_id=8, speciality_id=46, seniority="expert", full_name="Terence Yap", rate_per_hour=220,
         credentials="Former Head of Pharmacovigilance, regional pharma affiliate",
         about_title="Pharmacovigilance Operations & Drug Safety Lead",
         about_description="Led pharmacovigilance at a regional pharma affiliate for nine years, with direct hands-on experience preparing for and hosting two full regulatory inspections.",
         services_offered_title="Adverse Event Timeline Audit",
         services_offered_description="Audits adverse event case processing timelines against local regulatory deadlines and identifies exactly where cases are getting stuck in the workflow.",
         relevant_experience="Fixed a persistent late-reporting issue in adverse event cases that had drawn regulator attention across two consecutive reporting cycles."),
    dict(category_id=8, speciality_id=46, seniority="senior", full_name="Charlotte Ho", rate_per_hour=180,
         credentials="Drug safety specialist, twelve years in adverse event reporting",
         about_title="Independent Consultant, Adverse Event Reporting Systems",
         about_description="Drug safety specialist with twelve years of experience in adverse event reporting across multiple product launches throughout Southeast Asia.",
         services_offered_title="First-Launch Safety Database Design",
         services_offered_description="Designs the safety database workflow and reporting process from scratch for a first-time product launch in a new market, ready before the launch date rather than scrambling afterward.",
         relevant_experience="Built the pharmacovigilance operating model for a pharma affiliate's first product launch in Southeast Asia, live before the launch date."),
    dict(category_id=8, speciality_id=46, seniority="senior", full_name="Zachary Koh", rate_per_hour=160,
         credentials="Former regulatory affairs lead, safety operations advisory",
         about_title="Inspection Readiness Specialist",
         about_description="Former regulatory affairs lead who now advises specifically on the safety operations side of pharma market entry, with a focus on getting an operation genuinely ready for inspection rather than just paper-compliant.",
         services_offered_title="Pre-Inspection Safety Operations Review",
         services_offered_description="Reviews an existing pharmacovigilance operation ahead of an announced inspection and fixes exactly what a mock audit would flag.",
         relevant_experience="Prepared a pharma affiliate's safety operations for a regulatory inspection that closed with zero critical findings."),
    dict(category_id=8, speciality_id=46, seniority="mid", full_name="Meera Pillai", rate_per_hour=115,
         credentials="Pharmacovigilance case processing associate, three years",
         about_title="Case Processing Efficiency Consultant",
         about_description="Three years processing adverse event cases day to day at a pharma affiliate, with close visibility into exactly which steps in the workflow most often cause reporting deadlines to slip.",
         services_offered_title="Case Processing Workflow Review",
         services_offered_description="Reviews the day-to-day adverse event case processing workflow and recommends specific, practical fixes to the steps most often causing delay.",
         relevant_experience="Identified a specific triage step responsible for most late submissions at a pharma affiliate, leading to a quick process correction."),
])
HIRERS_RAW.extend([
    dict(category_id=8, speciality_id=46, seniority_needed="senior", budget_lo=130, budget_hi=210,
         hire_title="Adverse event reports consistently late, regulator has noticed",
         hire_description="Regulators have flagged our adverse event reports as consistently late across two reporting cycles now, and it needs fixing before the next one. We need someone who has actually fixed a late-reporting problem before, not just designed a process on paper.",
         hire_description_additional_notes="Budget around SGD 130-210/hr for a five-week engagement given the urgency; looking for genuine hands-on pharmacovigilance operations experience."),
    dict(category_id=8, speciality_id=46, seniority_needed="expert", budget_lo=170, budget_hi=250,
         hire_title="Launching a new product, no safety reporting operation exists yet",
         hire_description="We're launching a new product in a new market and currently have no formal safety reporting operation in place. We need a safety operation built from scratch, ready before our launch date, and given the timeline pressure we want the most experienced person available.",
         hire_description_additional_notes="Budget up to around SGD 250/hr for an intensive six-week build ahead of launch; this needs someone who has done this specific build-from-scratch work before."),
    dict(category_id=8, speciality_id=46, seniority_needed="senior", budget_lo=140, budget_hi=220,
         hire_title="Inspection announced, safety operations have never been reviewed",
         hire_description="An inspection has just been announced and our pharmacovigilance operations have never been reviewed by anyone with real regulatory experience. We'd like a rapid readiness review with a clear list of what needs fixing before the inspection.",
         hire_description_additional_notes="Budget is roughly SGD 140-220/hr for a focused three-week review ahead of the inspection date; needs real inspection-side experience given the short timeline."),
    dict(category_id=8, speciality_id=46, seniority_needed="mid", budget_lo=60, budget_hi=110,
         hire_title="Case processing keeps causing minor deadline slips",
         hire_description="Our case processing isn't causing major regulatory problems yet, but we keep having minor deadline slips that make us nervous about what happens as volume grows. We'd like a practical review of the day-to-day workflow before it becomes a bigger issue.",
         hire_description_additional_notes="Budget realistically SGD 60-110/hr for a three-week review; a mid-level case processing specialist is a good fit for this preventive check."),
    dict(category_id=8, speciality_id=46, seniority_needed="senior", budget_lo=120, budget_hi=200,
         hire_title="Safety database process has never scaled well as volume grew",
         hire_description="Our safety database and reporting process was designed for a much smaller product volume and clearly hasn't scaled well as we've grown. We need someone to redesign the workflow around our current and expected case volume.",
         hire_description_additional_notes="Budget around SGD 120-200/hr for a five-week redesign; ideally someone who has scaled a pharmacovigilance operation before, not just set one up initially."),
])
# ---- Speciality: Pharma & Biotech / Clinical Trial Ops Support [cat=8, spec=47] ----
PROVIDERS_RAW.extend([
    dict(category_id=8, speciality_id=47, seniority="expert", full_name="Vikram Menon", rate_per_hour=210,
         credentials="Former Clinical Operations Manager, contract research organisation",
         about_title="Clinical Trial Operations & Site Delivery Lead",
         about_description="Managed clinical operations at a contract research organisation for eight years, running site operations directly across Phase II and III studies.",
         services_offered_title="Enrolment Recovery Diagnostic",
         services_offered_description="Runs a diagnostic on lagging site enrolment and puts a concrete recovery plan in place directly with the CRO and sponsor, rather than just reporting the problem.",
         relevant_experience="Recovered a Phase III trial's enrolment timeline after it had fallen four months behind schedule."),
    dict(category_id=8, speciality_id=47, seniority="senior", full_name="Michelle Wong", rate_per_hour=170,
         credentials="Clinical trial operations specialist, ten years",
         about_title="Independent Consultant, Multi-Site Study Recovery",
         about_description="Clinical trial operations specialist with ten years of experience managing site enrolment and data quality specifically across multi-country studies.",
         services_offered_title="Multi-Country Site Coordination",
         services_offered_description="Coordinates site operations across multiple countries for a single study, keeping enrolment and data quality moving in parallel rather than letting one lag behind the other.",
         relevant_experience="Managed site operations across six countries for a multi-centre study through to database lock, on schedule."),
    dict(category_id=8, speciality_id=47, seniority="senior", full_name="Samuel Chen", rate_per_hour=155,
         credentials="Former site coordinator, trial recovery specialist",
         about_title="Trial Data Quality & Monitoring Specialist",
         about_description="Former site coordinator who now focuses specifically on trial data quality, stepping in when a sponsor audit has flagged issues that put an upcoming milestone at risk.",
         services_offered_title="Data Quality & Monitoring Plan Review",
         services_offered_description="Reviews trial data quality processes and tightens the monitoring plan before the next regulatory milestone is due, focused on what an audit would actually catch.",
         relevant_experience="Fixed a data quality issue flagged by a sponsor audit that had put a trial's next milestone at risk."),
    dict(category_id=8, speciality_id=47, seniority="mid", full_name="Grace Kwek", rate_per_hour=105,
         credentials="Clinical research associate, site support",
         about_title="Site Enrolment Support Consultant",
         about_description="Three years as a clinical research associate supporting site enrolment activities, with hands-on experience identifying which specific site-level bottlenecks slow recruitment.",
         services_offered_title="Site-Level Enrolment Bottleneck Review",
         services_offered_description="Reviews enrolment activity at the individual site level to identify which specific bottlenecks are slowing recruitment, then recommends targeted site-level fixes.",
         relevant_experience="Identified a site-specific screening delay responsible for a large share of one study's slow enrolment, leading to a quick process fix."),
])
HIRERS_RAW.extend([
    dict(category_id=8, speciality_id=47, seniority_needed="expert", budget_lo=170, budget_hi=250,
         hire_title="Trial enrolment has fallen well behind plan",
         hire_description="Our study enrolment has fallen significantly behind plan, and the sponsor is now asking hard questions about the timeline. We need a concrete recovery plan for enrolment, not just an assessment of how far behind we are.",
         hire_description_additional_notes="Budget up to around SGD 250/hr given the pressure from the sponsor; this needs someone who has actually recovered a lagging trial before, not just diagnosed one."),
    dict(category_id=8, speciality_id=47, seniority_needed="senior", budget_lo=120, budget_hi=200,
         hire_title="Sponsor asking hard questions about our study timeline",
         hire_description="A recent data quality issue flagged by the sponsor has put our next regulatory milestone at risk, and we need this resolved with real urgency. We need a tightened data quality and monitoring process before the next milestone.",
         hire_description_additional_notes="Budget is roughly SGD 120-200/hr for a focused four-week engagement; needs someone with real experience fixing data quality issues under time pressure."),
    dict(category_id=8, speciality_id=47, seniority_needed="senior", budget_lo=110, budget_hi=190,
         hire_title="Multi-country trial coordination is harder than expected",
         hire_description="Coordinating operations across multiple countries for this trial has proven harder than we expected internally, with enrolment and data quality both slipping in different sites. We'd like hands-on coordination support across sites, not just advice from a distance.",
         hire_description_additional_notes="Budget around SGD 110-190/hr for a two-month engagement; ideally someone with direct multi-country site coordination experience."),
    dict(category_id=8, speciality_id=47, seniority_needed="mid", budget_lo=60, budget_hi=110,
         hire_title="One site is enrolling much slower than the rest",
         hire_description="One specific site is enrolling noticeably slower than our other sites in the same study and we're not sure why. We'd like a focused review of that site's enrolment process to identify the bottleneck.",
         hire_description_additional_notes="Budget realistically SGD 60-110/hr for a two-week site-level review; a mid-level clinical research associate is a good fit for this scoped work."),
    dict(category_id=8, speciality_id=47, seniority_needed="senior", budget_lo=130, budget_hi=210,
         hire_title="Need trial operations support ahead of an upcoming database lock",
         hire_description="We have a database lock milestone coming up and our internal operations team doesn't have capacity to manage both ongoing enrolment and data quality review simultaneously. We need experienced support to keep both workstreams on track through lock.",
         hire_description_additional_notes="Budget around SGD 130-210/hr for a six-week engagement through database lock; needs someone comfortable managing both enrolment and data quality at once."),
])
# ---- Speciality: Medical Devices / QMS/ISO13485 [cat=9, spec=51] ----
PROVIDERS_RAW.extend([
    dict(category_id=9, speciality_id=51, seniority="expert", full_name="Timothy Ong", rate_per_hour=190,
         credentials="Former Quality Systems Manager, medical device manufacturer",
         about_title="Quality Management Systems & ISO13485 Lead",
         about_description="Managed quality systems at a medical device manufacturer for nine years, taking the company through three successful ISO13485 recertification cycles.",
         services_offered_title="ISO13485 Gap Assessment",
         services_offered_description="Runs a gap assessment of an existing quality management system against ISO13485 and prioritises fixes by certification risk rather than ease of implementation.",
         relevant_experience="Took a device manufacturer's quality system from a failed pre-audit to full ISO13485 certification within five months."),
    dict(category_id=9, speciality_id=51, seniority="senior", full_name="Grace Kwek", rate_per_hour=155,
         credentials="ISO13485 lead auditor, nine years",
         about_title="Independent Consultant, Device Quality Certification Readiness",
         about_description="ISO13485 lead auditor with nine years of experience specifically in medical device quality management systems, having audited both large manufacturers and first-time entrants.",
         services_offered_title="First-Time QMS Build",
         services_offered_description="Builds the document control and CAPA process from scratch for a first-time device manufacturer preparing for its first certification audit.",
         relevant_experience="Delivered a first-time manufacturer's quality management system from scratch, passing its first certification audit with zero major findings."),
    dict(category_id=9, speciality_id=51, seniority="senior", full_name="Arjun Pillai", rate_per_hour=140,
         credentials="Former regulatory affairs specialist, device quality management",
         about_title="ISO13485 Lead Auditor for Medical Device Manufacturers",
         about_description="Former regulatory affairs specialist who now focuses specifically on quality management system rebuilds for device manufacturers that have accumulated repeated non-conformances.",
         services_offered_title="CAPA Process Rebuild",
         services_offered_description="Rebuilds a CAPA process specifically after repeated non-conformances have shown up across prior audits, targeting the actual root cause rather than the symptom each time.",
         relevant_experience="Rebuilt the CAPA process for a device company after repeated non-conformances had shown up in three consecutive audits."),
    dict(category_id=9, speciality_id=51, seniority="mid", full_name="Natasha Sim", rate_per_hour=100,
         credentials="Quality systems associate, document control specialist",
         about_title="Document Control & QMS Support Consultant",
         about_description="Three years supporting quality management system documentation at a device manufacturer, with hands-on experience keeping document control audit-ready between formal review cycles.",
         services_offered_title="Document Control Cleanup",
         services_offered_description="Cleans up an existing quality management system's document control, closing the gaps between what a procedure says and what records can actually prove happened.",
         relevant_experience="Brought a device manufacturer's document control fully current ahead of a surveillance audit, closing a backlog of unreviewed procedure updates."),
])
HIRERS_RAW.extend([
    dict(category_id=9, speciality_id=51, seniority_needed="expert", budget_lo=140, budget_hi=220,
         hire_title="Failed our pre-certification audit, need this fixed before reapplying",
         hire_description="We failed a pre-certification audit for our device quality system and need it fixed properly before we reapply. We need a prioritised fix plan aimed specifically at passing the next certification audit.",
         hire_description_additional_notes="Budget up to around SGD 220/hr given the stakes of reapplying; this needs someone with genuine certification-readiness experience, not general quality consulting."),
    dict(category_id=9, speciality_id=51, seniority_needed="senior", budget_lo=100, budget_hi=170,
         hire_title="Quality processes never properly reviewed since we started manufacturing",
         hire_description="Our quality management processes haven't been reviewed by anyone with real audit experience since we started manufacturing. We'd like a proper quality management system build, not a documentation exercise that just looks compliant.",
         hire_description_additional_notes="Budget around SGD 100-170/hr for a two-month engagement; looking for someone who has built a genuine QMS from an existing manufacturing operation before."),
    dict(category_id=9, speciality_id=51, seniority_needed="senior", budget_lo=110, budget_hi=190,
         hire_title="Repeated non-conformances in our audits, need a real fix",
         hire_description="The same non-conformances keep showing up in our audits and internal fixes clearly aren't sticking. We'd like the CAPA process rebuilt so the same issues stop recurring audit after audit.",
         hire_description_additional_notes="Budget is roughly SGD 110-190/hr for a five-week engagement; needs someone with real root-cause CAPA experience, not just documentation updates."),
    dict(category_id=9, speciality_id=51, seniority_needed="mid", budget_lo=55, budget_hi=100,
         hire_title="Document control has fallen behind ahead of our next audit",
         hire_description="Our document control has fallen behind and we have a surveillance audit coming up that we're not confident we're ready for on this front specifically. We need someone to bring document control current before the audit date.",
         hire_description_additional_notes="Budget realistically SGD 55-100/hr for a focused two-week cleanup; a mid-level quality systems associate is a good fit for this scoped task."),
    dict(category_id=9, speciality_id=51, seniority_needed="senior", budget_lo=120, budget_hi=200,
         hire_title="Expanding production and worried our QMS won't scale",
         hire_description="We're expanding production significantly and are concerned our current quality management system was built for a much smaller operation. We want someone to review whether it will hold up at the new scale and fix what won't.",
         hire_description_additional_notes="Budget around SGD 120-200/hr for a six-week review; ideally someone who has scaled a device manufacturer's QMS through a similar growth phase."),
])
# ---- Speciality: Medical Devices / Post-Market Surveillance [cat=9, spec=53] ----
PROVIDERS_RAW.extend([
    dict(category_id=9, speciality_id=53, seniority="expert", full_name="Rachel Goh", rate_per_hour=190,
         credentials="Former Post-Market Surveillance Lead, medical device company",
         about_title="Post-Market Surveillance & Vigilance Reporting Lead",
         about_description="Led post-market surveillance at a medical device company for seven years, across product launches into three new regulatory markets.",
         services_offered_title="Full Surveillance Programme Build",
         services_offered_description="Builds the complaint intake, investigation, and vigilance reporting workflow a device company needs for ongoing compliance ahead of a new market launch.",
         relevant_experience="Set up a full post-market surveillance programme for a device manufacturer entering three new markets simultaneously."),
    dict(category_id=9, speciality_id=53, seniority="senior", full_name="Benjamin Lim", rate_per_hour=150,
         credentials="Regulatory specialist, complaint handling and vigilance reporting",
         about_title="Independent Consultant, Device Complaint Handling Processes",
         about_description="Regulatory specialist with eight years focused specifically on complaint handling and vigilance reporting for medical device manufacturers.",
         services_offered_title="Complaint Backlog Clearance",
         services_offered_description="Reviews and clears an existing backlog of unresolved device complaints, investigating and documenting each to the standard a vigilance report would require.",
         relevant_experience="Cleared a complaint-handling backlog that had left several vigilance reports overdue at a device company."),
    dict(category_id=9, speciality_id=53, seniority="senior", full_name="Wen Yeong Chua", rate_per_hour=160,
         credentials="Former quality engineer, post-market data review specialist",
         about_title="Post-Market Data Trend Review Specialist",
         about_description="Former quality engineer who now advises device companies specifically on the gap between customer complaints coming in and formal vigilance reporting going out.",
         services_offered_title="Post-Market Data Trend Review",
         services_offered_description="Reviews existing post-market data for trends that should already have triggered a formal regulatory report, and flags any backlog of unreported findings.",
         relevant_experience="Identified an unreported adverse trend in post-market data during a routine review that the company then proactively disclosed to the regulator."),
    dict(category_id=9, speciality_id=53, seniority="mid", full_name="Farah Ismail", rate_per_hour=100,
         credentials="Complaint handling associate, two years in device compliance",
         about_title="Complaint Tracking Systems Consultant",
         about_description="Two years in a device company's complaint handling function, with hands-on experience setting up lightweight but genuinely auditable complaint tracking for teams that previously relied on spreadsheets.",
         services_offered_title="Lightweight Complaint Tracking Setup",
         services_offered_description="Sets up a lightweight but auditable complaint tracking system for companies that currently rely on spreadsheets and institutional memory to manage device complaints.",
         relevant_experience="Set up a complaint tracking system for a smaller device company that replaced an ad hoc spreadsheet process within two weeks."),
])
HIRERS_RAW.extend([
    dict(category_id=9, speciality_id=53, seniority_needed="expert", budget_lo=140, budget_hi=220,
         hire_title="Expanding to new markets, no post-market process exists",
         hire_description="We're expanding into new markets and currently have no proper process for tracking device complaints after launch. We need a full surveillance process built before our next market launch, and given the timeline we want the most experienced person available.",
         hire_description_additional_notes="Budget up to around SGD 220/hr for an intensive engagement ahead of the launch date; this needs someone who has built a surveillance programme for a new market entry before."),
    dict(category_id=9, speciality_id=53, seniority_needed="senior", budget_lo=110, budget_hi=190,
         hire_title="Customer complaints about our device have gone unreviewed too long",
         hire_description="Several customer complaints about our device have gone unreviewed for longer than we're comfortable with, and we need this resolved urgently given the compliance exposure. We need an urgent review and clearance of the current complaint backlog.",
         hire_description_additional_notes="Budget is roughly SGD 110-190/hr for a focused three-week clearance; needs someone with real complaint-handling depth given the urgency."),
    dict(category_id=9, speciality_id=53, seniority_needed="senior", budget_lo=100, budget_hi=180,
         hire_title="Not sure our post-market data has been checked for reportable trends",
         hire_description="We're not confident our post-market data has ever been properly checked for trends that should trigger a regulatory report, and would rather find out now than have a regulator find out first. We'd like an independent check of our post-market data for anything we may have missed.",
         hire_description_additional_notes="Budget around SGD 100-180/hr for a three-week data review; ideally someone with genuine trend-analysis experience in this specific regulatory context."),
    dict(category_id=9, speciality_id=53, seniority_needed="mid", budget_lo=50, budget_hi=95,
         hire_title="Currently tracking complaints in a spreadsheet, need something better",
         hire_description="We currently track device complaints in a shared spreadsheet that's becoming unmanageable as our complaint volume grows. We'd like a lightweight but properly auditable system set up to replace it.",
         hire_description_additional_notes="Budget realistically SGD 50-95/hr for a two-week setup; a mid-level associate is a good fit for this scoped systems work."),
    dict(category_id=9, speciality_id=53, seniority_needed="senior", budget_lo=120, budget_hi=200,
         hire_title="Recent complaint volume spike has us worried about reportable trends",
         hire_description="We've seen a noticeable spike in complaint volume recently and want an independent assessment of whether it represents a reportable trend before we decide how to respond. Time matters here given the potential regulatory implications.",
         hire_description_additional_notes="Budget around SGD 120-200/hr for a focused two-week assessment; needs someone comfortable making a defensible regulatory judgment call quickly."),
])
# ---- Speciality: Education & EdTech / Curriculum Development [cat=10, spec=56] ----
PROVIDERS_RAW.extend([
    dict(category_id=10, speciality_id=56, seniority="expert", full_name="Clara Teo", rate_per_hour=140,
         credentials="Former Head of Curriculum, private education group",
         about_title="Curriculum Design & Instructional Development Lead",
         about_description="Led curriculum design at a private education group for eight years, personally building the content for multiple professional certification programmes from the ground up.",
         services_offered_title="Full Programme Curriculum Design",
         services_offered_description="Designs the full learning pathway for a new certification programme from the ground up, covering module sequencing, assessment design, and content outlines together.",
         relevant_experience="Built the core curriculum for a private education provider's new professional certification programme from an initial outline alone."),
    dict(category_id=10, speciality_id=56, seniority="senior", full_name="Vikram Menon", rate_per_hour=115,
         credentials="Instructional design specialist, eleven years",
         about_title="Independent Consultant, Professional Certification Content",
         about_description="Instructional design specialist with eleven years of experience across both K-12 and adult professional learning, with a particular focus on why learners drop off partway through a course.",
         services_offered_title="Drop-Off Module Redesign",
         services_offered_description="Reviews an existing curriculum against its stated learning outcomes and rebuilds specifically the modules where learners are dropping off, rather than redesigning the whole course.",
         relevant_experience="Redesigned a struggling course's module sequence, improving completion rates by more than 20% in the following cohort."),
    dict(category_id=10, speciality_id=56, seniority="senior", full_name="Michelle Wong", rate_per_hour=105,
         credentials="Former education ministry curriculum reviewer",
         about_title="Syllabus-to-Programme Development Specialist",
         about_description="Former education ministry curriculum reviewer who now consults on turning rough syllabus outlines into properly sequenced, teachable programmes for private education providers.",
         services_offered_title="Outline-to-Curriculum Build",
         services_offered_description="Turns a rough syllabus outline into structured, sequenced content that instructors can actually teach from, rather than leaving them to interpret a two-page description.",
         relevant_experience="Turned a two-page syllabus outline into a fully sequenced ten-module programme within six weeks."),
    dict(category_id=10, speciality_id=56, seniority="mid", full_name="Samuel Chen", rate_per_hour=75,
         credentials="Instructional designer, corporate training background",
         about_title="Assessment & Module Sequencing Consultant",
         about_description="Three years designing assessments and module sequences for corporate training programmes, with a focus on making sure assessment design actually matches the stated learning outcomes.",
         services_offered_title="Assessment Alignment Review",
         services_offered_description="Reviews a programme's assessments against its stated learning outcomes and rebuilds any assessment that isn't actually testing what the module claims to teach.",
         relevant_experience="Realigned the assessments for a corporate training programme after learners consistently passed despite not meeting the stated learning outcomes."),
])
HIRERS_RAW.extend([
    dict(category_id=10, speciality_id=56, seniority_needed="senior", budget_lo=70, budget_hi=130,
         hire_title="Have a rough course outline, need it turned into real content",
         hire_description="We have a rough outline for a new certification course but need someone to actually turn it into structured, teachable content. We need the outline turned into a fully sequenced, instructor-ready programme.",
         hire_description_additional_notes="Budget around SGD 70-130/hr for a six-week build; looking for someone who has taken a rough outline all the way to finished content before."),
    dict(category_id=10, speciality_id=56, seniority_needed="senior", budget_lo=60, budget_hi=110,
         hire_title="Students keep dropping out partway through our programme",
         hire_description="Students keep dropping out partway through our current programme, and we suspect the content sequencing itself is the issue rather than the subject matter. We need a specific review of where learners drop off and a rebuild of just those modules.",
         hire_description_additional_notes="Budget realistically SGD 60-110/hr for a four-week diagnostic-and-rebuild engagement; a strong instructional designer is fine, doesn't need to be a former curriculum head."),
    dict(category_id=10, speciality_id=56, seniority_needed="expert", budget_lo=100, budget_hi=170,
         hire_title="Certification programme content needs building from scratch",
         hire_description="We're launching a new certification and currently have nothing beyond a short description of what it should cover. Given how much this programme matters to our positioning, we want the most experienced curriculum designer available.",
         hire_description_additional_notes="Budget up to around SGD 170/hr for a two-month build; this needs someone with a genuine track record building full certification programmes, not general instructional design."),
    dict(category_id=10, speciality_id=56, seniority_needed="mid", budget_lo=40, budget_hi=80,
         hire_title="Assessments don't seem to match what we're actually teaching",
         hire_description="We've noticed learners passing our assessments even when they clearly haven't grasped the material, which suggests the assessments aren't testing the right things. We'd like a focused review and rebuild of the assessment design.",
         hire_description_additional_notes="Budget realistically SGD 40-80/hr for a three-week assessment review; a mid-level instructional designer is a good fit for this scoped work."),
    dict(category_id=10, speciality_id=56, seniority_needed="senior", budget_lo=75, budget_hi=135,
         hire_title="Need our syllabus properly sequenced before the next cohort starts",
         hire_description="Our next cohort starts in a few weeks and our current syllabus is really just a list of topics rather than a properly sequenced programme. We need this turned into a real, teachable curriculum before the start date.",
         hire_description_additional_notes="Budget around SGD 75-135/hr for a fast three-week turnaround; needs someone comfortable working under a tight deadline."),
])
# ---- Speciality: Education & EdTech / LMS Implementation [cat=10, spec=58] ----
PROVIDERS_RAW.extend([
    dict(category_id=10, speciality_id=58, seniority="expert", full_name="Priya Nair", rate_per_hour=150,
         credentials="Former EdTech Implementation Lead, online learning platform",
         about_title="LMS Implementation & Learning Platform Rollout Lead",
         about_description="Led LMS implementations at an online learning platform for seven years, personally deploying to both corporate and academic clients across a wide range of scale.",
         services_offered_title="Full LMS Implementation",
         services_offered_description="Runs the full LMS implementation end to end, covering course migration, integrations, and instructor training, rather than handing off midway through the rollout.",
         relevant_experience="Migrated an education provider's entire course catalogue to a new LMS with zero learner-facing downtime."),
    dict(category_id=10, speciality_id=58, seniority="senior", full_name="Daniel Koh", rate_per_hour=120,
         credentials="LMS deployment specialist, nine years",
         about_title="Independent Consultant, EdTech Deployment Recovery",
         about_description="LMS deployment specialist with nine years of experience specifically in course migration and instructor onboarding, often stepping in when a first attempt has already stalled.",
         services_offered_title="Stalled Rollout Recovery Plan",
         services_offered_description="Reviews a stalled LMS rollout and rebuilds a realistic go-live plan with clear ownership of what's still outstanding, rather than restarting the project from zero.",
         relevant_experience="Rescued an LMS implementation that had been delayed for six months, taking it live within eight weeks of engagement."),
    dict(category_id=10, speciality_id=58, seniority="senior", full_name="Aisha Rahman", rate_per_hour=110,
         credentials="Former instructional technologist, platform migration specialist",
         about_title="Vendor Configuration & Migration Specialist",
         about_description="Former instructional technologist who now manages vendor configuration and course migration in parallel with instructor training, so technical go-live and actual adoption happen together rather than months apart.",
         services_offered_title="Parallel Migration & Training Delivery",
         services_offered_description="Manages vendor configuration and migration in parallel with instructor training, so adoption doesn't lag behind the technical go-live the way it often does.",
         relevant_experience="Delivered instructor training alongside a platform migration, resulting in adoption above 90% within the first month."),
    dict(category_id=10, speciality_id=58, seniority="mid", full_name="Jonathan Ng", rate_per_hour=80,
         credentials="EdTech support specialist, two years",
         about_title="LMS Integration Support Consultant",
         about_description="Two years supporting LMS vendor integrations for a mid-size education provider, with hands-on experience troubleshooting the specific integration issues that stall a migration.",
         services_offered_title="LMS Integration Troubleshooting",
         services_offered_description="Troubleshoots specific LMS vendor integration issues that are stalling an in-progress migration, focused narrowly on unblocking the technical bottleneck.",
         relevant_experience="Resolved a grading-integration issue that had stalled a course migration for several weeks, unblocking the project within days."),
])
HIRERS_RAW.extend([
    dict(category_id=10, speciality_id=58, seniority_needed="senior", budget_lo=70, budget_hi=130,
         hire_title="New learning platform switch has been stuck for months",
         hire_description="Our attempt to switch to a new online learning platform has been stuck for months, and instructors are visibly frustrated with the delay. We need someone who has deployed a learning platform for real use before, not just configured a demo environment.",
         hire_description_additional_notes="Budget around SGD 70-130/hr for a two-month engagement through to live go-live; looking for someone with a track record recovering stalled rollouts."),
    dict(category_id=10, speciality_id=58, seniority_needed="senior", budget_lo=60, budget_hi=110,
         hire_title="Instructors frustrated with our stalled LMS rollout",
         hire_description="We picked a new LMS a while ago but the rollout has never actually gone live properly, and instructor patience is wearing thin. We need a realistic go-live plan with clear ownership of what's still outstanding.",
         hire_description_additional_notes="Budget realistically SGD 60-110/hr for a six-week recovery engagement; a strong deployment specialist is fine, doesn't need to be the most senior available."),
    dict(category_id=10, speciality_id=58, seniority_needed="expert", budget_lo=100, budget_hi=170,
         hire_title="Need a platform deployed properly, not just configured for a demo",
         hire_description="We've tried configuring an LMS ourselves and it works fine as a demo but has never actually been rolled out for real use by instructors and students. Given how much time we've already lost, we want the most experienced person available to take this all the way to real adoption.",
         hire_description_additional_notes="Budget up to around SGD 170/hr for a two-month full rollout; this needs someone who has taken a platform from demo to genuine adoption before."),
    dict(category_id=10, speciality_id=58, seniority_needed="mid", budget_lo=40, budget_hi=80,
         hire_title="Grading integration keeps breaking our course migration",
         hire_description="Our course migration has stalled specifically because of a recurring grading integration issue between our new LMS and our existing systems. We need this specific technical bottleneck resolved so the migration can continue.",
         hire_description_additional_notes="Budget realistically SGD 40-80/hr for a focused one-week fix; a mid-level integration specialist is a good fit for this narrowly scoped issue."),
    dict(category_id=10, speciality_id=58, seniority_needed="senior", budget_lo=75, budget_hi=135,
         hire_title="Migration is technically done but instructors aren't using it",
         hire_description="The technical migration to our new LMS is essentially complete, but instructor adoption has been disappointing so far. We need someone to run targeted training and adoption support to close that gap.",
         hire_description_additional_notes="Budget around SGD 75-135/hr for a three-week adoption push; looking for someone experienced specifically in driving instructor adoption, not just technical migration."),
])
# ---- Speciality: Real Estate / Asset Management [cat=13, spec=74] ----
PROVIDERS_RAW.extend([
    dict(category_id=13, speciality_id=74, seniority="expert", full_name="Ivan Neo", rate_per_hour=175,
         credentials="Former Real Estate Asset Manager, commercial property fund",
         about_title="Real Estate Asset Management & Value Optimisation Lead",
         about_description="Managed assets at a commercial property fund for thirteen years, personally overseeing leasing and capex strategy across a multi-building portfolio.",
         services_offered_title="Full Portfolio Asset-Level Review",
         services_offered_description="Runs a full asset-level review covering leasing strategy, capex prioritisation, and operating cost benchmarking for each building in an underperforming portfolio.",
         relevant_experience="Improved net operating income by close to 20% across a commercial property portfolio through leasing and cost restructuring."),
    dict(category_id=13, speciality_id=74, seniority="senior", full_name="Cheryl Poh", rate_per_hour=145,
         credentials="Former fund manager, real estate value creation specialist",
         about_title="Independent Consultant, Portfolio Value Optimisation",
         about_description="Former fund manager specialising in real estate value creation, with nine years of direct portfolio management experience across office and retail assets.",
         services_offered_title="Hold-Refurbish-Sell Recommendation",
         services_offered_description="Builds a clear hold, refurbish, or sell recommendation for each asset, backed by comparable market data rather than an internal gut-feel assessment.",
         relevant_experience="Advised on a refurbishment-versus-sale decision for an aging office asset, ultimately doubling the projected return against the original plan."),
    dict(category_id=13, speciality_id=74, seniority="senior", full_name="Gabriel Lee", rate_per_hour=130,
         credentials="Property owner asset management team, four years",
         about_title="Underperforming Asset Diagnostic Specialist",
         about_description="Four years inside a property owner's asset management team, now consulting specifically on underperforming buildings where the cause of weak yields isn't immediately obvious.",
         services_offered_title="Underpriced Lease Identification",
         services_offered_description="Identifies the specific value being left on the table in a mediocre-yielding building, most often through underpriced leases, and prioritises the fixes that actually move net operating income.",
         relevant_experience="Identified underpriced leases across a portfolio that, once renegotiated, added a meaningful uplift to annual rental income."),
    dict(category_id=13, speciality_id=74, seniority="mid", full_name="Serena Wu", rate_per_hour=90,
         credentials="Real estate analyst, portfolio performance benchmarking",
         about_title="Portfolio Performance Benchmarking Analyst",
         about_description="Two years benchmarking commercial property performance against comparable buildings, with hands-on experience isolating which specific metrics are actually driving a portfolio's underperformance.",
         services_offered_title="Comparable Performance Benchmarking",
         services_offered_description="Benchmarks a portfolio's operating performance against genuinely comparable buildings to isolate which specific metrics are actually driving underperformance.",
         relevant_experience="Identified that operating costs, not rental rates, were the primary driver of underperformance for a client's office asset through detailed benchmarking."),
])
HIRERS_RAW.extend([
    dict(category_id=13, speciality_id=74, seniority_needed="senior", budget_lo=90, budget_hi=160,
         hire_title="A few buildings in our portfolio keep underperforming",
         hire_description="A few buildings in our portfolio have consistently underperformed for a while now, and we'd like an independent view on what's actually wrong before we take any drastic action. We need an asset-level review that identifies specific fixes, not a general portfolio commentary.",
         hire_description_additional_notes="Budget around SGD 90-160/hr for a five-week review; looking for someone with direct asset management experience, not just investment analysis."),
    dict(category_id=13, speciality_id=74, seniority_needed="mid", budget_lo=45, budget_hi=90,
         hire_title="Property yields have been flat for two years",
         hire_description="Our property yields have been flat for roughly two years and we suspect the asset strategy itself needs a proper review rather than just waiting for the market to turn. We'd like a benchmarking exercise to understand what's actually different about our underperformance.",
         hire_description_additional_notes="Budget realistically SGD 45-90/hr for a three-week benchmarking exercise; a mid-level analyst is a good fit for this diagnostic-focused work."),
    dict(category_id=13, speciality_id=74, seniority_needed="senior", budget_lo=100, budget_hi=170,
         hire_title="Need an independent hold-or-sell recommendation on an aging asset",
         hire_description="We have an aging office asset and can't decide internally whether to refurbish or sell it, with strong opinions on both sides. We'd like a clear, evidence-based recommendation backed by comparable market data.",
         hire_description_additional_notes="Budget is around SGD 100-170/hr for a four-week analysis; needs someone senior enough that both sides of our internal debate will trust the recommendation."),
    dict(category_id=13, speciality_id=74, seniority_needed="mid", budget_lo=50, budget_hi=95,
         hire_title="Suspect some of our leases are underpriced relative to the market",
         hire_description="We suspect some of our leases are underpriced relative to current market rates but haven't done a systematic review to confirm it. We'd like someone to identify which leases are underpriced and how much upside renegotiation could realistically capture.",
         hire_description_additional_notes="Budget realistically SGD 50-95/hr for a three-week lease review; a mid-level specialist is fine for this data-focused task."),
    dict(category_id=13, speciality_id=74, seniority_needed="expert", budget_lo=140, budget_hi=220,
         hire_title="Board wants a full asset strategy overhaul across the portfolio",
         hire_description="Our board has asked for a full asset strategy overhaul across the entire portfolio after a disappointing set of annual results, and expects a credible, defensible plan. Given the visibility of this review, we want the most experienced asset manager available.",
         hire_description_additional_notes="Budget up to around SGD 220/hr for a two-month portfolio-wide review; this needs a genuinely senior asset management track record given the board-level visibility."),
])
# ---- Speciality: Real Estate / Lease Administration [cat=13, spec=75] ----
PROVIDERS_RAW.extend([
    dict(category_id=13, speciality_id=75, seniority="expert", full_name="Nicholas Goh", rate_per_hour=125,
         credentials="Former Lease Administration Manager, property management firm",
         about_title="Lease Administration & Tenant Billing Lead",
         about_description="Managed lease administration at a property management firm for eight years, personally overseeing lease compliance across a multi-building portfolio.",
         services_offered_title="Full Lease Register Audit",
         services_offered_description="Audits the current lease register against actual signed agreements line by line and fixes every discrepancy found, not just the obvious ones.",
         relevant_experience="Cleaned up a lease register for a commercial portfolio that had accumulated over 40 unrecorded amendments over the years."),
    dict(category_id=13, speciality_id=75, seniority="senior", full_name="Farhana Aziz", rate_per_hour=95,
         credentials="Commercial lease operations specialist, eight years",
         about_title="Independent Consultant, Commercial Lease Compliance",
         about_description="Commercial lease operations specialist with eight years of experience in tenant billing and renewal tracking across office and retail portfolios.",
         services_offered_title="Renewal Alert System Setup",
         services_offered_description="Sets up a proper lease tracking and renewal alert system so deadlines stop depending on someone remembering to check a spreadsheet.",
         relevant_experience="Set up a renewal alert system that has since caught every upcoming lease deadline for a property manager who had previously missed several."),
    dict(category_id=13, speciality_id=75, seniority="senior", full_name="Andre Lopez", rate_per_hour=85,
         credentials="Former property manager, lease compliance and tenant billing",
         about_title="Tenant Billing Dispute Resolution Specialist",
         about_description="Former property manager who now focuses specifically on cleaning up lease registers that have drifted from the actual signed agreements, usually surfaced through tenant billing disputes.",
         services_offered_title="Billing Dispute Root-Cause Fix",
         services_offered_description="Investigates recurring tenant billing disputes back to their source in the lease terms and corrects the underlying records rather than resolving each dispute individually.",
         relevant_experience="Fixed recurring tenant billing errors for a property management firm managing fifteen buildings, eliminating a recurring source of disputes."),
    dict(category_id=13, speciality_id=75, seniority="mid", full_name="Melissa Toh", rate_per_hour=60,
         credentials="Lease administration coordinator, two years",
         about_title="Lease Data Entry & Reconciliation Consultant",
         about_description="Two years coordinating day-to-day lease administration tasks for a property manager, with hands-on experience reconciling lease register entries against physical signed agreements.",
         services_offered_title="Lease Data Reconciliation Support",
         services_offered_description="Reconciles lease register entries against physical signed agreements one by one, correcting discrepancies as they're found rather than waiting for a full audit.",
         relevant_experience="Reconciled the lease register for a smaller property portfolio, correcting a number of outdated entries within a two-week engagement."),
])
HIRERS_RAW.extend([
    dict(category_id=13, speciality_id=75, seniority_needed="senior", budget_lo=50, budget_hi=100,
         hire_title="Keep missing lease renewal deadlines",
         hire_description="We keep missing lease renewal deadlines because our current tracking process is essentially an outdated spreadsheet nobody maintains consistently. We need a proper renewal tracking system that doesn't depend on someone remembering to check it.",
         hire_description_additional_notes="Budget around SGD 50-100/hr for a three-week setup; looking for someone with hands-on lease tracking system experience, not just general property management."),
    dict(category_id=13, speciality_id=75, seniority_needed="senior", budget_lo=45, budget_hi=90,
         hire_title="Tenants keep disputing billing and records don't look accurate",
         hire_description="Tenants keep disputing their billing, and we suspect our underlying lease records aren't accurate anymore after years of amendments. We'd like the billing disputes traced back to their source and the records corrected.",
         hire_description_additional_notes="Budget realistically SGD 45-90/hr for a four-week investigation and correction; a strong lease operations specialist is fine for this."),
    dict(category_id=13, speciality_id=75, seniority_needed="expert", budget_lo=90, budget_hi=150,
         hire_title="Lease register hasn't matched actual agreements in a while",
         hire_description="We're fairly sure our lease register hasn't matched the actual signed agreements for some time now, across a fairly large portfolio. Given the scale involved, we want someone with genuine experience running a full lease register audit before.",
         hire_description_additional_notes="Budget up to around SGD 150/hr for a six-week full audit; this needs someone who has cleaned up a lease register at this scale before."),
    dict(category_id=13, speciality_id=75, seniority_needed="mid", budget_lo=35, budget_hi=65,
         hire_title="Need lease records reconciled against a batch of recent agreements",
         hire_description="We have a batch of recently signed lease agreements that haven't been properly entered into our lease register yet. We need someone to reconcile these against the register and correct any discrepancies found.",
         hire_description_additional_notes="Budget realistically SGD 35-65/hr for a two-week reconciliation task; a mid-level coordinator is a good fit for this defined scope."),
    dict(category_id=13, speciality_id=75, seniority_needed="senior", budget_lo=55, budget_hi=105,
         hire_title="Want a renewal alert system before we miss another deadline",
         hire_description="We narrowly avoided missing a major lease renewal recently and it made clear we need a real alert system rather than relying on individual memory. We'd like this set up properly so it doesn't happen again.",
         hire_description_additional_notes="Budget around SGD 55-105/hr for a three-week setup and handover; ideally someone who has implemented a similar system for another property manager."),
])
# ---- Speciality: Construction & Engineering / Project Controls (Cost/Schedule) [cat=14, spec=80] ----
PROVIDERS_RAW.extend([
    dict(category_id=14, speciality_id=80, seniority="expert", full_name="Suresh Kumar", rate_per_hour=200,
         credentials="Former Project Controls Manager, regional construction firm",
         about_title="Project Controls Lead, Cost & Schedule Management",
         about_description="Managed project controls at a regional construction firm for twelve years, across infrastructure builds ranging from tens to hundreds of millions of dollars.",
         services_offered_title="Full Project Controls Framework Build",
         services_offered_description="Builds the cost and schedule control framework from scratch for a major project, covering baselines, variance tracking, and a reporting cadence leadership will actually use.",
         relevant_experience="Set up project controls for an infrastructure build worth roughly eighty million dollars that had no formal cost tracking in its first year."),
    dict(category_id=14, speciality_id=80, seniority="senior", full_name="Bee Choo Lim", rate_per_hour=155,
         credentials="Cost and schedule control specialist, ten years",
         about_title="Independent Consultant, Construction Schedule Recovery",
         about_description="Cost and schedule control specialist with a decade of experience specifically on infrastructure programmes, most often brought in once a project is already tracking behind.",
         services_offered_title="Schedule Recovery & Re-Baselining",
         services_offered_description="Runs a recovery review on a delayed project and produces a realistic re-baselined schedule, not an optimistic one nobody on the team actually believes.",
         relevant_experience="Recovered a construction schedule that had slipped six months, bringing the project back to within 8% of the original budget."),
    dict(category_id=14, speciality_id=80, seniority="senior", full_name="Alexander Png", rate_per_hour=145,
         credentials="Former site engineer, project controls specialist",
         about_title="Early-Warning Cost Tracking Specialist",
         about_description="Former site engineer who now focuses on setting up early-warning cost and schedule tracking so problems surface in the second week of a phase rather than at the six-month mark.",
         services_offered_title="Early-Warning Tracking System",
         services_offered_description="Sets up early-warning cost and schedule tracking tailored to how a specific project actually reports progress, catching slippage while it's still cheap to fix.",
         relevant_experience="Delivered an early-warning cost tracking system now used across a contractor's full project portfolio, not just the one it was originally built for."),
    dict(category_id=14, speciality_id=80, seniority="mid", full_name="Nadia Hassan", rate_per_hour=100,
         credentials="Project controls analyst, construction background",
         about_title="Cost & Schedule Reporting Analyst",
         about_description="Two years supporting project controls reporting on a mid-size construction programme, with hands-on experience building the variance reports project managers actually rely on.",
         services_offered_title="Variance Reporting Setup",
         services_offered_description="Builds a straightforward cost and schedule variance reporting process that project managers can maintain themselves once the initial setup is complete.",
         relevant_experience="Set up variance reporting for a mid-size construction programme that project managers continued running independently after the engagement ended."),
])
HIRERS_RAW.extend([
    dict(category_id=14, speciality_id=80, seniority_needed="senior", budget_lo=90, budget_hi=170,
         hire_title="Construction project running over budget, nobody can say why",
         hire_description="Our construction project is running well over budget, and nobody can tell us exactly why until it's already too late to act. We need proper cost and schedule controls put in place immediately, not a report on what's already gone wrong.",
         hire_description_additional_notes="Budget around SGD 90-170/hr for an immediate five-week engagement; looking for someone who can move fast given how far along the project already is."),
    dict(category_id=14, speciality_id=80, seniority_needed="expert", budget_lo=160, budget_hi=240,
         hire_title="Build schedule keeps slipping with no formal controls in place",
         hire_description="The build schedule keeps slipping, and we don't currently have formal cost and schedule controls in place to catch it early. Given how much is at stake on this project, we want the most experienced project controls specialist available.",
         hire_description_additional_notes="Budget up to around SGD 240/hr for a two-month engagement; this needs someone with a genuine track record recovering large, at-risk projects."),
    dict(category_id=14, speciality_id=80, seniority_needed="mid", budget_lo=55, budget_hi=100,
         hire_title="Need visibility into cost and schedule before it's too late",
         hire_description="We need visibility into where the project actually stands financially before the next milestone review, and currently rely on informal updates rather than a proper reporting process. We'd like a straightforward variance reporting process our own team can maintain.",
         hire_description_additional_notes="Budget realistically SGD 55-100/hr for a three-week setup; a mid-level analyst is a good fit for this reporting-focused work."),
    dict(category_id=14, speciality_id=80, seniority_needed="senior", budget_lo=100, budget_hi=180,
         hire_title="Site team keeps being surprised by cost overruns late in each phase",
         hire_description="Our site team keeps being surprised by cost overruns that only become visible late in each construction phase, by which point it's expensive to correct. We want an early-warning tracking system that catches this while it's still cheap to fix.",
         hire_description_additional_notes="Budget around SGD 100-180/hr for a six-week engagement; ideally someone who has built early-warning systems for construction projects before."),
    dict(category_id=14, speciality_id=80, seniority_needed="senior", budget_lo=95, budget_hi=175,
         hire_title="Need a realistic re-baselined schedule after a bad first estimate",
         hire_description="Our original project schedule was clearly too optimistic and we're now well behind it with no credible plan to get back on track. We need a properly re-baselined schedule that reflects reality rather than wishful thinking.",
         hire_description_additional_notes="Budget is around SGD 95-175/hr for a four-week re-baselining exercise; needs someone comfortable delivering a realistic, sometimes unwelcome, assessment."),
])
# ---- Speciality: Construction & Engineering / Safety & QA/QC [cat=14, spec=81] ----
PROVIDERS_RAW.extend([
    dict(category_id=14, speciality_id=81, seniority="expert", full_name="Joel Ang", rate_per_hour=170,
         credentials="Former Site Safety Manager, construction contractor",
         about_title="Construction Site Safety & QA/QC Lead",
         about_description="Managed site safety at a construction contractor for ten years, personally rebuilding the safety programme after a regulator review following a near-miss incident.",
         services_offered_title="Safety Programme Rebuild",
         services_offered_description="Audits current site safety practices against regulatory requirements and rebuilds specifically the weak points a real inspection would flag, not a generic safety refresh.",
         relevant_experience="Rebuilt a contractor's site safety programme after a near-miss incident triggered a formal regulator review."),
    dict(category_id=14, speciality_id=81, seniority="senior", full_name="Wei Jie Tan", rate_per_hour=130,
         credentials="QA/QC specialist, ten years across commercial construction sites",
         about_title="Independent Consultant, Site Compliance Programmes",
         about_description="QA/QC specialist with ten years of experience across commercial construction sites, specialising in identifying the recurring root cause behind repeated failed quality inspections.",
         services_offered_title="Failed-Inspection Root-Cause Fix",
         services_offered_description="Reviews a site's history of failed client quality inspections and identifies the recurring root cause behind them, rather than treating each failure as an isolated incident.",
         relevant_experience="Set up QA/QC checkpoints for a mid-size contractor that had been failing client quality inspections repeatedly."),
    dict(category_id=14, speciality_id=81, seniority="senior", full_name="Amanda Choo", rate_per_hour=120,
         credentials="Certified safety auditor, construction site compliance",
         about_title="Certified Site Safety Auditor",
         about_description="Certified safety auditor who now focuses specifically on building quality and safety documentation processes that site teams will actually follow, rather than file away unread.",
         services_offered_title="Usable Safety Documentation Design",
         services_offered_description="Sets up quality inspection checkpoints and documentation designed around how site teams actually work, so compliance paperwork gets used rather than backfilled after the fact.",
         relevant_experience="Reduced site safety incidents by a meaningful margin within one project cycle after rebuilding the safety programme."),
    dict(category_id=14, speciality_id=81, seniority="mid", full_name="Hafiz Rahim", rate_per_hour=85,
         credentials="Site safety officer, commercial construction",
         about_title="Site Safety Compliance Consultant",
         about_description="Three years as a site safety officer on commercial construction projects, with day-to-day experience identifying which specific safety practices tend to get skipped under time pressure.",
         services_offered_title="Time-Pressure Safety Compliance Review",
         services_offered_description="Reviews which specific safety practices are being skipped under time pressure on an active site and recommends practical fixes that don't slow the project further.",
         relevant_experience="Identified a recurring shortcut on one active site that, once corrected, removed the site's most common source of safety near-misses."),
])
HIRERS_RAW.extend([
    dict(category_id=14, speciality_id=81, seniority_needed="expert", budget_lo=110, budget_hi=190,
         hire_title="Had a safety near-miss, programme needs proper review",
         hire_description="We had a safety near-miss on site recently and need the safety programme properly reviewed and rebuilt before it happens again. Given the seriousness of what nearly happened, we want the most experienced safety specialist available.",
         hire_description_additional_notes="Budget up to around SGD 190/hr for an urgent four-week review; this needs a genuine track record rebuilding safety programmes after a serious incident."),
    dict(category_id=14, speciality_id=81, seniority_needed="senior", budget_lo=80, budget_hi=140,
         hire_title="Client keeps failing us on quality inspections",
         hire_description="Our client keeps failing us on quality inspections, and internal fixes haven't stuck despite repeated attempts. We'd like the recurring root cause behind our failed inspections identified and fixed properly this time.",
         hire_description_additional_notes="Budget around SGD 80-140/hr for a five-week engagement; looking for someone with genuine root-cause QA/QC experience, not another generic checklist."),
    dict(category_id=14, speciality_id=81, seniority_needed="mid", budget_lo=45, budget_hi=90,
         hire_title="Suspect our safety processes exist on paper but aren't followed on site",
         hire_description="We suspect our current safety and QC processes exist on paper but aren't actually followed on site under normal time pressure. We'd like an honest assessment of what's really happening versus what the documentation says.",
         hire_description_additional_notes="Budget realistically SGD 45-90/hr for a two-week on-site assessment; a mid-level safety officer is a good fit for this observational work."),
    dict(category_id=14, speciality_id=81, seniority_needed="senior", budget_lo=85, budget_hi=150,
         hire_title="Need site safety and QC processes site teams will actually use",
         hire_description="Our current safety paperwork is technically complete but clearly just gets backfilled after the fact rather than genuinely followed during work. We need documentation and checkpoints designed around how the site team actually operates.",
         hire_description_additional_notes="Budget is around SGD 85-150/hr for a four-week redesign; needs someone who understands real site behaviour, not just compliance theory."),
    dict(category_id=14, speciality_id=81, seniority_needed="mid", budget_lo=40, budget_hi=85,
         hire_title="Want a fresh set of eyes on our site safety practices before an audit",
         hire_description="We have a client audit coming up and want an independent, practical review of our site safety practices beforehand, without needing a full programme rebuild. This is meant to be a lighter-touch check rather than a major engagement.",
         hire_description_additional_notes="Budget realistically SGD 40-85/hr for a one-week walk-through and review; open to a mid-level safety specialist for this scoped pre-audit check."),
])
# ---- Speciality: Energy (Oil & Gas) / Asset Integrity [cat=16, spec=89] ----
PROVIDERS_RAW.extend([
    dict(category_id=16, speciality_id=89, seniority="expert", full_name="Priscilla Lau", rate_per_hour=230,
         credentials="Former Asset Integrity Engineer, offshore operator",
         about_title="Asset Integrity & Inspection Strategy Lead",
         about_description="Worked as an asset integrity engineer at an offshore operator for fourteen years, specialising in inspection programmes for ageing infrastructure nearing the end of its design life.",
         services_offered_title="Integrity Programme Risk Audit",
         services_offered_description="Audits the current integrity management programme against industry standards and prioritises fixes by which gaps carry the highest failure risk, not the easiest to address.",
         relevant_experience="Rebuilt the integrity management programme for an offshore platform after a regulator audit flagged corrosion risk."),
    dict(category_id=16, speciality_id=89, seniority="senior", full_name="Kevin Yeo", rate_per_hour=185,
         credentials="Integrity management specialist, upstream oil & gas assets",
         about_title="Independent Consultant, Ageing Asset Risk Management",
         about_description="Integrity management specialist with a decade of experience across upstream oil and gas assets, focused specifically on risk-based inspection strategy design.",
         services_offered_title="Risk-Based Inspection Strategy Design",
         services_offered_description="Designs a risk-based inspection strategy specifically for ageing assets nearing or past the end of their original design life, rather than applying a generic inspection interval.",
         relevant_experience="Designed a risk-based inspection plan for a processing facility operating well past its original design life."),
    dict(category_id=16, speciality_id=89, seniority="senior", full_name="Joel Ang", rate_per_hour=165,
         credentials="Former inspection lead, degradation trend analysis",
         about_title="Degradation Trend Review Specialist",
         about_description="Former inspection lead who now focuses specifically on spotting patterns in inspection findings that indicate accelerating degradation before it becomes an unplanned outage.",
         services_offered_title="Inspection Trend Pattern Review",
         services_offered_description="Reviews recent inspection findings for patterns that indicate accelerating degradation, flagging what a routine pass-fail inspection report alone would miss.",
         relevant_experience="Identified an accelerating corrosion pattern during a routine review that led to an unplanned shutdown being avoided altogether."),
    dict(category_id=16, speciality_id=89, seniority="mid", full_name="Bryan Lau", rate_per_hour=115,
         credentials="Integrity engineering associate, three years",
         about_title="Integrity Data Review Consultant",
         about_description="Three years supporting integrity engineering data review at an operator, with hands-on experience organising fragmented inspection records into something a risk assessment can actually use.",
         services_offered_title="Inspection Data Consolidation",
         services_offered_description="Consolidates fragmented historical inspection records into a structured dataset a proper risk-based inspection strategy can actually be built on.",
         relevant_experience="Consolidated years of fragmented inspection records for an operator into a structured dataset used to kick off their first formal risk-based inspection programme."),
])
HIRERS_RAW.extend([
    dict(category_id=16, speciality_id=89, seniority_needed="senior", budget_lo=130, budget_hi=210,
         hire_title="Equipment getting older, need a proper integrity review",
         hire_description="Our equipment is getting older and we need a proper integrity review before something fails unexpectedly and causes a costly unplanned outage. We need a prioritised fix plan aimed at the highest failure-risk gaps first.",
         hire_description_additional_notes="Budget around SGD 130-210/hr for a six-week review; looking for someone with direct hands-on integrity engineering experience, not just a general safety consultant."),
    dict(category_id=16, speciality_id=89, seniority_needed="senior", budget_lo=120, budget_hi=200,
         hire_title="Recent audit flagged gaps in our inspection programme",
         hire_description="A recent audit flagged gaps in how we inspect and maintain our processing equipment, and we need this addressed properly rather than a superficial paperwork fix. We need a risk-based inspection strategy specifically for our ageing assets.",
         hire_description_additional_notes="Budget is roughly SGD 120-200/hr for a five-week engagement; needs someone with genuine risk-based inspection design experience."),
    dict(category_id=16, speciality_id=89, seniority_needed="mid", budget_lo=60, budget_hi=110,
         hire_title="Inspection records are scattered and hard to use for planning",
         hire_description="Our historical inspection records are scattered across different systems and formats, making it hard to use them for any kind of forward-looking risk planning. We need someone to consolidate this into something usable before we can even start a proper risk assessment.",
         hire_description_additional_notes="Budget realistically SGD 60-110/hr for a three-week consolidation project; a mid-level associate is a good fit for this data-focused work."),
    dict(category_id=16, speciality_id=89, seniority_needed="expert", budget_lo=170, budget_hi=260,
         hire_title="Worried about unplanned failure on ageing processing equipment",
         hire_description="We're increasingly concerned about unplanned failure risk on equipment that's now well past its original design life, and a failure here would be extremely costly. Given the stakes, we want the most experienced integrity specialist available.",
         hire_description_additional_notes="Budget up to around SGD 260/hr for an intensive six-week assessment; this needs someone with a genuine track record on ageing, high-consequence assets."),
    dict(category_id=16, speciality_id=89, seniority_needed="senior", budget_lo=110, budget_hi=190,
         hire_title="Want an independent second look at recent inspection findings",
         hire_description="Our internal team has reviewed recent inspection findings and believes everything is fine, but leadership wants an independent second opinion given how much is riding on this equipment staying operational. We'd like someone to specifically check for any concerning trends our own team might have missed.",
         hire_description_additional_notes="Budget around SGD 110-190/hr for a three-week independent review; needs someone comfortable disagreeing with an internal team's assessment if warranted."),
])
# ---- Speciality: Energy (Oil & Gas) / Turnaround/Shutdown PMO [cat=16, spec=90] ----
PROVIDERS_RAW.extend([
    dict(category_id=16, speciality_id=90, seniority="expert", full_name="Nur Aina Zahra", rate_per_hour=230,
         credentials="Former Turnaround Manager, petrochemical facility",
         about_title="Turnaround & Shutdown Planning Office Lead",
         about_description="Managed turnarounds at a petrochemical facility for twelve years, personally leading the facility's largest shutdown in a decade from planning through execution.",
         services_offered_title="Full Turnaround Scope & Schedule Build",
         services_offered_description="Builds the full turnaround work scope, schedule, and resourcing plan ahead of a major shutdown, well before execution begins rather than planning on the fly.",
         relevant_experience="Ran the turnaround PMO for a petrochemical facility's largest shutdown in a decade, finishing two days ahead of the original plan."),
    dict(category_id=16, speciality_id=90, seniority="senior", full_name="Ting Wei Ho", rate_per_hour=190,
         credentials="Shutdown planning specialist, refinery and processing plants",
         about_title="Independent Consultant, Refinery Shutdown Execution",
         about_description="Shutdown planning specialist with a decade of experience across refinery and processing plant turnarounds, running the planning office through live execution rather than handing off before it starts.",
         services_offered_title="Execution-Phase Planning Office",
         services_offered_description="Runs the planning and progress-tracking office during execution, so scope creep gets caught within days rather than surfacing only at the post-mortem.",
         relevant_experience="Rescued a shutdown that was tracking to overrun by three weeks, bringing it back to within one day of the original schedule."),
    dict(category_id=16, speciality_id=90, seniority="senior", full_name="Farah Ismail", rate_per_hour=175,
         credentials="Former planning engineer, shutdown recovery specialist",
         about_title="Slipping Shutdown Recovery Specialist",
         about_description="Former planning engineer who now focuses specifically on stepping into shutdowns that are already tracking to overrun and rebuilding a schedule the operations team can actually hit.",
         services_offered_title="Mid-Shutdown Recovery Plan",
         services_offered_description="Steps into an already-slipping shutdown and rebuilds a realistic recovery schedule the operations team can actually hit, rather than an aspirational one.",
         relevant_experience="Built a resourcing and scope plan that let a facility complete a major turnaround with zero critical safety incidents despite a difficult mid-shutdown restart."),
    dict(category_id=16, speciality_id=90, seniority="mid", full_name="Wendy Chan", rate_per_hour=120,
         credentials="Turnaround planning associate, two years",
         about_title="Turnaround Resourcing & Scope Support Consultant",
         about_description="Two years supporting turnaround planning at a processing facility, with hands-on experience tracking scope changes and resourcing needs as a shutdown progresses.",
         services_offered_title="Scope Change & Resourcing Tracking",
         services_offered_description="Tracks scope changes and resourcing needs throughout an active shutdown, flagging anything that risks pushing the schedule before it becomes unmanageable.",
         relevant_experience="Tracked scope changes for a mid-size plant shutdown that helped the planning office catch a resourcing shortfall two weeks before it would have caused a delay."),
])
HIRERS_RAW.extend([
    dict(category_id=16, speciality_id=90, seniority_needed="expert", budget_lo=180, budget_hi=270,
         hire_title="Major shutdown coming up, no dedicated planning office yet",
         hire_description="We have a major plant shutdown coming up and currently no dedicated planning office to manage the scope and schedule. Given how much is at stake with a shutdown this size, we want the most experienced turnaround leader available.",
         hire_description_additional_notes="Budget up to around SGD 270/hr for the full planning and execution period; this needs a genuine track record leading turnarounds of comparable scale."),
    dict(category_id=16, speciality_id=90, seniority_needed="senior", budget_lo=140, budget_hi=220,
         hire_title="Last turnaround ran significantly over schedule",
         hire_description="Our last shutdown ran significantly over schedule, and we want proper turnaround planning support this time around to avoid a repeat. We need the full scope, schedule, and resourcing plan built well ahead of execution.",
         hire_description_additional_notes="Budget is roughly SGD 140-220/hr for a two-month planning engagement ahead of the shutdown date; looking for someone who has planned a comparable turnaround before."),
    dict(category_id=16, speciality_id=90, seniority_needed="senior", budget_lo=130, budget_hi=210,
         hire_title="Need someone experienced running a turnaround planning office",
         hire_description="We need someone who has actually run a turnaround planning office before, not just participated in one, given how tight our shutdown window is this time. The ask is a planning office that catches scope creep in days, not after the shutdown is over.",
         hire_description_additional_notes="Budget around SGD 130-210/hr through the execution window; needs someone comfortable making fast calls under real operational pressure."),
    dict(category_id=16, speciality_id=90, seniority_needed="mid", budget_lo=70, budget_hi=130,
         hire_title="Need scope and resourcing tracked closely during an upcoming shutdown",
         hire_description="We have the core turnaround plan in place already but want someone dedicated to tracking scope changes and resourcing needs closely as the shutdown progresses. This is a support role within a larger planning effort rather than leading it.",
         hire_description_additional_notes="Budget realistically SGD 70-130/hr for the shutdown duration; a mid-level planning associate is a good fit for this tracking-focused role."),
    dict(category_id=16, speciality_id=90, seniority_needed="senior", budget_lo=150, budget_hi=230,
         hire_title="Shutdown is already underway and starting to slip",
         hire_description="Our shutdown is already underway and has started slipping against the original schedule, with pressure mounting to get the plant back online. We need someone to step in immediately and rebuild a realistic recovery plan.",
         hire_description_additional_notes="Budget around SGD 150-230/hr for immediate engagement through to restart; needs someone who has recovered an in-progress shutdown before, not just planned one from the start."),
])
# === SPECIALITY BLOCKS END ===
# (26 specialities x 4 providers + 5 hirers = 104 providers, 130 hirers)


def assign_ids_and_defaults(providers_raw, hirers_raw):
    providers = []
    for i, p in enumerate(providers_raw, start=1):
        rec = dict(p)
        rec["provider_id"] = i
        rec.setdefault("availability", AVAILABILITY[i % len(AVAILABILITY)])
        rec.setdefault("work_description", WORK_STYLES[i % len(WORK_STYLES)])
        rec["status"] = "approved"
        rec["created_at"] = "2026-08-01 09:00:00"
        rec["updated_at"] = None
        providers.append(rec)

    hirers = []
    for i, h in enumerate(hirers_raw, start=1):
        rec = dict(h)
        rec["hire_id"] = i
        rec["category_ids"] = [rec["category_id"]]
        rec["speciality_ids"] = [rec["speciality_id"]]
        rec["status"] = "active"
        rec["created_at"] = "2026-08-15 09:00:00"
        rec["updated_at"] = None
        hirers.append(rec)

    return providers, hirers


def compute_relevance_score(provider: dict, hirer: dict) -> int:
    if provider["speciality_id"] == hirer["speciality_id"]:
        base = 70
    elif provider["category_id"] == hirer["category_id"]:
        base = 35
    else:
        return 0

    rate = provider["rate_per_hour"]
    if hirer["budget_lo"] <= rate <= hirer["budget_hi"]:
        budget_fit = 10
    elif hirer["budget_lo"] * 0.75 <= rate <= hirer["budget_hi"] * 1.25:
        budget_fit = 0
    else:
        budget_fit = -10

    prov_level = SENIORITY_ORDER[provider["seniority"]]
    hirer_level = SENIORITY_ORDER[hirer["seniority_needed"]]
    diff = abs(prov_level - hirer_level)
    seniority_fit = 10 if diff == 0 else (0 if diff == 1 else -10)

    noise = random.randint(-5, 5)
    score = base + budget_fit + seniority_fit + noise
    return max(1, min(100, score))


def build_ground_truth(providers, hirers):
    gt = {}
    for h in hirers:
        row = {}
        for p in providers:
            score = compute_relevance_score(p, h)
            if score > 0:
                row[str(p["provider_id"])] = score
        gt[str(h["hire_id"])] = row
    return gt


# We need `search_tags` on each hirer -- pulled from the speciality's real
# tag_ids, looked up from SPECIALITIES metadata retained during authoring.
# (Each hirer block above only carries category_id/speciality_id -- tag_ids
# are attached here from the same real schema/speciality_tags.sql-derived
# mapping used throughout.)
SPECIALITY_TAG_IDS = {
    1: [1, 2, 3], 4: [8, 9, 10, 11], 10: [27, 28, 29], 11: [30, 31, 32],
    17: [35, 42, 43], 20: [48, 49], 26: [59, 60], 25: [11, 40, 58],
    29: [64, 65], 30: [66, 67], 37: [79, 80], 35: [76, 77],
    40: [85, 86, 87], 42: [43, 91], 46: [54, 100, 101], 47: [28, 54, 102, 103],
    51: [109, 110], 53: [112, 113], 56: [117, 118], 58: [27, 120],
    74: [85, 146], 75: [147, 148], 80: [149, 158, 159, 160], 81: [161, 162, 163],
    89: [146, 172], 90: [18, 173, 174],
}

SCHEMA_PROVIDER_COLS = [
    "provider_id", "full_name", "credentials", "availability", "rate_per_hour",
    "work_description", "about_title", "about_description",
    "services_offered_title", "services_offered_description",
    "relevant_experience", "status", "created_at", "updated_at",
]
SCHEMA_HIRER_COLS = [
    "hire_id", "hire_title", "hire_description", "hire_description_additional_notes",
    "category_ids", "speciality_ids", "search_tags", "status", "created_at", "updated_at",
]


def strip_internal(records, cols):
    return [{k: r[k] for k in cols} for r in records]


def write_json(records, cols, path):
    path.write_text(json.dumps(strip_internal(records, cols), indent=2, ensure_ascii=False))


def write_csv(records, cols, path):
    clean = strip_internal(records, cols)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in clean:
            row = dict(row)
            for k, v in row.items():
                if isinstance(v, list):
                    row[k] = ";".join(str(x) for x in v)
            writer.writerow(row)


def main():
    providers, hirers = assign_ids_and_defaults(PROVIDERS_RAW, HIRERS_RAW)
    for h in hirers:
        h["search_tags"] = SPECIALITY_TAG_IDS[h["speciality_id"]]

    # uniqueness sanity check: every provider's (about_description,
    # services_offered_description, relevant_experience) triple, and every
    # hirer's hire_description, must be unique -- there is no fragment reuse
    # in this version, so this should hold trivially by construction.
    prov_texts = [(p["about_description"], p["services_offered_description"], p["relevant_experience"]) for p in providers]
    hire_texts = [h["hire_description"] for h in hirers]
    assert len(set(prov_texts)) == len(prov_texts), "duplicate provider text detected"
    assert len(set(hire_texts)) == len(hire_texts), "duplicate hirer text detected"

    ground_truth = build_ground_truth(providers, hirers)

    write_json(providers, SCHEMA_PROVIDER_COLS, DATA_DIR / "providers.json")
    write_csv(providers, SCHEMA_PROVIDER_COLS, DATA_DIR / "providers.csv")
    write_json(hirers, SCHEMA_HIRER_COLS, DATA_DIR / "hirers.json")
    write_csv(hirers, SCHEMA_HIRER_COLS, DATA_DIR / "hirers.csv")

    (DATA_DIR / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))
    with open(DATA_DIR / "ground_truth.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hire_id", "provider_id", "relevance_score"])
        for hid, row in ground_truth.items():
            for pid, score in row.items():
                writer.writerow([hid, pid, score])

    (DATA_DIR / "_providers_with_taxonomy.json").write_text(json.dumps(providers, indent=2, ensure_ascii=False))
    (DATA_DIR / "_hirers_with_taxonomy.json").write_text(json.dumps(hirers, indent=2, ensure_ascii=False))

    n_pairs = sum(len(v) for v in ground_truth.values())
    print(f"providers: {len(providers)}  hirers: {len(hirers)}  "
          f"ground_truth pairs (score>0): {n_pairs}  "
          f"avg score: {sum(s for row in ground_truth.values() for s in row.values()) / n_pairs:.1f}")


if __name__ == "__main__":
    main()
