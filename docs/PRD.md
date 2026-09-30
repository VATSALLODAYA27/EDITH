# PRD: [Product name]

<!--
Product Requirements Document. The first file your agent reads.
It answers three things: what we are building, for whom, and what "done"
looks like. Fill every [bracket], then delete these comments.
Keep it under two pages. Longer than that is a spec, not a PRD.
-->

## 01 Product overview

| Field | Value |
| --- | --- |
| Product name | [Name] |
| Tagline | [One line, under 10 words: what it does, for whom] |
| Description | [2 to 3 sentences. What the product is, who uses it, the result they get.] |
| Stage | [Idea / MVP / Live] |
| Platform | [Web app / Mobile / Desktop / API] |

## 02 Problem

[The problem in the user's own words. What happens today, why it hurts,
what they do instead. One paragraph. No solution talk yet.]

Today, [user] has to [painful workaround] because [root cause].
This costs them [time / money / outcome].

## 03 Goal

[One sentence. The single outcome this product exists to deliver.]

Help [user] [achieve outcome] in [timeframe or effort] without [the pain].

## 04 Target users

Primary user:
- Who: [role, situation]
- Trigger: [the moment they go looking for this]
- Today they use: [current tool or workaround]
- They will switch because: [the one thing we do that it does not]

Secondary users (only if they change what we build):
- [Role]: [what they need from the product]

Not for:
- [Who we are explicitly not building for, and why]

## 05 Core features

Must ship in v1. Each feature has a "done when" so the agent knows when to stop.

| # | Feature | What the user can do | Done when |
| --- | --- | --- | --- |
| 1 | [Feature] | [Action the user takes] | [Observable result] |
| 2 | [Feature] | [Action the user takes] | [Observable result] |
| 3 | [Feature] | [Action the user takes] | [Observable result] |

Key user flow:
1. [User lands on / opens ...]
2. [They do ...]
3. [They get ...]

## 06 Success metrics

How we know it works. Numbers, not adjectives.

| Metric | Target | Measured by |
| --- | --- | --- |
| [Activation: users who complete the key flow] | [X%] | [Where the number comes from] |
| [Retention or repeat use] | [X] | [...] |
| [Business result: signups, revenue, time saved] | [X] | [...] |

## 07 Out of scope

Do not build these in v1, even if they look easy or obvious.

- [Feature or integration we are deliberately skipping]
- [Platform we are not supporting yet]
- [Nice-to-have that waits for v2]

## 08 Open questions

Not decided yet. The agent must ask before assuming an answer.

- [Question]
- [Question]
