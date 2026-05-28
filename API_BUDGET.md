# CrabQuant API Budget Configuration
# Created: 2026-04-25
# Last Updated: 2026-04-25
# Source: https://docs.bigmodel.cn/cn/coding-plan

## ⚠️ CRITICAL: It's PROMPT Limits, Not Token Limits

The GLM Coding Plan limits by **prompts per 5 hours**, not raw token count.
Each cron run = 1 prompt. Each prompt internally calls the model ~15-20 times.

## Provider
- **Provider:** Zhipu AI (z.ai) — GLM Coding Plan
- **API Base:** https://api.z.ai/api/coding/paas/v4/v1
- **Plan:** Grandfathered Pro (no weekly limit confirmed by user)

## Hard Limits

| Limit | Amount | Notes |
|-------|--------|-------|
| **Per 5 hours** | ~400 prompts | Rolling 5h window |
| **Per week** | N/A | Grandfathered plan — no weekly cap |

## Multiplier System

GLM-5-Turbo has a higher cost per prompt:

| Time Period | Multiplier | Notes |
|-------------|-----------|-------|
| Off-peak (now until Apr 30) | **1x** | Limited promo |
| Off-peak (after Apr 30) | **2x** | Standard |
| Peak (14:00-18:00 UTC+8 / 02:00-06:00 PST) | **3x** | |
| **GLM-4.7** | **1x always** | No multiplier! |

**⚠️ OpenClaw is SECONDARY PRIORITY** — Coding Agent tasks get priority first.

## Current Cron Setup (Optimized)

| Agent | Model | Frequency | Prompts/5h | Multiplier |
|-------|-------|-----------|-----------|------------|
| crabquant-wave | **GLM-4.7** | Every 15min | 20 | 1x |
| crabquant-invent | **GLM-5-Turbo** | Every 2h | 2-3 | 2-3x |
| crabquant-validate | **GLM-4.7** | Every 2h | 2-3 | 1x |
| crabquant-meta | **GLM-4.7** | Every 3h | 1-2 | 1x |
| **CrabQuant Total** | | | **~27** | **~33 effective** |
| main (CodeCrab) | GLM-5-Turbo | Chat/heartbeat | ~10 | 2-3x |
| bella | GLM-5-Turbo | Weekday crons | ~2 | 2-3x |
| **Grand Total** | | | **~39** | **~52 effective** |

**52 effective prompts per 5h — well under the 400 budget.**

## Key Optimization: GLM-4.7 for Script-Only Tasks

Wave, validate, and meta agents just run Python scripts — no creative intelligence needed.
Using GLM-4.7 (1x multiplier always) instead of GLM-5-Turbo (2-3x) saves massive budget.

Only the **invent** agent stays on GLM-5-Turbo because it actually needs to write novel code.

## Emergency Procedures

### If you hit rate limits:
```bash
# Quick — disable non-essential
openclaw cron edit crabquant-meta --disabled
openclaw cron edit crabquant-validate --disabled

# Reduce wave frequency
openclaw cron edit crabquant-wave --every 30m

# Full stop
openclaw cron edit crabquant-wave --disabled
openclaw cron edit crabquant-improve --disabled
openclaw cron edit crabquant-validate --disabled
openclaw cron edit crabquant-meta --disabled
```

### Recovery: Wait for 5h window reset, re-enable one at a time.

## Throughput

| Metric | Value |
|--------|-------|
| Wave combos/day | 1,920 (20 combos × 96 runs) |
| Inventions/day | 12 |
| Validations/day | 12 |
| Meta analyses/day | 8 |
| Full sweep time | ~616 combos ÷ 20/cycle × 15min ≈ **8 hours** |
