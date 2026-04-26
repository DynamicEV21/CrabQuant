# CrabQuant API Budget Configuration
# Created: 2026-04-25
# Last Updated: 2026-04-25
# Source: https://docs.bigmodel.cn/cn/coding-plan

## ⚠️ CRITICAL: It's NOT Token Limits — It's PROMPT Limits

The GLM Coding Plan limits by **prompts per 5 hours**, not raw token count.
Each cron run = 1 prompt. Each prompt internally calls the model ~15-20 times.

## Provider
- **Provider:** Zhipu AI (z.ai) — GLM Coding Plan Pro
- **API Base:** https://api.z.ai/api/coding/paas/v4/v1
- **Plan:** Pro subscription ($X/month)

## Hard Limits (Pro Plan)

| Limit | Amount | Resets |
|-------|--------|--------|
| **Per 5 hours** | ~400 prompts | Rolling 5h window |
| **Per week** | ~2000 prompts | 7-day cycle from subscription |

## Multiplier System

GLM-5-Turbo is a "high-tier model" — each prompt costs MORE than 1x:

| Time Period | Multiplier | Effective Prompts per 5h |
|-------------|-----------|--------------------------|
| Off-peak (now until Apr 30) | **1x** (limited promo!) | 400 |
| Off-peak (after Apr 30) | **2x** | 200 |
| Peak (14:00-18:00 UTC+8 / 02:00-06:00 PST) | **3x** | ~133 |

**⚠️ OpenClaw is SECONDARY PRIORITY** — Coding Agent tasks (Claude Code, etc.) get priority. Under high load, OpenClaw tasks get queued/rate-limited automatically.

## Current Cron Schedule (Budget-Optimized)

| Agent | Frequency | Prompts/5h | Prompts/Week | Light Ctx |
|-------|-----------|-----------|--------------|-----------|
| crabquant-wave | Every 15min | 20 | 672 | ✅ |
| crabquant-invent | Every 2h | 2-3 | 84 | ❌ |
| crabquant-validate | Every 2h | 2-3 | 84 | ✅ |
| crabquant-meta | Every 3h | 1-2 | 56 | ✅ |
| **CrabQuant Total** | | **~27** | **~896** | |
| main (CodeCrab) | Heartbeats + chat | ~10 | ~336 | ❌ |
| bella | Weekday crons | ~2 | ~20 | ❌ |
| **Grand Total** | | **~39** | **~1252** | |

### At 2x multiplier (after April 30):
- 39 prompts × 2 = **78 effective per 5h** (well under 400)
- Weekly: 1252 × 2 = **2504 effective** (over 2000 weekly limit!)

### At 1x multiplier (current, until April 30):
- 39 prompts × 1 = **39 effective per 5h** (well under 400)
- Weekly: **1252** (under 2000 weekly limit) ✅

## Weekly Budget is the Real Constraint

After April 30, the weekly budget (2000) becomes the bottleneck:
- 2000 / 2x = 1000 real prompts/week
- Current schedule: ~1252 real prompts/week = **25% over budget**

### If Over Weekly Budget
Reduce wave to every 20min → 504/week → total ~1028/week (under budget with 2x)

## Emergency Procedures

### If you hit rate limits (429 errors, timeouts):

**Quick fix — disable non-essential crons:**
```bash
openclaw cron edit crabquant-meta --disabled
openclaw cron edit crabquant-validate --disabled
openclaw cron edit crabquant-improve --disabled
# Keep only wave running
```

**Slower fix — reduce wave frequency:**
```bash
openclaw cron edit crabquant-wave --every 30m
```

**Full stop:**
```bash
openclaw cron edit crabquant-wave --disabled
openclaw cron edit crabquant-improve --disabled
openclaw cron edit crabquant-validate --disabled
openclaw cron edit crabquant-meta --disabled
```

### Recovery
- Wait for 5-hour window to reset
- Re-enable crons one at a time
- Check usage: https://www.bigmodel.cn/coding-plan/personal/usage

## Optimization Strategies

1. **--light-context** on wave/validate/meta → shorter prompts = less chance of multiplier
2. **Batch 20 combos per wave run** → 1 prompt does 20 combos worth of work
3. **NO_REPLY for no-op cycles** → saves output tokens (not prompt count though)
4. **Minimal prompt text** → reduces internal model calls per prompt
5. **Avoid peak hours** (02:00-06:00 PST) for non-urgent tasks
6. **GLM-4.7 for simple tasks** → if we add it as a model option, 1x multiplier always

## Future Improvement: Use GLM-4.7 for Wave

GLM-4.7 doesn't have the 2x/3x multiplier. The wave agent just runs a script —
it doesn't need GLM-5-Turbo's intelligence. If we switch wave to GLM-4.7:
- Wave cost: 1x instead of 2-3x
- Frees up budget for invent/meta to stay on GLM-5-Turbo
- Can run wave more frequently

To implement:
```bash
openclaw cron edit crabquant-wave --model zai/glm-4.7
```

## Throughput Summary

| Config | Wave Combos/Day | Prompts/Week | Effective/Week (2x) |
|--------|----------------|--------------|---------------------|
| Current (safe) | 1,920 | 1,252 | 2,504 |
| Wave every 20min | 1,440 | 1,028 | 2,056 |
| Wave on GLM-4.7 | 1,920 | 896 | 1,792 ✅ |

Current config works at 1x (until April 30). After that, consider switching wave to GLM-4.7 or reducing frequency.
