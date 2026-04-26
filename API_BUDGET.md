# CrabQuant API Budget Configuration
# Created: 2026-04-25
# Last Updated: 2026-04-25

## Provider
- **Provider:** Zhipu AI (z.ai) — GLM Coding Plan (Pro)
- **API Base:** https://api.z.ai/api/coding/paas/v4/v1
- **Plan:** Global Coding Plan — Pro subscription
- **Model:** glm-5-turbo (primary)

## Budget Estimates
- **Daily budget:** ~1.7M tokens/day (all agents combined)
- **Monthly estimate:** ~52M tokens/month
- **This is across ALL OpenClaw agents**, not just CrabQuant

## Per-Agent Breakdown (optimized)

| Agent | Frequency | Light Ctx | Daily Input | Daily Output |
|-------|-----------|-----------|-------------|--------------|
| crabquant-wave | Every 10min | ✅ | 216K | 144K |
| crabquant-invent | Every 30min | ❌ | 720K | 192K |
| crabquant-validate | Every 30min | ✅ | 96K | 72K |
| crabquant-meta | Every 1h | ✅ | 96K | 72K |
| main (CodeCrab) | Heartbeats | ❌ | 50K | 20K |
| bella | Weekday crons | ❌ | 40K | 15K |

## Token Optimization Strategies Used

1. **--light-context** on wave, validate, meta crons
   - Strips SOUL.md, IDENTITY.md, AGENTS.md from bootstrap
   - Cuts input tokens by ~50% for simple exec tasks
   - Only used where agent doesn't need personality (just runs scripts)

2. **Minimal prompts** for wave/validate
   - Shortest possible instructions
   - "Run this command. Output winners or NO_REPLY."
   - Avoids the agent generating lengthy explanations

3. **NO_REPLY** for no-op cycles
   - Wave agent outputs NO_REPLY when no winners found
   - Prevents output token waste on empty results

4. **Batch processing** in scripts
   - Wave does 20 combos per agent call (not 1)
   - One LLM call processes 20 combos = massive savings

5. **Reduced frequency for non-critical agents**
   - Validate: 30min (doesn't need to be real-time)
   - Meta: 1h (analysis is valuable but not urgent)
   - Only Invent stays at 30min (that's where intelligence matters)

## If You Hit Rate Limits

### Symptoms
- Cron runs show "error" or "timeout"
- 429 rate limit errors
- Slow responses from gateway

### Emergency Actions
1. **Reduce wave frequency:** `openclaw cron edit crabquant-wave --every 30m`
2. **Reduce invent frequency:** `openclaw cron edit crabquant-improve --every 1h`
3. **Disable meta:** `openclaw cron edit crabquant-meta --disabled`
4. **Disable validate:** `openclaw cron edit crabquant-validate --disabled`
5. **Full stop:** Disable all crabquant crons

### Recovery
- Wait 5-10 minutes
- Re-enable crons one at a time
- Monitor with `openclaw cron list | grep crabquant`

## Throughput vs Cost Tradeoff

| Config | Combos/Day | Strategies/Day | Tokens/Day |
|--------|-----------|----------------|------------|
| Aggressive (5min wave) | 5,760 | 48 | 3.2M |
| **Balanced (current)** | **2,880** | **48** | **1.7M** |
| Conservative (30min wave) | 960 | 24 | 800K |
| Minimal (1h wave) | 480 | 12 | 500K |

Current config is "Balanced" — good throughput without burning tokens.

## Important Notes
- The wave agent is the biggest token consumer despite light context
- This is because it fires most often (144 times/day)
- The actual Python script uses ZERO API tokens — only the agent wrapper costs tokens
- If OpenClaw ever supports raw exec crons (no LLM), wave tokens → 0
- Bella crons use separate agent, separate budget consideration
- Main agent (CodeCrab) conversations add variable cost
