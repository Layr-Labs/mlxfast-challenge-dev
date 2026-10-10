// Qwen 3.6 27B native-MTP speculative decode driver.
//
// A faithful Swift port of MTPLX's `generate_mtpa` (MTPLX/mtplx/generation.py
// L10176-10420), driving the fork's own native MTP head (Qwen35MTPModule, reached
// through the MTPCapable conformance: callWithHidden / mtpForwardWithHidden /
// makeMTPCache) as the drafter — no external draft model.
//
// Per round:
//   1. emit the pending primary; clamp depth to the remaining token budget
//   2. draft `cycleDepth` tokens from the head (one fresh head cache per round,
//      shared across the sub-steps — MTPLX mtp_cache_policy default "persistent";
//      each sub-step chains the head's own post-`mtp.norm` hidden)
//   3. snapshot the non-trimmable (GDN/recurrent) cache state, then verify
//      `[primary] + drafts` in ONE batched target forward
//   4. accept the longest common prefix (row i of the verify output is the target's
//      greedy continuation of verify input i, i.e. the truth for draft i)
//   5. full acceptance -> keep the verify state, next primary = argmax of row D.
//      otherwise -> roll back the whole verify window (trim all 1+D positions from
//      the KV caches AND restore the recurrent snapshot) and re-forward the
//      committed block `[primary] + acceptedDrafts`; its last logits row is the
//      next primary and its last hidden row feeds the next draft round.
//
// Why not the vendored DFlash rollback: Vendor/dflash-swiftlm's
// RecurrentRollbackCache rolls the GDN state forward by replaying an "innovation
// tape" that the GDN forward is supposed to hand it via recordTape()
// (RecurrentRollbackCache.swift:112 guards on `tape != nil`). Nothing in
// mlx-swift-lm or in the vendored code ever calls recordTape, so the tape is always
// nil and the cache silently degenerates to a pre-verify snapshot restore while
// DFlashRuntime.restoreTargetCacheAfterAcceptance still trims the KV caches to
// prefix+1+accepted — the 48 GDN layers and the 16 attention layers desync on every
// partial acceptance. MTPLX's snapshot + rollback + re-forward needs no tape.
//
// This targets Qwen 3.6 (config model_type qwen3_6_mtp; the qwen3_5 code symbols
// are the fork's architecture-family names that 3.6 uses internally).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

@main
struct Qwen36MTPDriver {
    /// Last-resort stop set if the checkpoint's config/tokenizer cannot be read.
    /// These are the observed Qwen 3.6 values (`<|im_end|>`, `<|endoftext|>`).
    static let fallbackStopTokens: Set<Int> = [248046, 248044]

    /// Resolve the stop set from the checkpoint the way MTPLX's `_default_stop_tokens`
    /// does (generation.py:3830-3839): union of `eos_token_id` / `pad_token_id` over the
    /// checkpoint config, accepting either a scalar or a list, plus the tokenizer's own
    /// EOS id. Falls back to the hardcoded ids only if nothing could be parsed.
    static func resolveStopTokens(dir: URL, tokenizer: any MLXLMCommon.Tokenizer)
        -> (Set<Int>, String)
    {
        var ids = Set<Int>()
        var sources: [String] = []
        for name in ["config.json", "generation_config.json"] {
            guard let data = try? Data(contentsOf: dir.appendingPathComponent(name)),
                let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            else { continue }
            for key in ["eos_token_id", "pad_token_id"] {
                switch root[key] {
                case let value as Int:
                    ids.insert(value)
                    sources.append("\(name):\(key)")
                case let values as [Any]:
                    let parsed = values.compactMap { $0 as? Int }
                    guard !parsed.isEmpty else { continue }
                    ids.formUnion(parsed)
                    sources.append("\(name):\(key)")
                default:
                    continue
                }
            }
        }
        if let eos = tokenizer.eosTokenId {
            ids.insert(eos)
            sources.append("tokenizer:eos")
        }
        if ids.isEmpty {
            return (fallbackStopTokens, "hardcoded fallback (config parse failed)")
        }
        return (ids, sources.joined(separator: ", "))
    }

    static func main() async {
        do { try await run() } catch { print("error: \(error)"); exit(1) }
    }

    /// Per-step progress marker — stderr, flushed, so a run visibly advances.
    static func progress(_ s: String) {
        FileHandle.standardError.write(Data((s + "\n").utf8))
    }

    static func run() async throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("usage: qwen36-mtp <merged-weights-dir> [--prompt TEXT] [--tokens N] "
                + "[--depth D] [--hidden post|pre] [--chat]")
            exit(2)
        }
        let dir = URL(fileURLWithPath: args[1])
        let prompt = flag(args, "--prompt") ?? "Write a short poem about Apple Silicon."
        let maxTokens = Int(flag(args, "--tokens") ?? "128") ?? 128
        // Depth 2 is the useful setting: measured on the M5 Max, depth 2 gives
        // 1.34-1.74x while depth 3 regresses to 0.71-0.91x (acceptance ~40%,
        // below the break-even for the extra verify width).
        let depth = Int(flag(args, "--depth") ?? "2") ?? 2   // draft tokens per round
        // Raw `encode` never emits the chat control tokens, so a raw-prompted run can
        // only ever end on the token budget. --chat wraps the prompt in the tokenizer's
        // chat template so the model can actually emit <|im_end|> and the EOS paths run.
        let useChat = args.contains("--chat")
        // MTPLX default: base_hidden_variant = mtp_hidden_variant = "post_norm"
        // (MTPContract, mtplx/mtp_patch.py:50-51).
        let postNorm = (flag(args, "--hidden") ?? "post") != "pre"

        // 1. Attach the 3.6 MTP head at load time.
        _qwen35MTPEnabled = true

        // 2. Load the merged 3.6 checkpoint (model_type qwen3_6_mtp -> Qwen35Model + MTP head).
        progress("[load] \(dir.path)")
        let container = try await loadModelContainer(
            from: dir, using: #huggingFaceTokenizerLoader())
        progress("[load] done")

        try await container.perform { (context: ModelContext) in
            guard let mtp = context.model as? Qwen35Model, mtp.hasMTPHead else {
                throw Err("model is not MTP-capable or MTP head is not attached "
                    + "(is _qwen35MTPEnabled set and does the checkpoint carry mtp.* tensors?)")
            }
            let tokenizer = context.tokenizer
            // One token list, used by BOTH the serial and MTP paths, so the exactness
            // comparison stays apples-to-apples regardless of --chat.
            let promptTokens: [Int]
            if useChat {
                let messages: [[String: any Sendable]] = [["role": "user", "content": prompt]]
                // Qwen 3.6 is a thinking model; disable the think block so a short reply
                // actually reaches <|im_end|> inside a modest token budget.
                if let withoutThinking = try? tokenizer.applyChatTemplate(
                    messages: messages, tools: nil,
                    additionalContext: ["enable_thinking": false])
                {
                    promptTokens = withoutThinking
                } else {
                    // Never leave the enable_thinking state ambiguous: say so loudly.
                    progress("[warn] chat template rejected additionalContext "
                        + "enable_thinking=false; falling back to the DEFAULT template "
                        + "(thinking ENABLED). Expect a <think> block and possibly no EOS "
                        + "within the token budget.")
                    promptTokens = try tokenizer.applyChatTemplate(messages: messages)
                }
            } else {
                promptTokens = tokenizer.encode(text: prompt)
            }
            let (stopTokens, stopSource) = resolveStopTokens(dir: dir, tokenizer: tokenizer)
            progress("[setup] prompt tokens: \(promptTokens.count)  maxTokens: \(maxTokens)  "
                + "depth: \(depth)  hidden: \(postNorm ? "post_norm" : "pre_norm")  "
                + "chat: \(useChat)")
            progress("[setup] stop tokens: \(stopTokens.sorted()) (from \(stopSource))")

            // Reference: plain autoregressive greedy for exactness comparison.
            // `arMargins[i]` is the serial forward's top1-top2 logit gap at the position
            // that emitted `arTokens[i]` — one scalar per token, so memory stays bounded.
            let (arTokens, arSeconds, arMargins) = try serialGreedy(
                mtp, promptTokens, maxTokens: maxTokens, stopTokens: stopTokens)
            progress("[serial] tokens: \(arTokens)")
            // Native-MTP speculative greedy.
            let r = try mtpGreedy(
                mtp, promptTokens, maxTokens: maxTokens, depth: depth, postNorm: postNorm,
                stopTokens: stopTokens)

            let exact = arTokens == r.tokens
            progress("[mtp]    tokens: \(r.tokens)")
            if !exact {
                let firstDiff = zip(arTokens, r.tokens).enumerated()
                    .first(where: { $0.element.0 != $0.element.1 })?.offset
                    ?? min(arTokens.count, r.tokens.count)
                progress("[diff] first mismatch at index \(firstDiff)")
                // Classify the mismatch on the spot: a near-zero top1/top2 gap at the
                // divergent position is the tolerated bf16 batched-vs-serial argmax
                // tie-flip; a large gap means the two paths genuinely disagree (logic bug).
                if firstDiff < arMargins.count {
                    let margin = arMargins[firstDiff]
                    let verdict = margin < 1e-2
                        ? "TIE-FLIP (numerics: serial top1/top2 nearly tied)"
                        : "LOGIC BUG (serial was confident here)"
                    progress(String(
                        format: "[diff] serial top1-top2 margin at index %d: %.6g  -> %@",
                        firstDiff, Double(margin), verdict))
                    let serialTok = firstDiff < arTokens.count ? "\(arTokens[firstDiff])" : "n/a"
                    let mtpTok = firstDiff < r.tokens.count ? "\(r.tokens[firstDiff])" : "n/a"
                    progress("[diff] serial tok=\(serialTok)  mtp tok=\(mtpTok)")
                } else {
                    progress("[diff] no serial margin recorded at index \(firstDiff) "
                        + "(length mismatch: serial \(arTokens.count), mtp \(r.tokens.count))")
                }
                progress("[diff] serial text: \(tokenizer.decode(tokenIds: arTokens))")
            }

            let speedup = arSeconds / max(r.seconds, 1e-9)
            let acceptRate = 100.0 * Double(r.accepted) / Double(max(r.drafted, 1))
            print("""
                --- Qwen 3.6 native-MTP ---
                tokens: \(r.tokens.count)   depth: \(depth)
                exact-vs-serial: \(exact ? "MATCH ✓" : "MISMATCH ✗")
                accept rate: \(String(format: "%.1f%%", acceptRate)) (\(r.accepted)/\(r.drafted) drafted, \(r.rounds) rounds)
                serial: \(String(format: "%.2f", arSeconds))s   mtp: \(String(format: "%.2f", r.seconds))s   speedup: \(String(format: "%.2fx", speedup))
                text: \(tokenizer.decode(tokenIds: r.tokens))
                """)
        }
    }

    static func serialGreedy(
        _ m: Qwen35Model, _ prompt: [Int], maxTokens: Int, stopTokens: Set<Int>
    ) throws -> ([Int], Double, [Float]) {
        let cache = m.newCache(parameters: nil)
        var out: [Int] = []
        var margins: [Float] = []
        let t0 = now()
        var cur = MLXArray(prompt).reshaped([1, prompt.count])
        for i in 0 ..< maxTokens {
            let (logits, _) = m.callWithHidden(
                input: LMInput.Text(tokens: cur), cache: cache, nConfirmed: 0)
            let next = argmaxLast(logits)
            // One scalar per emitted token: how far the greedy pick beat the runner-up.
            margins.append(top2Margin(logits))
            out.append(next)
            cur = MLXArray([next]).reshaped([1, 1])
            eval(cache.flatMap { $0.state })
            let isStop = stopTokens.contains(next)
            progress(String(
                format: "[serial] %d/%d tok=%d  t=%.1fs%@",
                i + 1, maxTokens, next, now() - t0, isStop ? "  (eos)" : ""))
            if isStop { break }
        }
        return (out, now() - t0, margins)
    }

    struct MTPResult {
        var tokens: [Int]
        var seconds: Double
        var accepted: Int
        var drafted: Int
        var rounds: Int
    }

    static func mtpGreedy(
        _ m: Qwen35Model, _ prompt: [Int], maxTokens: Int, depth: Int, postNorm: Bool,
        stopTokens: Set<Int>
    ) throws -> MTPResult {
        let cache = m.newCache(parameters: nil)
        var tokens: [Int] = []
        var accepted = 0, drafted = 0, rounds = 0
        let t0 = now()

        // Prefill (MTPLX `_prefill(..., return_hidden=True)`): the pending primary is
        // sampled from these logits and is NOT yet written to the cache.
        let (pLogits, pHidden) = m.callWithHidden(
            input: LMInput.Text(tokens: MLXArray(prompt).reshaped([1, prompt.count])),
            cache: cache, nConfirmed: 0)
        var logitsRow = pLogits[0..., (pLogits.dim(1) - 1) ..< pLogits.dim(1), 0...]
        var hidden = hiddenRow(m, pHidden, pHidden.dim(1) - 1, postNorm)
        eval(cache.flatMap { $0.state })
        progress(String(format: "[mtp] prefill done  t=%.1fs", now() - t0))

        while tokens.count < maxTokens {
            rounds += 1
            // Cache invariant at the top of a round: every emitted token is in the
            // cache and the pending primary is not.
            let base = attentionOffset(cache)
            let expected = prompt.count + tokens.count
            if base != expected {
                throw Err("cache offset invariant broken: kv offset \(base) != "
                    + "promptLen+emitted \(expected) at round \(rounds)")
            }

            let primary = argmaxLast(logitsRow)
            tokens.append(primary)
            if tokens.count >= maxTokens || stopTokens.contains(primary) {
                let why = stopTokens.contains(primary) ? "eos" : "budget"
                progress(String(format: "[mtp] round %d  primary=%d  emitted %d/%d  t=%.1fs (%@)",
                    rounds, primary, tokens.count, maxTokens, now() - t0, why))
                break
            }

            // 1. Draft. cycleDepth is clamped AFTER the primary is appended
            //    (MTPLX: cycle_depth = min(planned_depth, max_tokens - len(tokens))).
            let cycleDepth = min(depth, maxTokens - tokens.count)
            let mtpCache = m.makeMTPCache()      // one per round, shared across sub-steps
            var drafts: [Int] = []
            var draftHidden = hidden
            var nextToken = primary
            for _ in 0 ..< cycleDepth {
                let (dl, dh) = m.mtpForwardWithHidden(
                    hidden: draftHidden,
                    nextTokenIds: MLXArray([nextToken]).reshaped([1, 1]),
                    cache: mtpCache)
                let d = argmaxLast(dl)
                drafts.append(d)
                // Chain the head's own post-`mtp.norm` hidden into the next sub-step.
                draftHidden = dh[0..., (dh.dim(1) - 1) ..< dh.dim(1), 0...]
                nextToken = d
                drafted += 1
            }

            // 2. Snapshot the recurrent (non-trimmable) state, then verify in one forward.
            let snapshot = snapshotRecurrent(cache)
            let verifyInput = [primary] + drafts
            let (vLogits, vHidden) = m.callWithHidden(
                input: LMInput.Text(tokens: MLXArray(verifyInput).reshaped([1, verifyInput.count])),
                cache: cache, nConfirmed: 0)
            let vArg = argmaxAll(vLogits)

            // 3. Longest-common-prefix acceptance over rows 0 ..< cycleDepth.
            var acceptedCount = 0
            for i in 0 ..< drafts.count {
                if vArg[i] == drafts[i] {
                    acceptedCount += 1
                    if stopTokens.contains(drafts[i]) { break }
                } else { break }
            }
            accepted += acceptedCount

            if acceptedCount == drafts.count {
                // Full acceptance: the verify state IS the committed state.
                tokens.append(contentsOf: drafts)
                logitsRow = vLogits[0..., drafts.count ..< (drafts.count + 1), 0...]
                hidden = hiddenRow(m, vHidden, vHidden.dim(1) - 1, postNorm)
            } else {
                // Partial: undo the whole verify window, then re-forward the committed block.
                let committed = [primary] + Array(drafts.prefix(acceptedCount))
                tokens.append(contentsOf: committed.dropFirst())
                rollbackAfterVerify(cache, snapshot, verifiedTokens: verifyInput.count, to: base)
                let (rLogits, rHidden) = m.callWithHidden(
                    input: LMInput.Text(tokens: MLXArray(committed).reshaped([1, committed.count])),
                    cache: cache, nConfirmed: 0)
                logitsRow = rLogits[0..., (rLogits.dim(1) - 1) ..< rLogits.dim(1), 0...]
                hidden = hiddenRow(m, rHidden, rHidden.dim(1) - 1, postNorm)
            }

            eval(cache.flatMap { $0.state })
            eval(logitsRow, hidden)
            progress(String(
                format: "[mtp] round %d  primary=%d drafts=%@ accepted %d/%d  emitted %d/%d  t=%.1fs",
                rounds, primary, "\(drafts)", acceptedCount, cycleDepth,
                tokens.count, maxTokens, now() - t0))

            if let stop = ([primary] + drafts.prefix(acceptedCount)).first(
                where: { stopTokens.contains($0) })
            {
                progress("[mtp] stop token \(stop) committed")
                if let idx = tokens.firstIndex(where: { stopTokens.contains($0) }) {
                    tokens = Array(tokens.prefix(idx + 1))
                }
                break
            }
        }
        return MTPResult(
            tokens: Array(tokens.prefix(maxTokens)), seconds: now() - t0,
            accepted: accepted, drafted: drafted, rounds: rounds)
    }

    // MARK: - cache snapshot / rollback (MTPLX cache_state.py)

    /// `snapshot_untrimmable_cache`: capture the recurrent (GDN) layers' state arrays.
    ///
    /// Each leaf is retained as a fresh slice *expression* (`[.ellipsis]`), not as a bare
    /// reference. A slice expression references the array's current value, so a later
    /// container write can never mutate the snapshot behind our back — whether the writer
    /// rebinds the slot (which `Qwen35GatedDeltaNet` happens to do today, Qwen35.swift:390-393)
    /// or setitem-mutates it (which nothing pins it against). No GPU work happens here;
    /// this is MTPLX's `_lazy_state_view` (cache_state.py:3442-3454, `value[...]`), and the
    /// same idiom the fork itself uses in `MambaCache.copy()` (KVCache.swift:1380).
    static func snapshotRecurrent(_ cache: [any KVCache]) -> [Int: [MLXArray?]] {
        var snap: [Int: [MLXArray?]] = [:]
        for (i, c) in cache.enumerated() {
            guard let a = c as? ArraysCache else { continue }
            snap[i] = [a[0]?[.ellipsis], a[1]?[.ellipsis]]
        }
        return snap
    }

    /// `rollback_after_verify`: trim every verified position from the trimmable (KV)
    /// caches and restore the recurrent snapshot.
    static func rollbackAfterVerify(
        _ cache: [any KVCache], _ snapshot: [Int: [MLXArray?]], verifiedTokens: Int, to base: Int
    ) {
        for (i, c) in cache.enumerated() {
            if let a = c as? ArraysCache {
                if let s = snapshot[i] {
                    a[0] = s[0]
                    a[1] = s[1]
                }
                a.rollbackState = nil
                continue
            }
            if c.isTrimmable && c.offset > base {
                c.trim(min(verifiedTokens, c.offset - base))
            }
        }
    }

    /// Offset of the first trimmable (global-attention) cache — the sequence position.
    static func attentionOffset(_ cache: [any KVCache]) -> Int {
        for c in cache where !(c is ArraysCache) { return c.offset }
        return -1
    }

    // MARK: - helpers

    /// One hidden row `[1, 1, H]`, in MTPLX's default "post_norm" variant unless
    /// `--hidden pre` was given. `callWithHidden` returns PRE-norm hidden, so the
    /// backbone's final `model.norm` is applied here (via the fork's `applyFinalNorm`,
    /// since `Qwen35TextModelInner.norm` is not visible outside the module).
    static func hiddenRow(_ m: Qwen35Model, _ h: MLXArray, _ idx: Int, _ postNorm: Bool)
        -> MLXArray
    {
        let row = h[0..., idx ..< (idx + 1), 0...]
        return postNorm ? m.applyFinalNorm(row) : row
    }

    /// Top1-top2 logit gap of the LAST position — the confidence of a greedy pick.
    /// Near zero means the two paths' argmax can legitimately disagree on a bf16 rounding
    /// difference (the tolerated batched-vs-serial tie-flip class); large means they don't.
    static func top2Margin(_ logits: MLXArray) -> Float {
        let row = logits[0..., (logits.dim(1) - 1) ..< logits.dim(1), 0...].asType(.float32)
        let two = top(row, k: 2, axis: -1)          // 2 largest values, order unspecified
        let gap = two.max(axis: -1) - two.min(axis: -1)
        return gap.item(Float.self)
    }

    static func argmaxLast(_ logits: MLXArray) -> Int {
        let l = logits[0..., (logits.dim(1) - 1) ..< logits.dim(1), 0...]
        return argMax(l, axis: -1).item(Int.self)
    }
    static func argmaxAll(_ logits: MLXArray) -> [Int] {
        let a = argMax(logits, axis: -1)[0]      // [S]
        return a.asArray(Int.self)
    }
    static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
    static func flag(_ a: [String], _ f: String) -> String? {
        guard let i = a.firstIndex(of: f), i + 1 < a.count else { return nil }
        return a[i + 1]
    }
    struct Err: Error, CustomStringConvertible { let m: String; init(_ m: String) { self.m = m }
        var description: String { m } }
}
