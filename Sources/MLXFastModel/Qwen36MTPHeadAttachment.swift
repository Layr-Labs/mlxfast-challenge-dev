import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

// Merge-at-load for the SEPARATELY PINNED Qwen 3.6 MTP head.
//
// THE DELTA FROM THE VALIDATED DRIVER, STATED PLAINLY. The MTP session's driver
// loaded a MERGED checkpoint: a directory produced by `merge-mtp-head.sh`, which
// hardlinks the backbone shards, writes the head's tensors into a new shard under
// an `mtp.` prefix, registers them in the index, and rewrites `config.json` to
// declare `model_type: qwen3_6_mtp` with `mtp_num_hidden_layers: 1`. The ranked
// track does NOT do that: the operator's Q8 decision is SEPARATE TREES, both
// upstream-revision-pinned and both byte-verified against their own manifests
// before anything loads. So the merge happens at load time, in memory, here.
//
// What has to be equivalent, and how each half is checked:
//
//  1. TENSOR NAMES. The published head (mlx-community/Qwen3.6-27B-MTP-4bit) uses
//     BARE names — `fc.*`, `norm.weight`, `pre_fc_norm_embedding.weight`,
//     `pre_fc_norm_hidden.weight`, `layers.0.*`. The merge script prefixes every
//     one with `mtp.`; `_additionalWeightSources` does exactly the same thing,
//     with the same prefix, before `sanitize` and before the quantization walk.
//     `expectedHeadTensorCount` pins the count the pinned revision carries, so a
//     head tree that lost or gained a tensor is a load-time failure rather than a
//     model that silently drafts from a partly-uninitialized head.
//
//  2. QUANTIZATION WIRING. `quantize(model:)` decides which submodules become
//     quantized layers by asking whether `weights["<path>.scales"]` exists. The
//     merge therefore has to land BEFORE that walk, which is why the vendored
//     hook sits where it does and not in a post-load `update(parameters:)`.
//
//  3. CONFIG SEMANTICS. The merge script sets `model_type: qwen3_6_mtp` and
//     `mtp_num_hidden_layers: 1`. Neither rewrite is needed here and NEITHER IS
//     SKIPPED SILENTLY: the pinned backbone's own `text_config` already carries
//     `mtp_num_hidden_layers: 1` (it is the same architecture family; the
//     backbone revision simply ships no `mtp.*` TENSORS), and the transform
//     copies `text_config` verbatim into the runtime `config.json` — so the
//     loaded configuration's `mtpNumHiddenLayers` is 1 either way, and
//     `Qwen35TextModel.init` attaches the head on `mtpNumHiddenLayers > 0 &&
//     _qwen35MTPEnabled`. The `qwen3_6_mtp` model_type alias exists only to route
//     the MERGED directory through the same factory entry the un-merged one
//     already reaches as `qwen3_5_text`. `verifyHeadConfiguration` asserts the
//     two config semantics the loader actually keys on, so a head tree whose
//     config drifted away from the backbone's family fails loudly.
//
//  4. PYTHON HAZARD, CARRIED FORWARD. mlx-lm's `qwen3_5` `sanitize()` treats any
//     `mtp.*` key as a signal to +1-shift every trunk norm weight, which corrupts
//     the model. That hazard belongs to Python consumers of a MERGED directory
//     (QWEN36-BOX3-MTP-RESPONSE.md §3.2); the Swift loader is unaffected, and this
//     path never writes a merged directory for anyone else to read. The note is
//     kept here because this is now the place the recipe lives.

/// Loads and validates the separately pinned Qwen 3.6 MTP head alongside a
/// backbone weights tree.
public enum Qwen36MTPHeadAttachment {
    /// Tensors the pinned head revision (83795d54) carries, counted from its own
    /// `model.safetensors.index.json`: `fc.{weight,scales,biases}`, `norm.weight`,
    /// `pre_fc_norm_embedding.weight`, `pre_fc_norm_hidden.weight` and the single
    /// decoder layer's 25 entries.
    public static let expectedHeadTensorCount = 31

    /// The key prefix the head's bare tensor names are merged under.
    public static let headKeyPrefix = "mtp."

    /// Run `body` with the head tree registered as an additional weight source
    /// and `_qwen35MTPEnabled` set, restoring both afterwards.
    ///
    /// BOTH globals are restored on EVERY exit path. They are process-global and
    /// read by the next load of any model; leaking either would silently change
    /// how an unrelated later load behaves, which on a worker that serves a
    /// reference replay after the candidate is exactly the kind of cross-phase
    /// coupling this track cannot have.
    public static func withHeadAttached<T>(
        headDirectory: URL,
        _ body: () throws -> T
    ) throws -> T {
        try verifyHeadTree(headDirectory)
        let previousSources = _additionalWeightSources
        let previousEnabled = _qwen35MTPEnabled
        _additionalWeightSources = [
            AdditionalWeightSource(
                directory: headDirectory, keyPrefix: headKeyPrefix)
        ]
        _qwen35MTPEnabled = true
        defer {
            _additionalWeightSources = previousSources
            _qwen35MTPEnabled = previousEnabled
        }
        return try body()
    }

    /// Structural checks that do not need MLX and are therefore unit-testable.
    ///
    /// Deliberately NOT a hash check: the byte identity of the head tree is the
    /// ranked workflow's job (`verify_cache` against
    /// `fixtures/qwen3_6_27b_mtp_head.sha256`, every run, before anything loads).
    /// Repeating it here would be a second, weaker copy of a stronger gate; what
    /// this adds is the shape the LOADER depends on.
    public static func verifyHeadTree(_ headDirectory: URL) throws {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(
            atPath: headDirectory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head directory does not exist: \(headDirectory.path)")
        }
        let indexURL = headDirectory.appendingPathComponent(
            "model.safetensors.index.json")
        guard let indexData = try? Data(contentsOf: indexURL) else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head tree carries no model.safetensors.index.json")
        }
        try verifyHeadIndex(indexData)
        let configURL = headDirectory.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head tree carries no config.json")
        }
        try verifyHeadConfiguration(configData)
    }

    /// The head index must name exactly the pinned tensor set, under BARE names.
    ///
    /// A head tree that already carries `mtp.`-prefixed names is REFUSED rather
    /// than accommodated: it would be a pre-merged artifact, the prefix would be
    /// applied twice, and the resulting `mtp.mtp.*` keys would be dropped by
    /// `update(parameters:)` leaving the head at its initialization values — a
    /// model that runs, drafts nonsense, accepts ~nothing, and still reports
    /// exact. Failing the load is the only outcome that is not silent.
    public static func verifyHeadIndex(_ indexData: Data) throws {
        guard let root = try? JSONSerialization.jsonObject(with: indexData)
            as? [String: Any],
            let weightMap = root["weight_map"] as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head index is not a JSON object with a weight_map")
        }
        guard weightMap.count == expectedHeadTensorCount else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head index names \(weightMap.count) tensors; the "
                    + "pinned head revision carries \(expectedHeadTensorCount)")
        }
        if let prefixed = weightMap.keys.first(where: {
            $0.hasPrefix(headKeyPrefix)
        }) {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head tree already carries prefixed tensor names "
                    + "(e.g. \(prefixed)); this loader merges a BARE head tree and "
                    + "would double-prefix a pre-merged one")
        }
        for required in ["fc.weight", "norm.weight", "pre_fc_norm_hidden.weight"] {
            guard weightMap[required] != nil else {
                throw MLXFastError.invalidInput(
                    "the Qwen MTP head index is missing \(required)")
            }
        }
    }

    /// The two config semantics the loader keys on, asserted on the HEAD tree.
    ///
    /// The head's own `config.json` is not what the model is built from — the
    /// backbone's is — so this is a compatibility assertion, not a load input: it
    /// proves the head being merged belongs to the same architecture family and
    /// declares the single MTP layer the backbone's `mtp_num_hidden_layers: 1`
    /// promises.
    public static func verifyHeadConfiguration(_ configData: Data) throws {
        guard let root = try? JSONSerialization.jsonObject(with: configData)
            as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head config.json is not a JSON object")
        }
        guard let modelType = root["model_type"] as? String,
              modelType.hasPrefix("qwen3_5") || modelType.hasPrefix("qwen3_6")
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head declares model_type "
                    + "\(root["model_type"] as? String ?? "<missing>"), which is not "
                    + "a Qwen 3.5/3.6 MTP head")
        }
        let textConfig = (root["text_config"] as? [String: Any]) ?? root
        guard let mtpLayers = textConfig["mtp_num_hidden_layers"] as? Int,
              mtpLayers == 1
        else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head declares mtp_num_hidden_layers "
                    + "\(textConfig["mtp_num_hidden_layers"] as? Int ?? -1); the "
                    + "pinned head is a single-layer head and the backbone's own "
                    + "text_config promises exactly one")
        }
    }
}
