import MLX

struct Gemma4NativeResidentLayerWeights {
    let block: Gemma4BlockWeights
    let attention: Gemma4AttentionWeights
    let mlp: Gemma4MLPWeights
    let qMetadata: IndexedAffineMetadata?
    let kMetadata: IndexedAffineMetadata?
    let vMetadata: IndexedAffineMetadata?
    let outputMetadata: IndexedAffineMetadata?
    let gateMetadata: IndexedAffineMetadata?
    let upMetadata: IndexedAffineMetadata?
    let downMetadata: IndexedAffineMetadata?

    init(
        block: Gemma4BlockWeights,
        attention: Gemma4AttentionWeights,
        mlp: Gemma4MLPWeights,
        qMetadata: IndexedAffineMetadata? = nil,
        kMetadata: IndexedAffineMetadata? = nil,
        vMetadata: IndexedAffineMetadata? = nil,
        outputMetadata: IndexedAffineMetadata? = nil,
        gateMetadata: IndexedAffineMetadata? = nil,
        upMetadata: IndexedAffineMetadata? = nil,
        downMetadata: IndexedAffineMetadata? = nil
    ) {
        self.block = block
        self.attention = attention
        self.mlp = mlp
        self.qMetadata = qMetadata
        self.kMetadata = kMetadata
        self.vMetadata = vMetadata
        self.outputMetadata = outputMetadata
        self.gateMetadata = gateMetadata
        self.upMetadata = upMetadata
        self.downMetadata = downMetadata
    }
}

struct Gemma4NativeResidentWeights {
    let model: Gemma4ModelWeights
    let layers: [Gemma4NativeResidentLayerWeights]
}
