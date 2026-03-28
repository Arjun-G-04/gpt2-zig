const n_embd: u8 = 384;
const vocab_size: u8 = 65;
const block_size: u8 = 256;
const n_layer: u8 = 6;

const Block = struct {
    something: u8
}; 

const Transformer = struct {
    wte: [vocab_size][n_embd]f16,
    wpe: [block_size][n_embd]f16,
    h: [n_layer]Block
};

const GPT = struct {
    transformer: Transformer
};
