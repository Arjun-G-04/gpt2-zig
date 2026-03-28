const n_embd: u8 = 384;
const vocab_size: u8 = 65;
const block_size: u8 = 256;
const n_layer: u8 = 6;

// To-Do
const Linear = struct {};

// To-Do
const LayerNorm = struct {};

// To-Do
const Block = struct {};

const Transformer = struct { wte: [vocab_size][n_embd]f16, wpe: [block_size][n_embd]f16, h: [n_layer]Block, ln_f: LayerNorm };

const GPT = struct { transformer: Transformer, lm_head: Linear };
