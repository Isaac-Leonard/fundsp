## Future

This is a list of feature ideas for the future.

- What is the best approach to making `Granular` real-time safe.
- `AudioUnit` versions of `oversample` and `resample` that accept an inner `AudioUnit`.
- Compressor without lookahead.
- Adaptive normalizer without lookahead.
- Exponential follower (`follow` is linear).
- More physical models. Karplus-Strong exists already; figure out if it could be improved somehow.
- Dynamic bypass wrapper that bypasses a node when input and output levels drop low enough.
- Improve basic effects implemented in graph notation such as `reverb` (e.g., early reflections), `chorus`, `flanger` and `phaser`.
- Some kind of parameter system that works with both `AudioNode` and `AudioUnit` systems. At its simplest, it could be a key-value system with a fixed set of keys.
- More sound generators in the `gen` module.
- Improve or replace the drum sounds in the library.
- Real-time safe sound server that uses `cpal`. It could have a static set of read/write channels for rendering audio, including hardware channels.
- Conversion of graphs into a graphical form. Format associative operator chains appropriately.
- Interpreter for simple FunDSP expressions.
- Expand `README.md` into a book.
