//! The central `AudioNode` abstraction and basic components.

use super::buffer::*;
use super::combinator::*;
use super::math::*;
use super::signal::*;
use super::*;
use num_complex::Complex64;
use numeric_array::typenum::*;
use std::marker::PhantomData;

/// Type-level integer.
pub trait Size<T>: numeric_array::ArrayLength<T> {}

impl<T, A: numeric_array::ArrayLength<T>> Size<T> for A {}

/// Frames are used to transport audio data between `AudioNode` instances.
pub type Frame<T, Size> = numeric_array::NumericArray<T, Size>;

/// Tags are identified with parameters. Parameter values are always `f64`.
pub type Tag = i64;

/*
Order of type arguments in nodes:
1. Basic input and output arities excepting filter input selector arities.
2. Interface float type.
3. Processing float type.
4. Unary or binary operation type.
5. Filter input selector arity.
6. Contained node types.
7. The rest in any order.
*/

/// Generic audio processor.
/// `AudioNode` has a static number of inputs (`AudioNode::Inputs`) and outputs (`AudioNode::Outputs`).
/// `AudioNode` processes samples of type `AudioNode::Sample`, chosen statically.
pub trait AudioNode {
    /// Unique ID for hashing.
    const ID: u64;
    /// Sample type for input and output.
    type Sample: Float;
    /// Input arity.
    type Inputs: Size<Self::Sample>;
    /// Output arity.
    type Outputs: Size<Self::Sample>;

    /// Reset the input state of the component to an initial state where it has
    /// not processed any samples. In other words, reset time to zero.
    /// The sample rate can be set optionally. The default sample rate is 44.1 kHz.
    /// The default implementation does nothing.
    fn reset(&mut self, _sample_rate: Option<f64>) {}

    /// Process one sample.
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs>;

    /// Process up to 64 (`MAX_BUFFER_SIZE`) samples.
    /// The number of input and output buffers must match the number of inputs and outputs, respectively.
    /// All input and output buffers must be at least as large as `size`.
    /// The default implementation is a fallback that calls into `tick`.
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        debug_assert!(size > 0 && size <= MAX_BUFFER_SIZE);
        debug_assert!(input.len() == self.inputs());
        debug_assert!(output.len() == self.outputs());
        debug_assert!(input.iter().all(|x| x.len() >= size));
        debug_assert!(output.iter().all(|x| x.len() >= size));
        for i in 0..size {
            let result = self.tick(&Frame::generate(|j| input[j][i]));
            for (j, &x) in result.iter().enumerate() {
                output[j][i] = x;
            }
        }
    }

    /// Set node pseudorandom phase hash. Override this to use the hash.
    /// This is called from `ping`. It should not be called by users.
    /// The default implementation does nothing.
    fn set_hash(&mut self, _hash: u64) {}

    /// Ping contained `AudioNode`s to obtain a deterministic pseudorandom hash.
    /// The local hash includes children, too.
    /// Leaf nodes should not need to override this.
    /// If `probe` is true, then this is a probe for computing the network hash
    /// and `set_hash` should not be called yet.
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        if !probe {
            self.set_hash(hash.value());
        }
        hash.hash(Self::ID)
    }

    /// Route constants, latencies and frequency responses at `frequency` Hz
    /// from inputs to outputs. Return output signal.
    /// Default implementation marks all outputs unknown.
    fn route(&self, _input: &SignalFrame, _frequency: f64) -> SignalFrame {
        new_signal_frame(self.outputs())
    }

    /// Set parameter value recursively. The default implementation does nothing.
    fn set(&mut self, _parameter: Tag, _value: f64) {}

    /// Query parameter value recursively. The first matching parameter is returned.
    /// The default implementation returns `None`.
    fn get(&self, _parameter: Tag) -> Option<f64> {
        None
    }

    // End of interface. There is no need to override the following.

    /// Number of inputs.
    #[inline]
    fn inputs(&self) -> usize {
        Self::Inputs::USIZE
    }

    /// Number of outputs.
    #[inline]
    fn outputs(&self) -> usize {
        Self::Outputs::USIZE
    }

    /// Evaluate frequency response of `output` at `frequency` Hz.
    /// Any linear response can be composed.
    /// Return `None` if there is no response or it could not be calculated.
    fn response(&self, output: usize, frequency: f64) -> Option<Complex64> {
        assert!(output < self.outputs());
        let mut input = new_signal_frame(self.inputs());
        for i in 0..self.inputs() {
            input[i] = Signal::Response(Complex64::new(1.0, 0.0), 0.0);
        }
        let response = self.route(&input, frequency);
        match response[output] {
            Signal::Response(rx, _) => Some(rx),
            _ => None,
        }
    }

    /// Evaluate frequency response of `output` in dB at `frequency Hz`.
    /// Any linear response can be composed.
    /// Return `None` if there is no response or it could not be calculated.
    fn response_db(&self, output: usize, frequency: f64) -> Option<f64> {
        assert!(output < self.outputs());
        self.response(output, frequency).map(|r| amp_db(r.norm()))
    }

    /// Causal latency in (fractional) samples.
    /// After a reset, we can discard this many samples from the output to avoid incurring a pre-delay.
    /// The latency can depend on the sample rate and is allowed to change after `reset`.
    fn latency(&self) -> Option<f64> {
        if self.outputs() == 0 {
            return None;
        }
        let mut input = new_signal_frame(self.inputs());
        for i in 0..self.inputs() {
            input[i] = Signal::Latency(0.0);
        }
        // The frequency argument can be anything as there are no responses to propagate,
        // only latencies. Latencies are never promoted to responses during signal routing.
        let response = self.route(&input, 1.0);
        // Return the minimum latency.
        let mut result: Option<f64> = None;
        for output in 0..self.outputs() {
            match (result, response[output]) {
                (None, Signal::Latency(x)) => result = Some(x),
                (Some(r), Signal::Latency(x)) => result = Some(r.min(x)),
                _ => (),
            }
        }
        result
    }

    /// Retrieve the next mono sample from a generator.
    /// The node must have no inputs and 1 or 2 outputs.
    /// If there are two outputs, average the channels.
    #[inline]
    fn get_mono(&mut self) -> Self::Sample {
        assert!(
            Self::Inputs::USIZE == 0 && (Self::Outputs::USIZE == 1 || Self::Outputs::USIZE == 2)
        );
        let output = self.tick(&Frame::default());
        if self.outputs() == 1 {
            output[0]
        } else {
            (output[0] + output[1]) / Self::Sample::new(2)
        }
    }

    /// Retrieve the next stereo sample (left, right) from a generator.
    /// The node must have no inputs and 1 or 2 outputs.
    /// If there is just one output, duplicate it.
    #[inline]
    fn get_stereo(&mut self) -> (Self::Sample, Self::Sample) {
        assert!(
            Self::Inputs::USIZE == 0 && (Self::Outputs::USIZE == 1 || Self::Outputs::USIZE == 2)
        );
        let output = self.tick(&Frame::default());
        (
            output[0],
            output[if Self::Outputs::USIZE > 1 { 1 } else { 0 }],
        )
    }

    /// Filter the next mono sample `x`.
    /// The node must have exactly 1 input and 1 output.
    #[inline]
    fn filter_mono(&mut self, x: Self::Sample) -> Self::Sample {
        assert!(Self::Inputs::USIZE == 1 && Self::Outputs::USIZE == 1);
        let output = self.tick(&Frame::splat(x));
        output[0]
    }

    /// Filter the next stereo sample `(x, y)`.
    /// The node must have exactly 2 inputs and 2 outputs.
    #[inline]
    fn filter_stereo(&mut self, x: Self::Sample, y: Self::Sample) -> (Self::Sample, Self::Sample) {
        assert!(Self::Inputs::USIZE == 2 && Self::Outputs::USIZE == 2);
        let output = self.tick(&Frame::generate(|i| if i == 0 { x } else { y }));
        (output[0], output[1])
    }
}

/// Pass through inputs unchanged.
#[derive(Default)]
pub struct MultiPass<N, T> {
    _marker: PhantomData<(N, T)>,
}

impl<N: Size<T>, T: Float> MultiPass<N, T> {
    pub fn new() -> Self {
        MultiPass::default()
    }
}

impl<N: Size<T>, T: Float> AudioNode for MultiPass<N, T> {
    const ID: u64 = 0;
    type Sample = T;
    type Inputs = N;
    type Outputs = N;

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        input.clone()
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        for i in 0..self.inputs() {
            output[i][..size].clone_from_slice(&input[i][..size]);
        }
    }
    fn route(&self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        input.clone()
    }
}

/// Pass through input unchanged.
#[derive(Default)]
pub struct Pass<T> {
    _marker: PhantomData<T>,
}

impl<T: Float> Pass<T> {
    pub fn new() -> Self {
        Pass::default()
    }
}

// Note. We have separate Pass and MultiPass structs
// because it helps a little with type inference.
impl<T: Float> AudioNode for Pass<T> {
    const ID: u64 = 48;
    type Sample = T;
    type Inputs = U1;
    type Outputs = U1;

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        *input
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        output[0][..size].clone_from_slice(&input[0][..size]);
    }
    fn route(&self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        input.clone()
    }
}

/// Present time as a parameter.
#[derive(Default)]
pub struct Timer<T> {
    _marker: PhantomData<T>,
    tag: Tag,
    time: f64,
    sample_duration: f64,
}

impl<T: Float> Timer<T> {
    /// Create a new timer node. Current time can be read by querying the tag.
    pub fn new(sample_rate: f64, tag: Tag) -> Self {
        Self {
            _marker: PhantomData::default(),
            tag,
            time: 0.0,
            sample_duration: 1.0 / sample_rate,
        }
    }
}

impl<T: Float> AudioNode for Timer<T> {
    const ID: u64 = 57;
    type Sample = T;
    type Inputs = U0;
    type Outputs = U0;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.time = 0.0;
        if let Some(sr) = sample_rate {
            self.sample_duration = 1.0 / sr;
        }
    }

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        self.time += self.sample_duration;
        *input
    }

    fn process(
        &mut self,
        size: usize,
        _input: &[&[Self::Sample]],
        _output: &mut [&mut [Self::Sample]],
    ) {
        self.time += size as f64 * self.sample_duration;
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        if self.tag == parameter {
            Some(self.time)
        } else {
            None
        }
    }
}

/// Discard inputs.
#[derive(Default)]
pub struct Sink<N, T> {
    _marker: PhantomData<(N, T)>,
}

impl<N: Size<T>, T: Float> Sink<N, T> {
    pub fn new() -> Self {
        Sink::default()
    }
}

impl<N: Size<T>, T: Float> AudioNode for Sink<N, T> {
    const ID: u64 = 1;
    type Sample = T;
    type Inputs = N;
    type Outputs = U0;

    #[inline]
    fn tick(
        &mut self,
        _input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        Frame::default()
    }
    fn process(
        &mut self,
        _size: usize,
        _input: &[&[Self::Sample]],
        _output: &mut [&mut [Self::Sample]],
    ) {
    }
}

/// Output a constant value.
pub struct Constant<N: Size<T>, T: Float> {
    output: Frame<T, N>,
}

impl<N: Size<T>, T: Float> Constant<N, T> {
    /// Construct constant.
    pub fn new(output: Frame<T, N>) -> Self {
        Constant { output }
    }
    /// Set the value of the constant.
    pub fn set_value(&mut self, output: Frame<T, N>) {
        self.output = output;
    }
    /// Get the value of the constant.
    pub fn value(&self) -> Frame<T, N> {
        self.output.clone()
    }
}

impl<N: Size<T>, T: Float> AudioNode for Constant<N, T> {
    const ID: u64 = 2;
    type Sample = T;
    type Inputs = U0;
    type Outputs = N;

    #[inline]
    fn tick(
        &mut self,
        _input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        self.output.clone()
    }

    fn process(
        &mut self,
        size: usize,
        _input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        for i in 0..self.outputs() {
            output[i][..size].fill(self.output[i]);
        }
    }

    fn route(&self, _input: &SignalFrame, _frequency: f64) -> SignalFrame {
        let mut output = new_signal_frame(self.outputs());
        for i in 0..N::USIZE {
            output[i] = Signal::Value(self.output[i].to_f64());
        }
        output
    }
}

/// Output a tagged constant.
pub struct Tagged<T: Float> {
    tag: Tag,
    value: T,
}

impl<T: Float> Tagged<T> {
    /// Construct constant whose value can be set and queried via the tag.
    pub fn new(tag: Tag, value: T) -> Self {
        Tagged { tag, value }
    }
    /// Set the value of the constant.
    pub fn set_value(&mut self, value: T) {
        self.value = value;
    }
    /// Get the value of the constant.
    pub fn value(&self) -> T {
        self.value
    }
}

impl<T: Float> AudioNode for Tagged<T> {
    const ID: u64 = 54;
    type Sample = T;
    type Inputs = U0;
    type Outputs = U1;

    #[inline]
    fn tick(
        &mut self,
        _input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        [self.value].into()
    }

    fn process(
        &mut self,
        size: usize,
        _input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        output[0][..size].fill(self.value);
    }

    fn route(&self, _input: &SignalFrame, _frequency: f64) -> SignalFrame {
        let mut output = new_signal_frame(self.outputs());
        output[0] = Signal::Value(self.value.to_f64());
        output
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        if self.tag == parameter {
            self.value = T::from_f64(value);
        }
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        if self.tag == parameter {
            Some(self.value.to_f64())
        } else {
            None
        }
    }
}

/// Split input into `N` channels.
pub struct Split<N, T> {
    _marker: PhantomData<(N, T)>,
}

// Note. We have separate split and multisplit (and join and multijoin)
// implementations because it helps with type inference.
impl<N, T> Split<N, T>
where
    N: Size<T>,
    T: Float,
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            _marker: PhantomData::default(),
        }
    }
}

impl<N, T> AudioNode for Split<N, T>
where
    N: Size<T>,
    T: Float,
{
    const ID: u64 = 40;
    type Sample = T;
    type Inputs = U1;
    type Outputs = N;

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        Frame::splat(input[0])
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        for channel in 0..N::USIZE {
            output[channel][..size].clone_from_slice(&input[0][..size]);
        }
    }
    fn route(&self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        Routing::Split.propagate(input, self.outputs())
    }
}

/// Split `M` inputs into `N` branches, with `M` * `N` outputs.
pub struct MultiSplit<M, N, T> {
    _marker: PhantomData<(M, N, T)>,
}

impl<M, N, T> MultiSplit<M, N, T>
where
    M: Size<T> + Mul<N>,
    N: Size<T>,
    <M as Mul<N>>::Output: Size<T>,
    T: Float,
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            _marker: PhantomData::default(),
        }
    }
}

impl<M, N, T> AudioNode for MultiSplit<M, N, T>
where
    M: Size<T> + Mul<N>,
    N: Size<T>,
    <M as Mul<N>>::Output: Size<T>,
    T: Float,
{
    const ID: u64 = 38;
    type Sample = T;
    type Inputs = M;
    type Outputs = numeric_array::typenum::Prod<M, N>;

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        Frame::generate(|i| input[i % M::USIZE])
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        for channel in 0..M::USIZE * N::USIZE {
            output[channel][..size].clone_from_slice(&input[channel % M::USIZE][..size]);
        }
    }
    fn route(&self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        Routing::Split.propagate(input, self.outputs())
    }
}

/// Join `N` channels into one by averaging. Inverse of `Split<N, T>`.
pub struct Join<N, T> {
    _marker: PhantomData<(N, T)>,
}

impl<N, T> Join<N, T>
where
    N: Size<T>,
    T: Float,
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            _marker: PhantomData::default(),
        }
    }
}

impl<N, T> AudioNode for Join<N, T>
where
    N: Size<T>,
    T: Float,
{
    const ID: u64 = 41;
    type Sample = T;
    type Inputs = N;
    type Outputs = U1;

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let mut output = input[0];
        for i in 1..N::USIZE {
            output += input[i];
        }
        [output / T::new(N::I64)].into()
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        let z = T::one() / T::new(N::I64);
        for i in 0..size {
            output[0][i] = input[0][i] * z;
        }
        for channel in 1..N::USIZE {
            for i in 0..size {
                output[0][i] += input[channel][i] * z;
            }
        }
    }
    fn route(&self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        Routing::Join.propagate(input, self.outputs())
    }
}

/// Average `N` branches of `M` channels into one branch with `M` channels.
/// The input has `M` * `N` channels. Inverse of `MultiSplit<M, N, T>`.
pub struct MultiJoin<M, N, T> {
    _marker: PhantomData<(M, N, T)>,
}

impl<M, N, T> MultiJoin<M, N, T>
where
    M: Size<T> + Mul<N>,
    N: Size<T>,
    <M as Mul<N>>::Output: Size<T>,
    T: Float,
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            _marker: PhantomData::default(),
        }
    }
}

impl<M, N, T> AudioNode for MultiJoin<M, N, T>
where
    M: Size<T> + Mul<N>,
    N: Size<T>,
    <M as Mul<N>>::Output: Size<T>,
    T: Float,
{
    const ID: u64 = 39;
    type Sample = T;
    type Inputs = numeric_array::typenum::Prod<M, N>;
    type Outputs = M;

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        Frame::generate(|j| {
            let mut output = input[j];
            for i in 1..N::USIZE {
                output += input[j + i * M::USIZE];
            }
            output / T::new(N::I64)
        })
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        let z = T::one() / T::new(N::I64);
        for channel in 0..M::USIZE {
            for i in 0..size {
                output[channel][i] = input[channel][i] * z;
            }
        }
        for channel in M::USIZE..M::USIZE * N::USIZE {
            for i in 0..size {
                output[channel % M::USIZE][i] += input[channel][i] * z;
            }
        }
    }
    fn route(&self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        Routing::Join.propagate(input, self.outputs())
    }
}

/// Provides binary operator implementations to the `Binop` node.
pub trait FrameBinop<N: Size<T>, T: Float> {
    /// Do binary op (x op y) channelwise.
    fn binop(x: &Frame<T, N>, y: &Frame<T, N>) -> Frame<T, N>;
    /// Do binary op (x op y) on signals.
    fn propagate(x: Signal, y: Signal) -> Signal;
    /// Do binary op (x op y) in-place lengthwise.
    fn assign(size: usize, x: &mut [T], y: &[T]);
}

/// Addition operator.
#[derive(Default)]
pub struct FrameAdd<N: Size<T>, T: Float> {
    _marker: PhantomData<(N, T)>,
}

impl<N: Size<T>, T: Float> FrameAdd<N, T> {
    pub fn new() -> FrameAdd<N, T> {
        FrameAdd::default()
    }
}

impl<N: Size<T>, T: Float> FrameBinop<N, T> for FrameAdd<N, T> {
    #[inline]
    fn binop(x: &Frame<T, N>, y: &Frame<T, N>) -> Frame<T, N> {
        x + y
    }
    fn propagate(x: Signal, y: Signal) -> Signal {
        x.combine_linear(y, 0.0, |x, y| x + y, |x, y| x + y)
    }
    #[inline]
    fn assign(size: usize, x: &mut [T], y: &[T]) {
        for i in 0..size {
            x[i] += y[i];
        }
    }
}

/// Subtraction operator.
#[derive(Default)]
pub struct FrameSub<N: Size<T>, T: Float> {
    _marker: PhantomData<(N, T)>,
}

impl<N: Size<T>, T: Float> FrameSub<N, T> {
    pub fn new() -> FrameSub<N, T> {
        FrameSub::default()
    }
}

impl<N: Size<T>, T: Float> FrameBinop<N, T> for FrameSub<N, T> {
    #[inline]
    fn binop(x: &Frame<T, N>, y: &Frame<T, N>) -> Frame<T, N> {
        x - y
    }
    fn propagate(x: Signal, y: Signal) -> Signal {
        x.combine_linear(y, 0.0, |x, y| x - y, |x, y| x - y)
    }
    #[inline]
    fn assign(size: usize, x: &mut [T], y: &[T]) {
        for i in 0..size {
            x[i] -= y[i];
        }
    }
}

/// Multiplication operator.
#[derive(Default)]
pub struct FrameMul<N: Size<T>, T: Float> {
    _marker: PhantomData<(N, T)>,
}

impl<N: Size<T>, T: Float> FrameMul<N, T> {
    pub fn new() -> FrameMul<N, T> {
        FrameMul::default()
    }
}

impl<N: Size<T>, T: Float> FrameBinop<N, T> for FrameMul<N, T> {
    #[inline]
    fn binop(x: &Frame<T, N>, y: &Frame<T, N>) -> Frame<T, N> {
        x * y
    }
    fn propagate(x: Signal, y: Signal) -> Signal {
        match (x, y) {
            (Signal::Value(vx), Signal::Value(vy)) => Signal::Value(vx * vy),
            (Signal::Latency(lx), Signal::Latency(ly)) => Signal::Latency(min(lx, ly)),
            (Signal::Response(_, lx), Signal::Response(_, ly)) => Signal::Latency(min(lx, ly)),
            (Signal::Response(_, lx), Signal::Latency(ly)) => Signal::Latency(min(lx, ly)),
            (Signal::Latency(lx), Signal::Response(_, ly)) => Signal::Latency(min(lx, ly)),
            (Signal::Response(rx, lx), Signal::Value(vy)) => {
                Signal::Response(rx * Complex64::new(vy, 0.0), lx)
            }
            (Signal::Value(vx), Signal::Response(ry, ly)) => {
                Signal::Response(ry * Complex64::new(vx, 0.0), ly)
            }
            (Signal::Latency(lx), _) => Signal::Latency(lx),
            (Signal::Response(_, lx), _) => Signal::Latency(lx),
            (_, Signal::Latency(ly)) => Signal::Latency(ly),
            (_, Signal::Response(_, ly)) => Signal::Latency(ly),
            _ => Signal::Unknown,
        }
    }
    #[inline]
    fn assign(size: usize, x: &mut [T], y: &[T]) {
        for i in 0..size {
            x[i] *= y[i];
        }
    }
}

/// Combine outputs of two nodes with a binary operation.
/// Inputs are disjoint.
/// Outputs are combined channelwise.
/// The nodes must have the same number of outputs.
pub struct Binop<T, B, X, Y>
where
    T: Float,
{
    _marker: PhantomData<T>,
    x: X,
    y: Y,
    #[allow(dead_code)]
    b: B,
    buffer: Buffer<T>,
}

impl<T, B, X, Y> Binop<T, B, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T, Outputs = X::Outputs>,
    B: FrameBinop<X::Outputs, T>,
    X::Inputs: Size<T> + Add<Y::Inputs>,
    X::Outputs: Size<T>,
    Y::Inputs: Size<T>,
    <X::Inputs as Add<Y::Inputs>>::Output: Size<T>,
{
    pub fn new(x: X, y: Y, b: B) -> Self {
        let mut node = Binop {
            _marker: PhantomData,
            x,
            y,
            b,
            buffer: Buffer::new(),
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access the left node of the binary operation.
    pub fn left_mut(&mut self) -> &mut X {
        &mut self.x
    }

    /// Access the left node of the binary operation.
    pub fn left(&self) -> &X {
        &self.x
    }

    /// Access the right node of the binary operation.
    pub fn right_mut(&mut self) -> &mut Y {
        &mut self.y
    }

    /// Access the right node of the binary operation.
    pub fn right(&self) -> &Y {
        &self.y
    }
}

impl<T, B, X, Y> AudioNode for Binop<T, B, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T, Outputs = X::Outputs>,
    B: FrameBinop<X::Outputs, T>,
    X::Outputs: Size<T>,
    X::Inputs: Size<T> + Add<Y::Inputs>,
    Y::Inputs: Size<T>,
    <X::Inputs as Add<Y::Inputs>>::Output: Size<T>,
{
    const ID: u64 = 3;
    type Sample = T;
    type Inputs = Sum<X::Inputs, Y::Inputs>;
    type Outputs = X::Outputs;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
        self.y.reset(sample_rate);
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let input_x = &input[..X::Inputs::USIZE];
        let input_y = &input[X::Inputs::USIZE..];
        let x = self.x.tick(input_x.into());
        let y = self.y.tick(input_y.into());
        B::binop(&x, &y)
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.x.process(size, &input[..X::Inputs::USIZE], output);
        self.y.process(
            size,
            &input[X::Inputs::USIZE..],
            self.buffer.get_mut(self.outputs()),
        );
        for i in 0..self.outputs() {
            B::assign(size, output[i], self.buffer.at(i));
        }
    }
    #[inline]
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.y.ping(probe, self.x.ping(probe, hash.hash(Self::ID)))
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        let mut signal_x = self.x.route(input, frequency);
        let signal_y = self.y.route(
            &copy_signal_frame(input, X::Inputs::USIZE, Y::Inputs::USIZE),
            frequency,
        );
        for i in 0..Self::Outputs::USIZE {
            signal_x[i] = B::propagate(signal_x[i], signal_y[i]);
        }
        signal_x
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        self.x.set(parameter, value);
        self.y.set(parameter, value);
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        self.x.get(parameter).or_else(|| self.y.get(parameter))
    }
}

/// Provides unary operator implementations to the `Unop` node.
pub trait FrameUnop<N: Size<T>, T: Float> {
    /// Do unary op channelwise.
    fn unop(x: &Frame<T, N>) -> Frame<T, N>;
    /// Do unary op on signal.
    fn propagate(x: Signal) -> Signal;
    /// Do unary op in-place lengthwise.
    fn assign(size: usize, x: &mut [T]);
}

/// Negation operator.
#[derive(Default)]
pub struct FrameNeg<N: Size<T>, T: Float> {
    _marker: PhantomData<(N, T)>,
}

impl<N: Size<T>, T: Float> FrameNeg<N, T> {
    pub fn new() -> FrameNeg<N, T> {
        FrameNeg::default()
    }
}

impl<N: Size<T>, T: Float> FrameUnop<N, T> for FrameNeg<N, T> {
    #[inline]
    fn unop(x: &Frame<T, N>) -> Frame<T, N> {
        -x
    }
    fn propagate(x: Signal) -> Signal {
        match x {
            Signal::Value(vx) => Signal::Value(-vx),
            Signal::Response(rx, lx) => Signal::Response(-rx, lx),
            s => s,
        }
    }
    #[inline]
    fn assign(size: usize, x: &mut [T]) {
        for i in 0..size {
            x[i] = -x[i];
        }
    }
}

/// Identity op.
#[derive(Default)]
pub struct FrameId<N: Size<T>, T: Float> {
    _marker: PhantomData<(N, T)>,
}

impl<N: Size<T>, T: Float> FrameId<N, T> {
    pub fn new() -> FrameId<N, T> {
        FrameId::default()
    }
}

impl<N: Size<T>, T: Float> FrameUnop<N, T> for FrameId<N, T> {
    #[inline]
    fn unop(x: &Frame<T, N>) -> Frame<T, N> {
        x.clone()
    }
    fn propagate(x: Signal) -> Signal {
        x
    }
    #[inline]
    fn assign(_size: usize, _x: &mut [T]) {}
}

/// Apply a unary operation to output of contained node.
pub struct Unop<T, X, U> {
    _marker: PhantomData<T>,
    x: X,
    #[allow(dead_code)]
    u: U,
}

impl<T, X, U> Unop<T, X, U>
where
    T: Float,
    X: AudioNode<Sample = T>,
    U: FrameUnop<X::Outputs, T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    pub fn new(x: X, u: U) -> Self {
        let mut node = Unop {
            _marker: PhantomData,
            x,
            u,
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }
}

impl<T, X, U> AudioNode for Unop<T, X, U>
where
    T: Float,
    X: AudioNode<Sample = T>,
    U: FrameUnop<X::Outputs, T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    const ID: u64 = 4;
    type Sample = T;
    type Inputs = X::Inputs;
    type Outputs = X::Outputs;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        U::unop(&self.x.tick(input))
    }
    #[inline]
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.x.process(size, input, output);
        for i in 0..self.outputs() {
            U::assign(size, output[i]);
        }
    }
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.x.ping(probe, hash.hash(Self::ID))
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        let mut signal_x = self.x.route(input, frequency);
        for i in 0..Self::Outputs::USIZE {
            signal_x[i] = U::propagate(signal_x[i]);
        }
        signal_x
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        self.x.set(parameter, value);
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        self.x.get(parameter)
    }
}

/// Map any number of channels.
pub struct Map<T, M, I, O> {
    f: M,
    routing: Routing,
    _marker: PhantomData<(T, I, O)>,
}

impl<T, M, I, O> Map<T, M, I, O>
where
    T: Float,
    M: Fn(&Frame<T, I>) -> O,
    I: Size<T>,
    O: ConstantFrame<Sample = T>,
    O::Size: Size<T>,
{
    pub fn new(f: M, routing: Routing) -> Self {
        Self {
            f,
            routing,
            _marker: PhantomData::default(),
        }
    }
}

impl<T, M, I, O> AudioNode for Map<T, M, I, O>
where
    T: Float,
    M: Fn(&Frame<T, I>) -> O,
    I: Size<T>,
    O: ConstantFrame<Sample = T>,
    O::Size: Size<T>,
{
    const ID: u64 = 5;
    type Sample = T;
    type Inputs = I;
    type Outputs = O::Size;

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        (self.f)(input).convert()
    }

    fn route(&self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        self.routing.propagate(input, O::Size::USIZE)
    }
}

/// Pipe the output of `X` to `Y`.
pub struct Pipe<T, X, Y>
where
    T: Float,
{
    _marker: PhantomData<T>,
    x: X,
    y: Y,
    buffer: Buffer<T>,
}

impl<T, X, Y> Pipe<T, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T, Inputs = X::Outputs>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
    Y::Outputs: Size<T>,
{
    pub fn new(x: X, y: Y) -> Self {
        let mut node = Pipe {
            _marker: PhantomData,
            x,
            y,
            buffer: Buffer::new(),
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access the left node of the pipe.
    pub fn left_mut(&mut self) -> &mut X {
        &mut self.x
    }

    /// Access the left node of the pipe.
    pub fn left(&self) -> &X {
        &self.x
    }

    /// Access the right node of the pipe.
    pub fn right_mut(&mut self) -> &mut Y {
        &mut self.y
    }

    /// Access the right node of the pipe.
    pub fn right(&self) -> &Y {
        &self.y
    }
}

impl<T, X, Y> AudioNode for Pipe<T, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T, Inputs = X::Outputs>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
    Y::Outputs: Size<T>,
{
    const ID: u64 = 6;
    type Sample = T;
    type Inputs = X::Inputs;
    type Outputs = Y::Outputs;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
        self.y.reset(sample_rate);
    }

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        self.y.tick(&self.x.tick(input))
    }

    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.x
            .process(size, input, self.buffer.get_mut(self.x.outputs()));
        self.y
            .process(size, self.buffer.get_ref(self.y.inputs()), output);
    }

    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.y.ping(probe, self.x.ping(probe, hash.hash(Self::ID)))
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        self.y.route(&self.x.route(input, frequency), frequency)
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        self.x.set(parameter, value);
        self.y.set(parameter, value);
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        self.x.get(parameter).or_else(|| self.y.get(parameter))
    }
}

/// Stack `X` and `Y` in parallel.
pub struct Stack<T, X, Y> {
    _marker: PhantomData<T>,
    x: X,
    y: Y,
}

impl<T, X, Y> Stack<T, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Add<Y::Inputs>,
    X::Outputs: Size<T> + Add<Y::Outputs>,
    Y::Inputs: Size<T>,
    Y::Outputs: Size<T>,
    <X::Inputs as Add<Y::Inputs>>::Output: Size<T>,
    <X::Outputs as Add<Y::Outputs>>::Output: Size<T>,
{
    pub fn new(x: X, y: Y) -> Self {
        let mut node = Stack {
            _marker: PhantomData,
            x,
            y,
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access the left node of the stack.
    pub fn left_mut(&mut self) -> &mut X {
        &mut self.x
    }

    /// Access the left node of the stack.
    pub fn left(&self) -> &X {
        &self.x
    }

    /// Access the right node of the stack.
    pub fn right_mut(&mut self) -> &mut Y {
        &mut self.y
    }

    /// Access the right node of the stack.
    pub fn right(&self) -> &Y {
        &self.y
    }
}

impl<T, X, Y> AudioNode for Stack<T, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Add<Y::Inputs>,
    X::Outputs: Size<T> + Add<Y::Outputs>,
    Y::Inputs: Size<T>,
    Y::Outputs: Size<T>,
    <X::Inputs as Add<Y::Inputs>>::Output: Size<T>,
    <X::Outputs as Add<Y::Outputs>>::Output: Size<T>,
{
    const ID: u64 = 7;
    type Sample = T;
    type Inputs = Sum<X::Inputs, Y::Inputs>;
    type Outputs = Sum<X::Outputs, Y::Outputs>;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
        self.y.reset(sample_rate);
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let input_x = &input[..X::Inputs::USIZE];
        let input_y = &input[X::Inputs::USIZE..];
        let output_x = self.x.tick(input_x.into());
        let output_y = self.y.tick(input_y.into());
        Frame::generate(|i| {
            if i < X::Outputs::USIZE {
                output_x[i]
            } else {
                output_y[i - X::Outputs::USIZE]
            }
        })
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.x.process(
            size,
            &input[..X::Inputs::USIZE],
            &mut output[..X::Outputs::USIZE],
        );
        self.y.process(
            size,
            &input[X::Inputs::USIZE..],
            &mut output[X::Outputs::USIZE..],
        );
    }

    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.y.ping(probe, self.x.ping(probe, hash.hash(Self::ID)))
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        let mut signal_x = self.x.route(input, frequency);
        let signal_y = self.y.route(
            &copy_signal_frame(input, X::Inputs::USIZE, Y::Inputs::USIZE),
            frequency,
        );
        signal_x.resize(self.outputs(), Signal::Unknown);
        signal_x[X::Outputs::USIZE..Self::Outputs::USIZE]
            .copy_from_slice(&signal_y[0..Y::Outputs::USIZE]);
        signal_x
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        self.x.set(parameter, value);
        self.y.set(parameter, value);
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        self.x.get(parameter).or_else(|| self.y.get(parameter))
    }
}

/// Send the same input to `X` and `Y`. Concatenate outputs.
pub struct Branch<T, X, Y> {
    _marker: PhantomData<T>,
    x: X,
    y: Y,
}

impl<T, X, Y> Branch<T, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T, Inputs = X::Inputs>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T> + Add<Y::Outputs>,
    Y::Outputs: Size<T>,
    <X::Outputs as Add<Y::Outputs>>::Output: Size<T>,
{
    pub fn new(x: X, y: Y) -> Self {
        let mut node = Branch {
            _marker: PhantomData,
            x,
            y,
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access the left node of the branch.
    pub fn left_mut(&mut self) -> &mut X {
        &mut self.x
    }

    /// Access the left node of the branch.
    pub fn left(&self) -> &X {
        &self.x
    }

    /// Access the right node of the branch.
    pub fn right_mut(&mut self) -> &mut Y {
        &mut self.y
    }

    /// Access the right node of the branch.
    pub fn right(&self) -> &Y {
        &self.y
    }
}

impl<T, X, Y> AudioNode for Branch<T, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T, Inputs = X::Inputs>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T> + Add<Y::Outputs>,
    Y::Outputs: Size<T>,
    <X::Outputs as Add<Y::Outputs>>::Output: Size<T>,
{
    const ID: u64 = 8;
    type Sample = T;
    type Inputs = X::Inputs;
    type Outputs = Sum<X::Outputs, Y::Outputs>;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
        self.y.reset(sample_rate);
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let output_x = self.x.tick(input);
        let output_y = self.y.tick(input);
        Frame::generate(|i| {
            if i < X::Outputs::USIZE {
                output_x[i]
            } else {
                output_y[i - X::Outputs::USIZE]
            }
        })
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.x
            .process(size, input, &mut output[..X::Outputs::USIZE]);
        self.y
            .process(size, input, &mut output[X::Outputs::USIZE..]);
    }
    #[inline]
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.y.ping(probe, self.x.ping(probe, hash.hash(Self::ID)))
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        let mut signal_x = self.x.route(input, frequency);
        let signal_y = self.y.route(input, frequency);
        signal_x.resize(self.outputs(), Signal::Unknown);
        signal_x[X::Outputs::USIZE..Self::Outputs::USIZE]
            .copy_from_slice(&signal_y[0..Y::Outputs::USIZE]);
        signal_x
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        self.x.set(parameter, value);
        self.y.set(parameter, value);
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        self.x.get(parameter).or_else(|| self.y.get(parameter))
    }
}

/// Mix together `X` and `Y` sourcing from the same inputs.
pub struct Bus<T, X, Y>
where
    T: Float,
{
    _marker: PhantomData<T>,
    x: X,
    y: Y,
    buffer: Buffer<T>,
}

impl<T, X, Y> Bus<T, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T, Inputs = X::Inputs, Outputs = X::Outputs>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
    Y::Inputs: Size<T>,
    Y::Outputs: Size<T>,
{
    pub fn new(x: X, y: Y) -> Self {
        let mut node = Bus {
            _marker: PhantomData,
            x,
            y,
            buffer: Buffer::new(),
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access the left node of the bus.
    pub fn left_mut(&mut self) -> &mut X {
        &mut self.x
    }

    /// Access the left node of the bus.
    pub fn left(&self) -> &X {
        &self.x
    }

    /// Access the right node of the bus.
    pub fn right_mut(&mut self) -> &mut Y {
        &mut self.y
    }

    /// Access the right node of the bus.
    pub fn right(&self) -> &Y {
        &self.y
    }
}

impl<T, X, Y> AudioNode for Bus<T, X, Y>
where
    T: Float,
    X: AudioNode<Sample = T>,
    Y: AudioNode<Sample = T, Inputs = X::Inputs, Outputs = X::Outputs>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
    Y::Inputs: Size<T>,
    Y::Outputs: Size<T>,
{
    const ID: u64 = 10;
    type Sample = T;
    type Inputs = X::Inputs;
    type Outputs = X::Outputs;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
        self.y.reset(sample_rate);
    }

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let output_x = self.x.tick(input);
        let output_y = self.y.tick(input);
        output_x + output_y
    }

    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.x.process(size, input, output);
        self.y
            .process(size, input, self.buffer.get_mut(self.outputs()));
        for channel in 0..self.outputs() {
            let src = self.buffer.at(channel);
            let dst = &mut output[channel];
            for i in 0..size {
                dst[i] += src[i];
            }
        }
    }

    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.y.ping(probe, self.x.ping(probe, hash.hash(Self::ID)))
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        let mut signal_x = self.x.route(input, frequency);
        let signal_y = self.y.route(input, frequency);
        for i in 0..Self::Outputs::USIZE {
            signal_x[i] = signal_x[i].combine_linear(signal_y[i], 0.0, |x, y| x + y, |x, y| x + y);
        }
        signal_x
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        self.x.set(parameter, value);
        self.y.set(parameter, value);
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        self.x.get(parameter).or_else(|| self.y.get(parameter))
    }
}

/// Pass through inputs without matching outputs.
/// Adjusts output arity to match input arity, adapting a filter to a pipeline.
pub struct Thru<X: AudioNode> {
    x: X,
    buffer: Buffer<X::Sample>,
}

impl<X: AudioNode> Thru<X> {
    pub fn new(x: X) -> Self {
        let mut node = Thru {
            x,
            buffer: Buffer::new(),
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }
}

impl<X: AudioNode> AudioNode for Thru<X> {
    const ID: u64 = 12;
    type Sample = X::Sample;
    type Inputs = X::Inputs;
    type Outputs = X::Inputs;

    #[inline]
    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
    }

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let output = self.x.tick(input);
        Frame::generate(|channel| {
            if channel < X::Outputs::USIZE {
                output[channel]
            } else {
                input[channel]
            }
        })
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        if X::Inputs::USIZE == 0 {
            // This is an empty node.
            return;
        }
        if X::Inputs::USIZE < X::Outputs::USIZE {
            // The intermediate buffer is only used in this "degenerate" case where
            // we are not passing through inputs - we are cutting out some of them.
            self.x
                .process(size, input, self.buffer.get_mut(X::Inputs::USIZE));
            for channel in 0..X::Inputs::USIZE {
                output[channel][..size].clone_from_slice(&self.buffer.at(channel)[..size]);
            }
        } else {
            self.x
                .process(size, input, &mut output[..X::Outputs::USIZE]);
            for channel in X::Outputs::USIZE..X::Inputs::USIZE {
                output[channel][..size].clone_from_slice(&input[channel][..size]);
            }
        }
    }

    #[inline]
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.x.ping(probe, hash.hash(Self::ID))
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        let mut output = self.x.route(input, frequency);
        output[X::Outputs::USIZE..Self::Outputs::USIZE]
            .copy_from_slice(&input[X::Outputs::USIZE..Self::Outputs::USIZE]);
        output
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        self.x.set(parameter, value);
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        self.x.get(parameter)
    }
}

/// Mix together a bunch of similar nodes sourcing from the same inputs.
pub struct MultiBus<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    _marker: PhantomData<T>,
    x: Frame<X, N>,
    buffer: Buffer<T>,
}

impl<N, T, X> MultiBus<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    pub fn new(x: Frame<X, N>) -> Self {
        let mut node = MultiBus {
            _marker: PhantomData::default(),
            x,
            buffer: Buffer::new(),
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access a contained node.
    pub fn node_mut(&mut self, index: usize) -> &mut X {
        &mut self.x[index]
    }

    /// Access a contained node.
    pub fn node(&self, index: usize) -> &X {
        &self.x[index]
    }
}

impl<N, T, X> AudioNode for MultiBus<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    const ID: u64 = 28;
    type Sample = T;
    type Inputs = X::Inputs;
    type Outputs = X::Outputs;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.iter_mut().for_each(|node| node.reset(sample_rate));
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        self.x
            .iter_mut()
            .fold(Frame::splat(T::zero()), |acc, x| acc + x.tick(input))
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.x[0].process(size, input, output);
        for i in 1..N::USIZE {
            self.x[i].process(size, input, self.buffer.get_mut(X::Outputs::USIZE));
            for channel in 0..X::Outputs::USIZE {
                let src = self.buffer.at(channel);
                let dst = &mut output[channel];
                for j in 0..size {
                    dst[j] += src[j];
                }
            }
        }
    }
    #[inline]
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        let mut hash = hash.hash(Self::ID);
        for x in &mut self.x {
            hash = x.ping(probe, hash);
        }
        hash
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        if self.x.is_empty() {
            return new_signal_frame(self.outputs());
        }
        let mut output = self.x[0].route(input, frequency);
        for i in 1..self.x.len() {
            let output_i = self.x[i].route(input, frequency);
            for channel in 0..Self::Outputs::USIZE {
                output[channel] = output[channel].combine_linear(
                    output_i[channel],
                    0.0,
                    |x, y| x + y,
                    |x, y| x + y,
                );
            }
        }
        output
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        for x in self.x.iter_mut() {
            x.set(parameter, value);
        }
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        for x in self.x.iter() {
            if let Some(value) = x.get(parameter) {
                return Some(value);
            }
        }
        None
    }
}

/// Stack a bunch of similar nodes in parallel.
pub struct MultiStack<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
{
    _marker: PhantomData<(N, T)>,
    x: Frame<X, N>,
}

impl<N, T, X> MultiStack<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
{
    pub fn new(x: Frame<X, N>) -> Self {
        let mut node = MultiStack {
            _marker: PhantomData,
            x,
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access a contained node.
    pub fn node_mut(&mut self, index: usize) -> &mut X {
        &mut self.x[index]
    }

    /// Access a contained node.
    pub fn node(&self, index: usize) -> &X {
        &self.x[index]
    }
}

impl<N, T, X> AudioNode for MultiStack<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
{
    const ID: u64 = 30;
    type Sample = T;
    type Inputs = Prod<X::Inputs, N>;
    type Outputs = Prod<X::Outputs, N>;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.iter_mut().for_each(|node| node.reset(sample_rate));
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let mut output: Frame<Self::Sample, Self::Outputs> = Frame::splat(T::zero());
        for (i, node) in self.x.iter_mut().enumerate() {
            let node_input = &input[i * X::Inputs::USIZE..(i + 1) * X::Inputs::USIZE];
            let node_output = node.tick(node_input.into());
            output[i * X::Outputs::USIZE..(i + 1) * X::Outputs::USIZE]
                .copy_from_slice(node_output.as_slice());
        }
        output
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        let mut in_channel = 0;
        let mut out_channel = 0;
        for i in 0..N::USIZE {
            let next_in_channel = in_channel + X::Inputs::USIZE;
            let next_out_channel = out_channel + X::Outputs::USIZE;
            self.x[i].process(
                size,
                &input[in_channel..next_in_channel],
                &mut output[out_channel..next_out_channel],
            );
            in_channel = next_in_channel;
            out_channel = next_out_channel;
        }
    }
    #[inline]
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        let mut hash = hash.hash(Self::ID);
        for x in self.x.iter_mut() {
            hash = x.ping(probe, hash);
        }
        hash
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        if self.x.is_empty() {
            return new_signal_frame(self.outputs());
        }
        let mut output = self.x[0].route(input, frequency);
        output.resize(self.outputs(), Signal::Unknown);
        for i in 1..N::USIZE {
            let output_i = self.x[i].route(
                &copy_signal_frame(input, i * X::Inputs::USIZE, X::Inputs::USIZE),
                frequency,
            );
            output[i * X::Outputs::USIZE..(i + 1) * X::Outputs::USIZE]
                .copy_from_slice(&output_i[0..X::Outputs::USIZE]);
        }
        output
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        for x in self.x.iter_mut() {
            x.set(parameter, value);
        }
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        for x in self.x.iter() {
            if let Some(value) = x.get(parameter) {
                return Some(value);
            }
        }
        None
    }
}

/// Combine outputs of a bunch of similar nodes with a binary operation.
/// Inputs are disjoint.
/// Outputs are combined channel-wise.
pub struct Reduce<N, T, X, B>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    B: FrameBinop<X::Outputs, T>,
{
    x: Frame<X, N>,
    #[allow(dead_code)]
    b: B,
    buffer: Buffer<T>,
    _marker: PhantomData<T>,
}

impl<N, T, X, B> Reduce<N, T, X, B>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    B: FrameBinop<X::Outputs, T>,
{
    pub fn new(x: Frame<X, N>, b: B) -> Self {
        let mut node = Reduce {
            x,
            b,
            buffer: Buffer::new(),
            _marker: PhantomData,
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access a contained node.
    pub fn node_mut(&mut self, index: usize) -> &mut X {
        &mut self.x[index]
    }

    /// Access a contained node.
    pub fn node(&self, index: usize) -> &X {
        &self.x[index]
    }
}

impl<N, T, X, B> AudioNode for Reduce<N, T, X, B>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T> + Mul<N>,
    X::Outputs: Size<T>,
    <X::Inputs as Mul<N>>::Output: Size<T>,
    B: FrameBinop<X::Outputs, T>,
{
    const ID: u64 = 32;
    type Sample = T;
    type Inputs = Prod<X::Inputs, N>;
    type Outputs = X::Outputs;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.iter_mut().for_each(|node| node.reset(sample_rate));
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let mut output: Frame<Self::Sample, Self::Outputs> = Frame::splat(T::zero());
        for (i, node) in self.x.iter_mut().enumerate() {
            let node_input = &input[i * X::Inputs::USIZE..(i + 1) * X::Inputs::USIZE];
            let node_output = node.tick(node_input.into());
            if i > 0 {
                output = B::binop(&output, &node_output);
            } else {
                output = node_output;
            }
        }
        output
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        self.x[0].process(size, &input[..X::Inputs::USIZE], output);
        let mut in_channel = X::Inputs::USIZE;
        for i in 1..N::USIZE {
            let next_in_channel = in_channel + X::Inputs::USIZE;
            self.x[i].process(
                size,
                &input[in_channel..next_in_channel],
                self.buffer.get_mut(X::Outputs::USIZE),
            );
            in_channel = next_in_channel;
            for channel in 0..X::Outputs::USIZE {
                B::assign(size, output[channel], self.buffer.at(channel));
            }
        }
    }
    #[inline]
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        let mut hash = hash.hash(Self::ID);
        for x in self.x.iter_mut() {
            hash = x.ping(probe, hash);
        }
        hash
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        if self.x.is_empty() {
            return new_signal_frame(self.outputs());
        }
        let mut output = self.x[0].route(input, frequency);
        for j in 1..self.x.len() {
            let output_j = self.x[j].route(
                &copy_signal_frame(input, j * X::Inputs::USIZE, X::Inputs::USIZE),
                frequency,
            );
            for i in 0..Self::Outputs::USIZE {
                output[i] = B::propagate(output[i], output_j[i]);
            }
        }
        output
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        for x in self.x.iter_mut() {
            x.set(parameter, value);
        }
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        for x in self.x.iter() {
            if let Some(value) = x.get(parameter) {
                return Some(value);
            }
        }
        None
    }
}

/// Branch into a bunch of similar nodes in parallel.
pub struct MultiBranch<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
{
    _marker: PhantomData<T>,
    x: Frame<X, N>,
}

impl<N, T, X> MultiBranch<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
{
    pub fn new(x: Frame<X, N>) -> Self {
        let mut node = MultiBranch {
            _marker: PhantomData,
            x,
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access a contained node.
    pub fn node_mut(&mut self, index: usize) -> &mut X {
        &mut self.x[index]
    }

    /// Access a contained node.
    pub fn node(&self, index: usize) -> &X {
        &self.x[index]
    }
}

impl<N, T, X> AudioNode for MultiBranch<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T> + Mul<N>,
    <X::Outputs as Mul<N>>::Output: Size<T>,
{
    const ID: u64 = 33;
    type Sample = T;
    type Inputs = X::Inputs;
    type Outputs = Prod<X::Outputs, N>;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.iter_mut().for_each(|node| node.reset(sample_rate));
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let mut output: Frame<Self::Sample, Self::Outputs> = Frame::splat(T::zero());
        for (i, node) in self.x.iter_mut().enumerate() {
            let node_output = node.tick(input);
            output[i * X::Outputs::USIZE..(i + 1) * X::Outputs::USIZE]
                .copy_from_slice(node_output.as_slice());
        }
        output
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        let mut out_channel = 0;
        for i in 0..N::USIZE {
            let next_out_channel = out_channel + X::Outputs::USIZE;
            self.x[i].process(size, input, &mut output[out_channel..next_out_channel]);
            out_channel = next_out_channel;
        }
    }

    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        let mut hash = hash.hash(Self::ID);
        for x in self.x.iter_mut() {
            hash = x.ping(probe, hash);
        }
        hash
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        if self.x.is_empty() {
            return new_signal_frame(self.outputs());
        }
        let mut output = self.x[0].route(input, frequency);
        output.resize(self.outputs(), Signal::Unknown);
        for i in 1..N::USIZE {
            let output_i = self.x[i].route(input, frequency);
            output[i * X::Outputs::USIZE..(i + 1) * X::Outputs::USIZE]
                .copy_from_slice(&output_i[0..X::Outputs::USIZE]);
        }
        output
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        for x in self.x.iter_mut() {
            x.set(parameter, value);
        }
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        for x in self.x.iter() {
            if let Some(value) = x.get(parameter) {
                return Some(value);
            }
        }
        None
    }
}

/// Chain together a bunch of similar nodes.
pub struct Chain<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    x: Frame<X, N>,
    buffer_a: Buffer<T>,
    buffer_b: Buffer<T>,
    _marker: PhantomData<T>,
}

impl<N, T, X> Chain<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    pub fn new(x: Frame<X, N>) -> Self {
        let mut node = Chain {
            x,
            buffer_a: Buffer::new(),
            buffer_b: Buffer::new(),
            _marker: PhantomData,
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }

    /// Access a contained node.
    pub fn node_mut(&mut self, index: usize) -> &mut X {
        &mut self.x[index]
    }

    /// Access a contained node.
    pub fn node(&self, index: usize) -> &X {
        &self.x[index]
    }
}

impl<N, T, X> AudioNode for Chain<N, T, X>
where
    N: Size<T>,
    N: Size<X>,
    T: Float,
    X: AudioNode<Sample = T>,
    X::Inputs: Size<T>,
    X::Outputs: Size<T>,
{
    const ID: u64 = 32;
    type Sample = T;
    // TODO. We'd like to require that X::Inputs equals X::Outputs but
    // I don't know how to write such a trait bound.
    type Inputs = X::Inputs;
    type Outputs = X::Outputs;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.iter_mut().for_each(|node| node.reset(sample_rate));
    }
    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let mut output = self.x[0].tick(input);
        for i in 1..N::USIZE {
            output = self.x[i].tick(&Frame::generate(|i| output[i]));
        }
        output
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        if N::USIZE == 1 {
            self.x[0].process(size, input, output);
        } else {
            self.x[0].process(size, input, self.buffer_a.get_mut(X::Outputs::USIZE));
            for i in 1..N::USIZE - 1 {
                if i & 1 > 0 {
                    self.x[i].process(
                        size,
                        self.buffer_a.get_ref(X::Outputs::USIZE),
                        self.buffer_b.get_mut(X::Outputs::USIZE),
                    );
                } else {
                    self.x[i].process(
                        size,
                        self.buffer_b.get_ref(X::Outputs::USIZE),
                        self.buffer_a.get_mut(X::Outputs::USIZE),
                    );
                }
            }
            if (N::USIZE - 1) & 1 > 0 {
                self.x[N::USIZE - 1].process(
                    size,
                    self.buffer_a.get_ref(X::Outputs::USIZE),
                    output,
                );
            } else {
                self.x[N::USIZE - 1].process(
                    size,
                    self.buffer_b.get_ref(X::Outputs::USIZE),
                    output,
                );
            }
        }
    }

    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        let mut hash = hash.hash(Self::ID);
        for x in self.x.iter_mut() {
            hash = x.ping(probe, hash);
        }
        hash
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        let mut output = self.x[0].route(input, frequency);
        for i in 1..self.x.len() {
            output = self.x[i].route(&output, frequency);
        }
        output
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        for x in self.x.iter_mut() {
            x.set(parameter, value);
        }
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        for x in self.x.iter() {
            if let Some(value) = x.get(parameter) {
                return Some(value);
            }
        }
        None
    }
}

/// Swap stereo channels.
#[derive(Default)]
pub struct Swap<T> {
    _marker: PhantomData<T>,
}

impl<T: Float> Swap<T> {
    pub fn new() -> Self {
        Swap::default()
    }
}

impl<T: Float> AudioNode for Swap<T> {
    const ID: u64 = 45;
    type Sample = T;
    type Inputs = U2;
    type Outputs = U2;

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        [input[1], input[0]].into()
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        output[0][..size].clone_from_slice(&input[1][..size]);
        output[1][..size].clone_from_slice(&input[0][..size]);
    }
    fn route(&self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        let mut output = new_signal_frame(self.outputs());
        output[0] = input[1];
        output[1] = input[0];
        output
    }
}
